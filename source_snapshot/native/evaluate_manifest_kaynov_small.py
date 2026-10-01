#!/usr/bin/env python3
"""Unified fixed-demand evaluation for the manifest K0-K6/S1-S2 matrix.

The script evaluates project DRL actors and heuristic policies. Kaynov-source
PPO checkpoints are handled by ``evaluate_manifest_source_ppo.py`` because
they require the legacy Python 3.9/Ray environment.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from Benchmark import (
    OWMRInventoryEnv,
    auto_hyperparameter,
    decomposition_aggregation_distribution_aware,
    env_input_initialize,
    generate_env,
    heuristic_order_qty,
    recursive_heuristic_tail_fixed,
    relaxation_based_heuristic_distribution_aware_capped,
)
from dataset import data_kaynov, data_small_rvi
from demand_utils import sample_demand
from elegantrl.agents.AgentSAC import AgentANSAC, AgentLagANSAC, AgentPenaltyANSAC
from elegantrl.train.config import Config
from evaluation_statistics import mean_se_median
from main_DRL_agents import get_env_args
from run_kaynov_single_combo import _raw_kaynov_mean


ROOT = Path(__file__).resolve().parent
MANIFEST = Path(
    os.environ.get(
        "OWMR_RESULT_MANIFEST",
        ROOT / "outputs/valid_result_manifest_13x24.csv",
    )
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs/evaluation/manifest_13x9_unified_20260728"
CASES = [f"K{i}" for i in range(7)] + ["S1", "S2"]
HEURISTICS = {"DA", "RB", "RO"}
AN_FAMILY = {"AN-SAC", "Penalty-AN-SAC", "Lag-AN-SAC"}
SOURCE_PPO = "PPO (Kaynov source)"
DEFAULT_SEED = 20260728


def json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_manifest_rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as file:
        return [row for row in csv.DictReader(file) if row["case"] in CASES]


def case_data(case: str):
    if case.startswith("K"):
        return data_kaynov()[int(case[1:])]
    return data_small_rvi()[int(case[1:]) - 1]


def case_seed(case: str, seed: int) -> int:
    return int(seed) + CASES.index(case) * 1009


def demand_bank_path(output_dir: Path, case: str) -> Path:
    return output_dir / "demand_banks" / f"{case}.npz"


def build_demand_bank(case: str, episodes: int, episode_steps: int, seed: int) -> dict[int, np.ndarray]:
    data = case_data(case)
    info = env_input_initialize(*data)[0]
    rng = np.random.default_rng(case_seed(case, seed))
    bank = {}
    for rdc in info.rdc_dict:
        values = np.asarray(
            sample_demand(
                info.demand_feature[rdc],
                size=(episodes, episode_steps),
                random_state=rng,
            ),
            dtype=np.float64,
        )
        # The project stationary environment intentionally makes period 0 demand-free.
        values[:, 0] = 0.0
        bank[int(rdc)] = values
    return bank


def save_demand_bank(path: Path, bank: dict[int, np.ndarray], case: str, seed: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"rdc_{key}": value for key, value in bank.items()})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    first = next(iter(bank.values()))
    return {
        "case": case,
        "seed": case_seed(case, seed),
        "episodes": int(first.shape[0]),
        "episode_steps": int(first.shape[1]),
        "rdcs": sorted(bank),
        "sha256": digest,
        "path": display_path(path),
    }


def load_demand_bank(path: Path) -> dict[int, np.ndarray]:
    with np.load(path) as data:
        return {int(key.split("_", 1)[1]): np.asarray(data[key], dtype=float) for key in data.files}


def split_top_level(text: str) -> list[str]:
    parts, start, depth, quote, escape = [], 0, 0, None, False
    for index, char in enumerate(text):
        if quote is not None:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def parse_value(text: str):
    text = text.strip()
    if text == "None":
        return None
    if text == "True":
        return True
    if text == "False":
        return False
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def parse_training_line(line: str) -> dict[str, Any]:
    if line.startswith(
        (
            "Training instance=",
            "Training kaynov_index=",
            "Training table case ",
            "Training dataset=",
        )
    ):
        line = line[len("Training "):]
    else:
        raise ValueError(f"Not a training protocol line: {line[:80]}")
    result = {}
    for part in split_top_level(line):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        result[key.strip()] = parse_value(value)
    return result


def parse_action_space_line(line: str) -> dict[str, Any]:
    if not line.startswith("Action-space config:"):
        raise ValueError(f"Not an action-space line: {line[:80]}")
    parsed = {}
    for part in split_top_level(line.split(":", 1)[1].strip()):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        parsed[key.strip()] = parse_value(value)
    aliases = {
        "profile": "table15_action_space_profile",
        "rdc_min": "rdc_bsl_min_override",
        "cdc_min": "cdc_bsl_min_override",
        "rdc_max": "rdc_bsl_max_override",
        "cdc_max": "cdc_bsl_max_override",
    }
    return {aliases.get(key, key): value for key, value in parsed.items()}


def protocol_alias(cwd: Path) -> str:
    parts = cwd.as_posix().replace("\\", "/").rstrip("/").split("/")
    return "alias::" + "/".join(parts[-2:])


def index_training_logs() -> dict[str, tuple[Path, dict[str, Any]]]:
    index: dict[str, tuple[Path, dict[str, Any]]] = {}
    # Transfer experiments keep their logs below outputs/transfer rather than
    # outputs/logs, so scan the complete output tree when rebuilding protocols.
    for log_path in (ROOT / "outputs").glob("**/*.log"):
        cwd = None
        protocol = None
        try:
            with log_path.open(encoding="utf-8", errors="ignore") as file:
                for line_no, raw in enumerate(file):
                    line = raw.strip()
                    if line.startswith(
                        (
                            "Training instance=",
                            "Training kaynov_index=",
                            "Training table case ",
                            "Training dataset=",
                        )
                    ):
                        protocol = parse_training_line(line)
                        if protocol.get("cwd"):
                            cwd = Path(str(protocol["cwd"]).replace("\\", "/"))
                    elif line.startswith("Action-space config:") and protocol is not None:
                        protocol.update(parse_action_space_line(line))
                    elif line.startswith("cwd="):
                        cwd = Path(line[4:].strip())
                    if line_no >= 100 and cwd is not None and protocol is not None:
                        break
        except OSError:
            continue
        if cwd is None or protocol is None:
            continue
        if not cwd.is_absolute():
            cwd = ROOT / "outputs/train" / cwd
        index[str(cwd.resolve())] = (log_path, protocol)
        index[protocol_alias(cwd)] = (log_path, protocol)
    return index


def protocol_index_path(output_dir: Path) -> Path:
    return output_dir / "training_protocol_index.json"


def prepare_protocol_index(output_dir: Path) -> dict[str, Any]:
    scanned = index_training_logs()
    selected: dict[str, Any] = {}
    missing: list[str] = []
    source_checkpoints = 0
    project_actors = 0
    for row in load_manifest_rows():
        if row["status"] == "not_applicable" or row["benchmark"] in HEURISTICS:
            continue
        checkpoint = (ROOT / row["checkpoint_or_policy"]).resolve()
        if not checkpoint.exists():
            missing.append(f"missing checkpoint: {row['benchmark']} {row['case']} {checkpoint}")
            continue
        if row["benchmark"] == SOURCE_PPO:
            source_checkpoints += 1
            continue
        project_actors += 1
        key = str(checkpoint.parent)
        alias = protocol_alias(checkpoint.parent)
        entry = scanned.get(key) or scanned.get(alias)
        if entry is None:
            missing.append(f"missing training protocol: {row['benchmark']} {row['case']} {key}")
            continue
        log_path, protocol = entry
        selected[key] = {
            "training_log": display_path(log_path),
            "protocol": json_safe(protocol),
        }
    if missing:
        raise RuntimeError("Manifest protocol audit failed:\n" + "\n".join(missing))
    payload = {
        "project_actors": project_actors,
        "source_checkpoints": source_checkpoints,
        "entries": selected,
    }
    protocol_index_path(output_dir).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def load_training_protocol(output_dir: Path, actor_path: Path) -> tuple[Path, dict[str, Any]]:
    path = protocol_index_path(output_dir)
    if not path.exists():
        prepare_protocol_index(output_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload["entries"][str(actor_path.parent.resolve())]
    return ROOT / entry["training_log"], entry["protocol"]


def effective_override(protocol: dict[str, Any], explicit_key: str, adaptive_key: str):
    explicit = protocol.get(explicit_key)
    value = explicit if explicit is not None else protocol.get(adaptive_key)
    if isinstance(value, dict):
        # JSON object keys are strings. RDC identifiers in the environment are
        # integers, so preserving the string keys silently disables per-RDC
        # action-bound overrides during manifest evaluation.
        return {int(key): item for key, item in value.items()}
    return value


def apply_kaynov_compatible_inventory(data):
    rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, _, _ = data
    rdc_inventory = {
        rdc: np.hstack((int(_raw_kaynov_mean(demand_feature[rdc]) * lead), np.zeros(lead)))
        for rdc, lead in rdc_dict.items()
    }
    cdc_start = int(item_lead_time * sum(_raw_kaynov_mean(demand_feature[k]) for k in rdc_dict))
    cdc_inventory = np.hstack((cdc_start, np.zeros(item_lead_time)))
    return rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, rdc_inventory, cdc_inventory


def build_project_env(case: str, protocol: dict[str, Any], bank: dict[int, np.ndarray]):
    data = case_data(case)
    if protocol.get("kaynov_k5_compatible", False):
        data = apply_kaynov_compatible_inventory(data)
    info = env_input_initialize(*data)[0]
    _, _, _, holding_weight, penalty_weight, _, _ = auto_hyperparameter(info, len(next(iter(bank.values()))))
    is_partial = case in {"K5", "K6"}
    input_args = {
        "all_demand_list": bank,
        "holding_weight": holding_weight,
        "penalty_weight": penalty_weight,
        "cost_plus": 0.0,
        "info_input": info,
        "if_generate_demand": False,
        "seed": DEFAULT_SEED,
        "rho": float(protocol.get("rho", 0.8 if is_partial else 1.0)),
        "allocation_rule": protocol.get("allocation_rule", "prioritized_greedy"),
        "demand_fulfillment": protocol.get(
            "demand_fulfillment",
            "kaynov_partial_backlogging" if is_partial else "partial_backorder",
        ),
        "emergency_shipment_prob": protocol.get("emergency_shipment_prob"),
        "rdc_bsl_min_override": effective_override(protocol, "rdc_bsl_min_override", "adaptive_rdc_bsl_min"),
        "cdc_bsl_min_override": effective_override(protocol, "cdc_bsl_min_override", "adaptive_cdc_bsl_min"),
        "rdc_bsl_max_override": effective_override(protocol, "rdc_bsl_max_override", "adaptive_rdc_bsl_max"),
        "cdc_bsl_max_override": effective_override(protocol, "cdc_bsl_max_override", "adaptive_cdc_bsl_max"),
    }
    if input_args["emergency_shipment_prob"] is None:
        input_args["emergency_shipment_prob"] = input_args["rho"]
    env = generate_env(OWMRInventoryEnv, get_env_args(input_args))
    env.total_periods = next(iter(bank.values())).shape[1]
    env.episode = -1
    return env, info, input_args


def load_actor(path: Path) -> torch.nn.Module:
    try:
        actor = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        actor = torch.load(path, map_location="cpu")
    if not isinstance(actor, torch.nn.Module):
        raise TypeError(f"Expected torch module at {path}, got {type(actor).__name__}")
    # Older serialized actors predate the optional finite-horizon action-mask
    # attributes now referenced by ActorBase.forward().  Their intended
    # stationary behavior is an all-active action vector.
    if not hasattr(actor, "action_mask_state_start"):
        actor.action_mask_state_start = None
    if not hasattr(actor, "action_mask_dim"):
        actor.action_mask_dim = 0
    if not hasattr(actor, "inactive_action_value"):
        actor.inactive_action_value = 0.0
    return actor.eval()


def build_an_agent(
    benchmark: str,
    actor: torch.nn.Module,
    env,
    input_args: dict[str, Any],
    protocol: dict[str, Any],
):
    class_map = {
        "AN-SAC": AgentANSAC,
        "Penalty-AN-SAC": AgentPenaltyANSAC,
        "Lag-AN-SAC": AgentLagANSAC,
    }
    # Recreate the complete environment contract used by training.  The AN
    # layers need info_input and demand metadata when rebuilding their action
    # normalization tensors; a partial Config is not sufficient.
    env_args = get_env_args(input_args.copy())
    env_args["state_dim"] = env.state_dim
    env_args["action_dim"] = env.action_dim
    args = Config(class_map[benchmark], env_class=OWMRInventoryEnv, env_args=env_args)
    args.gpu_id = -1
    args.device = "cpu"
    args.net_dims = tuple(protocol.get("net_dims", (64, 32)))
    for name in (
        "an_penalty_coef", "an_penalty_power", "an_violation_eps", "an_positive_tau",
        "lag_lambda_init", "lag_lambda_lr", "lag_delta", "lag_lambda_max", "lag_update_interval",
    ):
        if protocol.get(name) is not None:
            setattr(args, name, protocol[name])
    agent = class_map[benchmark](args.net_dims, env.state_dim, env.action_dim, gpu_id=-1, args=args)
    agent.act = actor
    return agent


def evaluate_project_actor(
    benchmark: str,
    actor_path: Path,
    env,
    input_args: dict[str, Any],
    protocol: dict[str, Any],
    episodes: int,
    episode_steps: int,
    warmup: int,
) -> np.ndarray:
    actor = load_actor(actor_path)
    if getattr(actor, "state_dim", env.state_dim) != env.state_dim:
        raise ValueError(f"{benchmark} actor state_dim does not match {env.state_dim}")
    an_agent = (
        build_an_agent(benchmark, actor, env, input_args, protocol)
        if benchmark in AN_FAMILY else None
    )
    costs = np.empty((episodes, episode_steps - warmup), dtype=np.float64)
    with torch.no_grad():
        for episode in range(episodes):
            state = env.reset()
            for period in range(episode_steps):
                tensor_state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                raw_action = actor(tensor_state)
                if isinstance(raw_action, tuple):
                    raw_action = raw_action[0]
                if an_agent is not None:
                    _, feasible_action = an_agent.soft_lagrangian_reallocation(tensor_state, raw_action)
                    state, _, done, info_out = env.step_feasible_action(feasible_action[0].cpu().numpy())
                else:
                    state, _, done, info_out = env.step(raw_action[0].cpu().numpy())
                if period >= warmup:
                    costs[episode, period - warmup] = sum(float(value) for value in info_out[0].values())
                if done and period + 1 != episode_steps:
                    raise RuntimeError(f"{benchmark} {episode=} ended at {period + 1}")
    return costs


def untruncated_small_ro_data(data):
    rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, rdc_inventory, cdc_inventory = data
    policy_demand = {}
    for rdc, spec in demand_feature.items():
        if isinstance(spec, tuple) and str(spec[0]).lower() == "truncated_poisson":
            policy_demand[rdc] = ("poisson", float(spec[1]))
        else:
            policy_demand[rdc] = spec
    return rdc_dict, policy_demand, holding_cost, backorder_cost, item_lead_time, rdc_inventory, cdc_inventory


def compute_heuristic_policy(benchmark: str, case: str):
    data = case_data(case)
    policy_data = untruncated_small_ro_data(data) if benchmark == "RO" and case.startswith("S") else data
    info = env_input_initialize(*policy_data)[0]
    if benchmark == "DA":
        rdc_bsl, cdc_bsl = decomposition_aggregation_distribution_aware(info)
        return rdc_bsl, float(cdc_bsl)
    if benchmark == "RB":
        _, rdc_bsl, cdc_bsl = relaxation_based_heuristic_distribution_aware_capped(info)
        return rdc_bsl, float(cdc_bsl)
    rdc_bsl, _, cdc_bsl = recursive_heuristic_tail_fixed(info)
    return rdc_bsl, float(cdc_bsl)


def evaluate_heuristic(
    benchmark: str,
    case: str,
    bank: dict[int, np.ndarray],
    episodes: int,
    episode_steps: int,
    warmup: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    protocol = {
        "allocation_rule": "proportion",
        "rho": 0.8 if case in {"K5", "K6"} else 1.0,
        "demand_fulfillment": "kaynov_partial_backlogging" if case in {"K5", "K6"} else "partial_backorder",
        "emergency_shipment_prob": 0.8 if case in {"K5", "K6"} else 1.0,
    }
    env, info, _ = build_project_env(case, protocol, bank)
    env.method = "heu"
    rdc_bsl, cdc_bsl = compute_heuristic_policy(benchmark, case)
    method_alias = {"DA": "DA_distribution_aware", "RB": "RB_distribution_aware_capped", "RO": "RO_tail_fixed"}[benchmark]
    costs = np.empty((episodes, episode_steps - warmup), dtype=np.float64)
    for episode in range(episodes):
        env.reset()
        for period in range(episode_steps):
            action = heuristic_order_qty(env, info, rdc_bsl, cdc_bsl, method=method_alias)
            _, _, done, info_out = env.step(action)
            if period >= warmup:
                costs[episode, period - warmup] = sum(float(value) for value in info_out[0].values())
            if done and period + 1 != episode_steps:
                raise RuntimeError(f"{benchmark} {case} ended at {period + 1}")
    return costs, {"rdc_bsl": json_safe(rdc_bsl), "cdc_bsl": cdc_bsl, **protocol}


def summarize_costs(costs: np.ndarray) -> dict[str, float]:
    episode_means = costs.mean(axis=1)
    mean, se, median = mean_se_median(episode_means)
    return {"mean_cost": mean, "se_cost": se, "median_cost": median}


def prepare(output_dir: Path, episodes: int, episode_steps: int, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for case in CASES:
        path = demand_bank_path(output_dir, case)
        bank = build_demand_bank(case, episodes, episode_steps, seed)
        metadata.append(save_demand_bank(path, bank, case, seed))
    (output_dir / "demand_banks.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    protocol_index = prepare_protocol_index(output_dir)
    print(
        f"Prepared {len(metadata)} fixed demand banks under {output_dir / 'demand_banks'}; "
        f"audited {protocol_index['project_actors']} project actors and "
        f"{protocol_index['source_checkpoints']} source checkpoints"
    )


def find_row(benchmark: str, case: str) -> dict[str, str]:
    for row in load_manifest_rows():
        if row["benchmark"] == benchmark and row["case"] == case:
            return row
    raise KeyError((benchmark, case))


def result_path(output_dir: Path, benchmark: str, case: str) -> Path:
    return output_dir / "parts" / f"{slug(benchmark)}__{case.lower()}.json"


def evaluate_one(
    benchmark: str,
    case: str,
    output_dir: Path,
    episodes: int,
    episode_steps: int,
    warmup: int,
) -> dict[str, Any]:
    row = find_row(benchmark, case)
    out = result_path(output_dir, benchmark, case)
    out.parent.mkdir(parents=True, exist_ok=True)
    if row["status"] == "not_applicable":
        result = {"benchmark": benchmark, "case": case, "status": "not_applicable", "reason": row["notes"]}
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result
    if benchmark == SOURCE_PPO:
        raise ValueError("Use evaluate_manifest_source_ppo.py for source PPO checkpoints")

    bank_path = demand_bank_path(output_dir, case)
    bank = load_demand_bank(bank_path)
    protocol_snapshot: dict[str, Any]
    if benchmark in HEURISTICS:
        costs, protocol_snapshot = evaluate_heuristic(
            benchmark, case, bank, episodes, episode_steps, warmup
        )
    else:
        actor_path = (ROOT / row["checkpoint_or_policy"]).resolve()
        log_path, protocol = load_training_protocol(output_dir, actor_path)
        env, _, input_args = build_project_env(case, protocol, bank)
        costs = evaluate_project_actor(
            benchmark, actor_path, env, input_args, protocol, episodes, episode_steps, warmup
        )
        protocol_snapshot = {
            "training_log": str(log_path.relative_to(ROOT)),
            "actor_path": str(actor_path.relative_to(ROOT)),
            "agent": protocol.get("agent"),
            "allocation_rule": input_args["allocation_rule"],
            "demand_fulfillment": input_args["demand_fulfillment"],
            "rho": input_args["rho"],
            "emergency_shipment_prob": input_args["emergency_shipment_prob"],
            "rdc_bsl_min_override": input_args["rdc_bsl_min_override"],
            "cdc_bsl_min_override": input_args["cdc_bsl_min_override"],
            "rdc_bsl_max_override": input_args["rdc_bsl_max_override"],
            "cdc_bsl_max_override": input_args["cdc_bsl_max_override"],
        }
        if benchmark in AN_FAMILY:
            protocol_snapshot.update({
                "rdc_integer_repair": "deterministic_largest_remainder",
                "rdc_repair_total": "nearest_aggregate_shipment_bounded_by_cdc_on_hand",
                "cdc_order_integer_repair": "floor(q + 0.5)",
            })

    result = {
        "benchmark": benchmark,
        "case": case,
        "status": "done",
        **summarize_costs(costs),
        "episodes": episodes,
        "episode_steps": episode_steps,
        "warmup_steps": warmup,
        "evaluated_periods_per_episode": episode_steps - warmup,
        "demand_bank": display_path(bank_path),
        "demand_bank_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
        "manifest_best_cost": (
            float(row["best_cost"]) if str(row.get("best_cost", "")).strip() else None
        ),
        "protocol": json_safe(protocol_snapshot),
    }
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("benchmark", "case", "mean_cost", "se_cost", "median_cost")}, ensure_ascii=False))
    return result


def merge(output_dir: Path) -> None:
    rows = []
    for benchmark_row in load_manifest_rows():
        benchmark, case = benchmark_row["benchmark"], benchmark_row["case"]
        path = result_path(output_dir, benchmark, case)
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            rows.append({"benchmark": benchmark, "case": case, "status": "missing_evaluation"})
    csv_path = output_dir / "results.csv"
    fieldnames = [
        "benchmark", "case", "status", "mean_cost", "se_cost", "median_cost",
        "episodes", "episode_steps", "warmup_steps", "demand_bank_sha256", "manifest_best_cost", "reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    completed_rows = [row for row in rows if row.get("status") == "done"]
    episodes = completed_rows[0]["episodes"] if completed_rows else "unknown"
    episode_steps = completed_rows[0]["episode_steps"] if completed_rows else "unknown"
    warmup_steps = completed_rows[0]["warmup_steps"] if completed_rows else "unknown"
    lines = [
        "# Unified Manifest Evaluation: K0-K6 and S1-S2",
        "",
        (
            "All methods within a case use the same saved demand bank. Statistics are "
            f"mean, SE, and median of {episodes} episode-average costs; each episode has "
            f"{episode_steps} periods and excludes its first {warmup_steps} warm-up periods."
        ),
        "",
        "| Benchmark | Case | Status | Mean cost | SE | Median cost |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] != "done":
            lines.append(f"| {row['benchmark']} | {row['case']} | {row['status']} |  |  |  |")
        else:
            lines.append(
                f"| {row['benchmark']} | {row['case']} | done | {row['mean_cost']:.6f} | "
                f"{row['se_cost']:.6f} | {row['median_cost']:.6f} |"
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Merged {len(rows)} rows into {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episode-steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--benchmark")
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if args.prepare:
        prepare(output_dir, args.episodes, args.episode_steps, args.seed)
    elif args.merge:
        merge(output_dir)
    elif args.benchmark and args.case:
        evaluate_one(
            args.benchmark, args.case, output_dir,
            args.episodes, args.episode_steps, args.warmup,
        )
    else:
        parser.error("choose --prepare, --merge, or both --benchmark and --case")


if __name__ == "__main__":
    main()
