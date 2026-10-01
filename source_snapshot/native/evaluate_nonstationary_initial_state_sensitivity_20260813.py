#!/usr/bin/env python3
"""Evaluate nonstationary policies under initial-inventory protocol variants.

The experiment keeps the formal actors and common demand banks fixed.  It
varies only the initial protection quantile and whether the same initial stock
is held entirely on hand (cold) or rebalanced across on-hand and forecast-shaped
pipeline positions (warm).  The warm protocol preserves total initial stock.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from Benchmark import (
    FiniteHorizonNonstationaryOWMREnv,
    allocation_prioritized_greedy,
    allocation_prioritized_greedy_inequality,
    decomposition_aggregation_rolling_distribution_aware,
    env_input_initialize,
    rolling_da_order_qty,
)
from call_agents_results import load_actor
from dataset import data_nonstationary_quantile_skus
from demand_utils import quantile_series_lead_time_pmf
from evaluation_statistics import mean_se_median


ROOT = Path(__file__).resolve().parent
SOURCE_RESULTS = (
    ROOT / "outputs/evaluation/nonstationary_long_top3_common1000_20260806/results.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/evaluation/nonstationary_initial_state_sensitivity_common1000_20260813"
)
HORIZON = 13
DEFAULT_ITEMS = ("02351RFH-005", "98011941", "98011894")
DEFAULT_QUANTILES = (0.5, 0.7, 0.9)
DEFAULT_START_MODES = ("cold", "warm_rebalanced")
METHODS = ("ra_sac", "ra_sac_ineq", "rolling_da")
ALLOCATION_RULES = {
    "ra_sac": "prioritized_greedy",
    "ra_sac_ineq": "prioritized_greedy_inequality",
    "rolling_da": "proportion",
}


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: np.ndarray) -> dict[str, float]:
    mean, se, median = mean_se_median(np.asarray(values, dtype=float))
    return {"mean": float(mean), "se": float(se), "median": float(median)}


def paired_stats(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    return {
        "mean_delta": float(delta.mean()),
        "se_delta": float(delta.std(ddof=1) / math.sqrt(delta.size)),
        "median_delta": float(np.median(delta)),
        "candidate_better_fraction": float(np.mean(delta < 0.0)),
    }


def period_mean(spec: dict, period: int) -> float:
    support, pmf = quantile_series_lead_time_pmf(spec, period, 1)
    return float(np.dot(support, pmf))


def forecast_rebalanced_inventory(total: float, forecast: list[float]) -> np.ndarray:
    """Split a fixed inventory position into on-hand and forecast-shaped pipeline."""
    total = max(float(total), 0.0)
    desired = np.maximum(np.asarray(forecast, dtype=float), 0.0)
    desired_sum = float(desired.sum())
    if desired_sum > total and desired_sum > 0.0:
        desired *= total / desired_sum
    on_hand = max(total - float(desired.sum()), 0.0)
    result = np.hstack(([on_hand], desired))
    # Absorb floating-point residue without changing the pipeline shape.
    result[0] += total - float(result.sum())
    return result


def apply_start_mode(info, start_mode: str):
    """Return a copied input object under the requested initial-state protocol."""
    adjusted = deepcopy(info)
    if start_mode == "cold":
        return adjusted
    if start_mode != "warm_rebalanced":
        raise ValueError(f"Unsupported start_mode={start_mode!r}")

    for k, lead_time in adjusted.rdc_dict.items():
        total = float(np.sum(adjusted.RDC_inventory[k]))
        forecast = [period_mean(adjusted.demand_feature[k], t) for t in range(int(lead_time))]
        adjusted.RDC_inventory[k] = forecast_rebalanced_inventory(total, forecast)

    cdc_lead = int(adjusted.item_lead_time)
    aggregate_forecast = [
        sum(period_mean(adjusted.demand_feature[k], t) for k in adjusted.rdc_dict)
        for t in range(cdc_lead)
    ]
    cdc_total = float(np.sum(adjusted.CDC_inventory))
    adjusted.CDC_inventory = forecast_rebalanced_inventory(cdc_total, aggregate_forecast)
    return adjusted


def load_formal_protocol(source_results: Path, item_codes: tuple[str, ...]) -> tuple[dict, dict]:
    payload = json.loads(source_results.read_text(encoding="utf-8"))
    actors: dict[tuple[str, str], dict] = {}
    for row in payload["formal_selections"]:
        key = (str(row["item_code"]), str(row["method"]))
        if key[0] in item_codes and key[1] in {"ra_sac", "ra_sac_ineq"}:
            actors[key] = {
                "actor_path": str(row["actor_path"]),
                "step": int(row["step"]),
                "training_rank": int(row["training_rank"]),
                "actor_sha256": str(row["actor_sha256"]),
            }

    banks = {
        str(row["item_code"]): {
            "path": str(row["path"]),
            "sha256": str(row["sha256"]),
            "episodes": int(row["episodes"]),
            "seed": int(row["seed"]),
        }
        for row in payload["demand_banks"]
        if str(row["item_code"]) in item_codes
    }
    missing_actors = [
        (item, method)
        for item in item_codes
        for method in ("ra_sac", "ra_sac_ineq")
        if (item, method) not in actors
    ]
    missing_banks = [item for item in item_codes if item not in banks]
    if missing_actors or missing_banks:
        raise KeyError(f"Missing formal inputs: actors={missing_actors}, banks={missing_banks}")
    return actors, banks


def load_bank(path: Path, episodes: int) -> dict[int, np.ndarray]:
    with np.load(path) as archive:
        bank = {
            int(key.removeprefix("rdc_")): np.asarray(archive[key], dtype=float)[:episodes]
            for key in archive.files
        }
    if any(values.shape != (episodes, HORIZON) for values in bank.values()):
        raise ValueError(f"Demand bank {path} does not provide {episodes} x {HORIZON} rows")
    return bank


def build_env(info, bank: dict[int, np.ndarray], allocation_rule: str, seed: int):
    env = FiniteHorizonNonstationaryOWMREnv(
        all_demand_list=bank,
        info_input=info,
        horizon=HORIZON,
        if_generate_demand=False,
        cost_plus=0.0,
        holding_weight=1.0,
        penalty_weight=0.0,
        allocation_rule=allocation_rule,
        train_cost_mode="controllable",
        include_period_in_state=True,
        include_action_mask_in_state=True,
        action_bound_profile="joint_quantile_v2",
        action_lower_quantile=0.5,
        action_upper_quantile=0.99,
        seed=seed,
        method="SAC",
    )
    env.episode = 0
    return env


def counterfactual_allocations(env, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compare equality and inequality allocation at one identical state/action."""
    env.nn_output_to_action(action)
    requests = []
    for k in env.rdcs:
        current = float(env.state_echelon_rdc[k][-1])
        request = round(max(0.0, float(env.target_inventory_position[k]) - current), 0)
        requests.append(request if env._is_rdc_order_effective(k) else 0.0)
    positions = {k: float(env.state_echelon_rdc[k][-1]) for k in env.rdcs}
    demand_para = {
        k: env._period_lead_time_pmf_model(k, env.period, env.rdcs[k].lead_time + 1)
        for k in env.rdcs
    }
    available = round(float(env.CDC_inventory[0]), 0)
    _, eq = allocation_prioritized_greedy(
        available, requests, demand_para, positions, env.holding_cost, env.backorder_cost
    )
    ineq_left, ineq = allocation_prioritized_greedy_inequality(
        available, requests, demand_para, positions, env.holding_cost, env.backorder_cost
    )
    return np.asarray(eq, dtype=float), np.hstack((np.asarray(ineq, dtype=float), ineq_left))


def empty_trace(episodes: int, rdc_count: int) -> dict[str, np.ndarray]:
    shape = (episodes, HORIZON, rdc_count)
    return {
        "request": np.zeros(shape, dtype=np.float32),
        "allocated": np.zeros(shape, dtype=np.float32),
        "start_backlog": np.zeros(shape, dtype=np.float32),
        "end_on_hand": np.zeros(shape, dtype=np.float32),
        "demand": np.zeros(shape, dtype=np.float32),
        "cdc_order": np.zeros((episodes, HORIZON), dtype=np.float32),
        "active_mask": np.zeros((episodes, HORIZON, rdc_count + 1), dtype=bool),
    }


def evaluate_actor(info, bank, actor_path: Path, allocation_rule: str, seed: int):
    actor = load_actor(actor_path, torch.device("cpu"))
    env = build_env(info, bank, allocation_rule, seed)
    episodes = len(next(iter(bank.values())))
    rdcs = list(info.rdc_dict)
    full = np.zeros(episodes, dtype=np.float64)
    controllable = np.zeros(episodes, dtype=np.float64)
    trace = empty_trace(episodes, len(rdcs))
    allocation = {
        "events": 0,
        "binding_events": 0,
        "unfilled_events": 0,
        "retention_events": 0,
        "retained_units": 0.0,
        "counterfactual_difference_events": 0,
        "counterfactual_difference_units": 0.0,
        "counterfactual_retention_events": 0,
        "counterfactual_retained_units": 0.0,
    }

    with torch.no_grad():
        for episode in range(episodes):
            state = env.reset()
            for period in range(HORIZON):
                trace["active_mask"][episode, period] = env.get_action_mask().astype(bool)
                trace["start_backlog"][episode, period] = [
                    max(-float(env.RDC_inventory[k][0]), 0.0) for k in rdcs
                ]
                action = actor(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
                if isinstance(action, tuple):
                    action = action[0]
                action_np = action.squeeze(0).cpu().numpy()

                eq_cf, ineq_cf = counterfactual_allocations(env, action_np)
                ineq_alloc_cf, ineq_left_cf = ineq_cf[:-1], float(ineq_cf[-1])
                cf_units = float(np.abs(eq_cf - ineq_alloc_cf).sum())
                if cf_units > 1e-9:
                    allocation["counterfactual_difference_events"] += 1
                    allocation["counterfactual_difference_units"] += cf_units
                retained_units_cf = float(eq_cf.sum() - ineq_alloc_cf.sum())
                if ineq_left_cf > 1e-9 and retained_units_cf > 1e-9:
                    allocation["counterfactual_retention_events"] += 1
                    allocation["counterfactual_retained_units"] += retained_units_cf

                state, _, done, (_, inv_info, order) = env.step(action_np)
                breakdown = env.last_cost_breakdown
                alloc_info = env.last_allocation_info
                full[episode] += sum(breakdown["full_cost"].values())
                controllable[episode] += sum(breakdown["train_cost"].values())
                trace["request"][episode, period] = alloc_info["requested"]
                trace["allocated"][episode, period] = alloc_info["allocated"]
                trace["end_on_hand"][episode, period] = [float(inv_info[k]) for k in rdcs]
                trace["demand"][episode, period] = [
                    float(breakdown["demand_info"][k]["demand"]) for k in rdcs
                ]
                trace["cdc_order"][episode, period] = float(order[0])
                allocation["events"] += 1
                allocation["binding_events"] += int(alloc_info["inventory_binding"])
                allocation["unfilled_events"] += int(alloc_info["unfilled_request"] > 1e-9)
                retained = float(alloc_info["retained_with_unfilled_request"])
                allocation["retention_events"] += int(retained > 1e-9)
                allocation["retained_units"] += retained
                if bool(done) != (period == HORIZON - 1):
                    raise RuntimeError(f"Unexpected done={done} at period={period}")

    return full / HORIZON, controllable / HORIZON, trace, allocation


def evaluate_da(info, bank, seed: int):
    schedule = decomposition_aggregation_rolling_distribution_aware(info, horizon=HORIZON)
    env = build_env(info, bank, "proportion", seed)
    env.method = "heu"
    episodes = len(next(iter(bank.values())))
    rdcs = list(info.rdc_dict)
    full = np.zeros(episodes, dtype=np.float64)
    controllable = np.zeros(episodes, dtype=np.float64)
    trace = empty_trace(episodes, len(rdcs))
    allocation = {
        "events": 0, "binding_events": 0, "unfilled_events": 0,
        "retention_events": 0, "retained_units": 0.0,
        "counterfactual_difference_events": 0, "counterfactual_difference_units": 0.0,
        "counterfactual_retention_events": 0, "counterfactual_retained_units": 0.0,
    }
    for episode in range(episodes):
        env.reset()
        for period in range(HORIZON):
            trace["active_mask"][episode, period] = env.get_action_mask().astype(bool)
            trace["start_backlog"][episode, period] = [
                max(-float(env.RDC_inventory[k][0]), 0.0) for k in rdcs
            ]
            action = rolling_da_order_qty(env, schedule)
            _, _, done, (_, inv_info, order) = env.step(action)
            breakdown = env.last_cost_breakdown
            alloc_info = env.last_allocation_info
            full[episode] += sum(breakdown["full_cost"].values())
            controllable[episode] += sum(breakdown["train_cost"].values())
            trace["request"][episode, period] = alloc_info["requested"]
            trace["allocated"][episode, period] = alloc_info["allocated"]
            trace["end_on_hand"][episode, period] = [float(inv_info[k]) for k in rdcs]
            trace["demand"][episode, period] = [
                float(breakdown["demand_info"][k]["demand"]) for k in rdcs
            ]
            trace["cdc_order"][episode, period] = float(order[0])
            allocation["events"] += 1
            allocation["binding_events"] += int(alloc_info["inventory_binding"])
            allocation["unfilled_events"] += int(alloc_info["unfilled_request"] > 1e-9)
            if bool(done) != (period == HORIZON - 1):
                raise RuntimeError(f"Unexpected done={done} at period={period}")
    return full / HORIZON, controllable / HORIZON, trace, allocation


def trace_summary(trace: dict[str, np.ndarray], info) -> dict:
    active_rdc = trace["active_mask"][:, :, :-1]
    active_cdc = trace["active_mask"][:, :, -1]
    total_demand = float(trace["demand"].sum())
    request_active = trace["request"][active_rdc]
    allocated_active = trace["allocated"][active_rdc]
    cdc_active_orders = trace["cdc_order"][active_cdc]
    first_arrival = {
        str(k): summarize(trace["start_backlog"][:, int(info.rdc_dict[k]), n])
        for n, k in enumerate(info.rdc_dict)
    }
    depletion = {}
    for n, k in enumerate(info.rdc_dict):
        depleted = trace["end_on_hand"][:, :, n] <= 0.0
        first = np.where(depleted.any(axis=1), depleted.argmax(axis=1) + 1, HORIZON + 1)
        depletion[str(k)] = {
            "fraction_depleted": float(np.mean(first <= HORIZON)),
            "median_first_period": (
                float(np.median(first[first <= HORIZON])) if np.any(first <= HORIZON) else None
            ),
        }
    return {
        "zero_request_rate_active": float(np.mean(request_active <= 1e-9)),
        "zero_allocated_rate_active": float(np.mean(allocated_active <= 1e-9)),
        "zero_cdc_order_rate_active": float(np.mean(cdc_active_orders <= 1e-9)),
        "requested_to_demand_ratio": float(trace["request"].sum() / max(total_demand, 1e-12)),
        "allocated_to_demand_ratio": float(trace["allocated"].sum() / max(total_demand, 1e-12)),
        "cdc_order_to_demand_ratio": float(trace["cdc_order"].sum() / max(total_demand, 1e-12)),
        "first_replenishment_arrival_backlog": first_arrival,
        "depletion": depletion,
    }


def period_node_rows(job: dict, method: str, trace: dict, info) -> list[dict]:
    rows = []
    rdcs = list(info.rdc_dict)
    for period in range(HORIZON):
        for n, k in enumerate(rdcs):
            active = trace["active_mask"][:, period, n]
            rows.append({
                "job_id": job["job_id"], "item_code": job["item_code"],
                "initial_quantile": job["initial_quantile"], "start_mode": job["start_mode"],
                "method": method, "period": period + 1, "node": f"RDC{k}",
                "active_fraction": float(active.mean()),
                "zero_request_rate": float(np.mean(trace["request"][:, period, n] <= 1e-9)),
                "zero_allocated_rate": float(np.mean(trace["allocated"][:, period, n] <= 1e-9)),
                "mean_request": float(trace["request"][:, period, n].mean()),
                "mean_allocated": float(trace["allocated"][:, period, n].mean()),
                "mean_start_backlog": float(trace["start_backlog"][:, period, n].mean()),
                "mean_end_on_hand": float(trace["end_on_hand"][:, period, n].mean()),
                "mean_demand": float(trace["demand"][:, period, n].mean()),
            })
        cdc_active = trace["active_mask"][:, period, -1]
        rows.append({
            "job_id": job["job_id"], "item_code": job["item_code"],
            "initial_quantile": job["initial_quantile"], "start_mode": job["start_mode"],
            "method": method, "period": period + 1, "node": "CDC",
            "active_fraction": float(cdc_active.mean()),
            "zero_request_rate": float(np.mean(trace["cdc_order"][:, period] <= 1e-9)),
            "zero_allocated_rate": float("nan"),
            "mean_request": float(trace["cdc_order"][:, period].mean()),
            "mean_allocated": float("nan"), "mean_start_backlog": 0.0,
            "mean_end_on_hand": float("nan"),
            "mean_demand": float(trace["demand"][:, period].sum(axis=1).mean()),
        })
    return rows


def run_job(job: dict, output_dir: Path) -> dict:
    torch.set_num_threads(1)
    logging.getLogger("Benchmark").setLevel(logging.WARNING)
    datasets, metadata = data_nonstationary_quantile_skus(
        item_codes=(job["item_code"],),
        initial_inventory_quantile=float(job["initial_quantile"]),
    )
    base_info = env_input_initialize(*datasets[0])[0]
    info = apply_start_mode(base_info, job["start_mode"])
    bank = load_bank(resolve_path(job["bank_path"]), int(job["episodes"]))
    episode_dir = output_dir / "episode_costs"
    trace_dir = output_dir / "trace_arrays"
    episode_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    method_results = {}
    traces = {}
    for method in METHODS:
        if method == "rolling_da":
            full, controllable, trace, allocation = evaluate_da(info, bank, job["seed"])
            actor_meta = None
        else:
            actor_meta = job["actors"][method]
            actor_path = resolve_path(actor_meta["actor_path"])
            full, controllable, trace, allocation = evaluate_actor(
                info, bank, actor_path, ALLOCATION_RULES[method], job["seed"]
            )
        np.savez_compressed(
            episode_dir / f"{job['job_id']}_{method}.npz",
            full_weekly_cost=full,
            controllable_weekly_cost=controllable,
        )
        np.savez_compressed(trace_dir / f"{job['job_id']}_{method}.npz", **trace)
        method_results[method] = {
            "allocation_rule": ALLOCATION_RULES[method],
            "actor": actor_meta,
            "full_weekly_cost": summarize(full),
            "controllable_weekly_cost": summarize(controllable),
            "trace_summary": trace_summary(trace, info),
            "allocation_diagnostics": allocation,
        }
        traces[method] = trace

    costs = {
        method: np.load(episode_dir / f"{job['job_id']}_{method}.npz")
        for method in METHODS
    }
    comparisons = {}
    for candidate, reference in (
        ("ra_sac", "rolling_da"),
        ("ra_sac_ineq", "rolling_da"),
        ("ra_sac_ineq", "ra_sac"),
    ):
        comparisons[f"{candidate}_vs_{reference}"] = {
            "full_weekly_cost": paired_stats(
                costs[candidate]["full_weekly_cost"], costs[reference]["full_weekly_cost"]
            ),
            "controllable_weekly_cost": paired_stats(
                costs[candidate]["controllable_weekly_cost"],
                costs[reference]["controllable_weekly_cost"],
            ),
        }

    eq_trace, ineq_trace = traces["ra_sac"], traces["ra_sac_ineq"]
    actual_allocation_delta = np.abs(eq_trace["allocated"] - ineq_trace["allocated"])
    actual_request_delta = np.abs(eq_trace["request"] - ineq_trace["request"])
    comparisons["eq_ineq_path_difference"] = {
        "request_difference_events": int(np.sum(np.any(actual_request_delta > 1e-9, axis=2))),
        "request_difference_units": float(actual_request_delta.sum()),
        "allocation_difference_events": int(np.sum(np.any(actual_allocation_delta > 1e-9, axis=2))),
        "allocation_difference_units": float(actual_allocation_delta.sum()),
    }

    initial = {
        "rdc": {str(k): np.asarray(info.RDC_inventory[k], dtype=float).tolist() for k in info.rdc_dict},
        "cdc": np.asarray(info.CDC_inventory, dtype=float).tolist(),
        "total": float(
            sum(np.sum(info.RDC_inventory[k]) for k in info.rdc_dict)
            + np.sum(info.CDC_inventory)
        ),
        "protocol": metadata[0]["initial_inventory_protocol"],
    }
    return {
        **job,
        "initial_inventory": initial,
        "methods": method_results,
        "comparisons": comparisons,
        "period_node_rows": sum(
            (period_node_rows(job, method, traces[method], info) for method in METHODS), []
        ),
    }


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge_results(output_dir: Path, jobs: list[dict]) -> None:
    parts = []
    for job in jobs:
        path = output_dir / "parts" / f"{job['job_id']}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        parts.append(json.loads(path.read_text(encoding="utf-8")))

    result_rows, comparison_rows, allocation_rows, period_rows = [], [], [], []
    for part in parts:
        common = {
            "item_code": part["item_code"],
            "initial_quantile": part["initial_quantile"],
            "start_mode": part["start_mode"],
        }
        for method, result in part["methods"].items():
            result_rows.append({
                **common, "method": method,
                "controllable_mean": result["controllable_weekly_cost"]["mean"],
                "controllable_se": result["controllable_weekly_cost"]["se"],
                "controllable_median": result["controllable_weekly_cost"]["median"],
                "full_mean": result["full_weekly_cost"]["mean"],
                "full_se": result["full_weekly_cost"]["se"],
                "full_median": result["full_weekly_cost"]["median"],
                **result["trace_summary"],
            })
            diagnostics = result["allocation_diagnostics"]
            allocation_rows.append({
                **common, "method": method, **diagnostics,
                "binding_rate": diagnostics["binding_events"] / max(diagnostics["events"], 1),
                "retention_rate": diagnostics["retention_events"] / max(diagnostics["events"], 1),
                "counterfactual_difference_rate": (
                    diagnostics["counterfactual_difference_events"] / max(diagnostics["events"], 1)
                ),
            })
        for label, result in part["comparisons"].items():
            if label == "eq_ineq_path_difference":
                comparison_rows.append({**common, "comparison": label, **result})
                continue
            ctl = result["controllable_weekly_cost"]
            full = result["full_weekly_cost"]
            comparison_rows.append({
                **common, "comparison": label,
                "controllable_mean_delta": ctl["mean_delta"],
                "controllable_se_delta": ctl["se_delta"],
                "controllable_better_fraction": ctl["candidate_better_fraction"],
                "full_mean_delta": full["mean_delta"],
                "full_se_delta": full["se_delta"],
            })
        period_rows.extend(part.pop("period_node_rows"))

    atomic_json(output_dir / "results.json", {"jobs": parts})
    write_csv(output_dir / "results.csv", result_rows)
    write_csv(output_dir / "comparisons.csv", comparison_rows)
    write_csv(output_dir / "allocation_diagnostics.csv", allocation_rows)
    write_csv(output_dir / "period_node_metrics.csv", period_rows)

    lines = [
        "# Nonstationary initial-state sensitivity",
        "",
        "Protocol: 1,000 common-demand 13-week episodes; fixed formal actors; "
        "warm_rebalanced preserves total initial stock while moving it into forecast-shaped pipelines.",
        "",
        "| Item | q | Start | RA-SAC | RA-SAC-Ineq | Rolling DA | Ineq-Eq | Ineq retention |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for part in parts:
        methods = part["methods"]
        eq = methods["ra_sac"]["controllable_weekly_cost"]["mean"]
        ineq = methods["ra_sac_ineq"]["controllable_weekly_cost"]["mean"]
        da = methods["rolling_da"]["controllable_weekly_cost"]["mean"]
        alloc = methods["ra_sac_ineq"]["allocation_diagnostics"]
        lines.append(
            f"| {part['item_code']} | {part['initial_quantile']:.1f} | {part['start_mode']} | "
            f"{eq:.4f} | {ineq:.4f} | {da:.4f} | {ineq-eq:+.4f} | "
            f"{alloc['retention_events']}/{alloc['events']} |"
        )
    lines.extend([
        "",
        "Detailed paired differences, allocation activation, zero-order rates, backlog, depletion, "
        "and period-node traces are in the companion CSV files.",
    ])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--item-codes", default=",".join(DEFAULT_ITEMS))
    parser.add_argument("--quantiles", default=",".join(str(x) for x in DEFAULT_QUANTILES))
    parser.add_argument("--start-modes", default=",".join(DEFAULT_START_MODES))
    parser.add_argument("--source-results", type=Path, default=SOURCE_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.episodes <= 1 or args.workers <= 0:
        parser.error("--episodes must exceed 1 and --workers must be positive")
    items = tuple(value.strip() for value in args.item_codes.split(",") if value.strip())
    quantiles = tuple(float(value) for value in args.quantiles.split(",") if value.strip())
    modes = tuple(value.strip() for value in args.start_modes.split(",") if value.strip())
    if any(not 0.0 <= q <= 1.0 for q in quantiles):
        parser.error("All quantiles must be in [0, 1]")
    if any(mode not in DEFAULT_START_MODES for mode in modes):
        parser.error(f"Supported start modes are {DEFAULT_START_MODES}")

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "parallel_status.log"
    launch_path = output_dir / "launch_manifest.json"
    complete_path = output_dir / "COMPLETE"
    if launch_path.exists() and not complete_path.exists():
        raise RuntimeError(f"Existing unfinished launch detected at {launch_path}")
    if complete_path.exists():
        print(f"Already complete: {output_dir}", flush=True)
        return

    source_results = resolve_path(args.source_results)
    actors, banks = load_formal_protocol(source_results, items)
    jobs = []
    for item_index, item in enumerate(items):
        for q in quantiles:
            for mode in modes:
                q_tag = str(q).replace(".", "p")
                job_id = f"{item}_q{q_tag}_{mode}"
                jobs.append({
                    "job_id": job_id,
                    "item_code": item,
                    "initial_quantile": q,
                    "start_mode": mode,
                    "episodes": args.episodes,
                    "seed": args.seed + item_index * 1009,
                    "bank_path": banks[item]["path"],
                    "bank_sha256": banks[item]["sha256"],
                    "actors": {
                        method: actors[(item, method)]
                        for method in ("ra_sac", "ra_sac_ineq")
                    },
                })
    manifest = {
        "manager_pid": os.getpid(),
        "episodes": args.episodes,
        "workers": args.workers,
        "source_results": portable_path(source_results),
        "initial_state_protocols": {
            "cold": "all selected initial inventory is on hand; pipelines are empty",
            "warm_rebalanced": (
                "the same total initial inventory is split between on-hand and forecast-shaped "
                "pipeline slots; no stock is added"
            ),
        },
        "jobs": jobs,
    }
    atomic_json(launch_path, manifest)
    (output_dir / "manager_pid.txt").write_text(str(os.getpid()) + "\n", encoding="ascii")
    status_path.write_text(
        f"START total={len(jobs)} workers={args.workers} pid={os.getpid()}\n",
        encoding="utf-8",
    )

    failures = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_job, job, output_dir): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                atomic_json(parts_dir / f"{job['job_id']}.json", result)
                done += 1
                line = f"DONE {done}/{len(jobs)} {job['job_id']}\n"
            except Exception as exc:  # pragma: no cover - background diagnostics
                failures.append({
                    "job_id": job["job_id"],
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                })
                line = f"FAIL {job['job_id']} {exc!r}\n"
            with status_path.open("a", encoding="utf-8") as file:
                file.write(line)
            print(line.rstrip(), flush=True)

    atomic_json(output_dir / "failures.json", failures)
    if failures:
        with status_path.open("a", encoding="utf-8") as file:
            file.write(f"SUMMARY total={len(jobs)} done={done} failures={len(failures)}\n")
        raise RuntimeError(f"{len(failures)} jobs failed; see {output_dir / 'failures.json'}")

    merge_results(output_dir, jobs)
    complete_path.write_text("complete\n", encoding="ascii")
    with status_path.open("a", encoding="utf-8") as file:
        file.write(f"SUMMARY total={len(jobs)} done={done} failures=0\n")
    print(f"COMPLETE {output_dir}", flush=True)


if __name__ == "__main__":
    main()
