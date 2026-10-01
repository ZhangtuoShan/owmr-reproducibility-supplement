#!/usr/bin/env python3
"""Common-demand 1000-episode evaluation for the newest SAC candidates.

This is deliberately separate from the result manifest.  It validates new
candidate actors against the currently selected actors before any manifest
entry is changed.  Every pair uses exactly the same demand paths and reports
the paired cost difference in addition to the usual mean/SE/median summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from Benchmark import OWMRInventoryEnv, auto_hyperparameter, env_input_initialize, generate_env
from dataset import data_random_table_benchmarks, data_random_table_benchmark_metadata
from demand_utils import sample_demand
from evaluate_manifest_kaynov_small import (
    build_demand_bank,
    build_project_env,
    evaluate_project_actor,
    index_training_logs,
    json_safe,
    load_demand_bank,
    parse_value,
    save_demand_bank,
    split_top_level,
    summarize_costs,
)
from evaluation_statistics import mean_se_median
from main_DRL_agents import get_env_args


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs/evaluation/recent_sac_candidates_common_1000ep_20260731"
EPISODES = 1000
EPISODE_STEPS = 1000
WARMUP_STEPS = 500
SEED = 20260731


SPECS: tuple[dict[str, str], ...] = (
    # New priority-search candidates versus the current manifest actors.
    {
        "job": "k1_ineq_manifest",
        "pair": "K1_RA-SAC_ineq",
        "role": "baseline",
        "case": "K1",
        "benchmark": "RA-SAC ineq.",
        "kind": "project",
        "actor": "outputs/train/sac_eq_ineq_k0_k4_two_stage_v2_fixed_stage2_sac_ineq_K1_k0_two_stage/data_kaynov_1/actor__000000696832_-0012.879.pt",
    },
    {
        "job": "k1_ineq_new",
        "pair": "K1_RA-SAC_ineq",
        "role": "candidate",
        "case": "K1",
        "benchmark": "RA-SAC ineq.",
        "kind": "project",
        "actor": "outputs/train/priority1_sac_ineq_K1_k4best_lr0p01_rs1_alr0p0001_1m/data_kaynov_1/actor__000000860672_-0012.821.pt",
    },
    {
        "job": "k3_eq_manifest",
        "pair": "K3_RA-SAC",
        "role": "baseline",
        "case": "K3",
        "benchmark": "RA-SAC",
        "kind": "project",
        "actor": "outputs/train/sac_eq_K3_k0_adaptive_fixed/data_kaynov_3/actor__000000328192_-0019.026.pt",
    },
    {
        "job": "k3_eq_new",
        "pair": "K3_RA-SAC",
        "role": "candidate",
        "case": "K3",
        "benchmark": "RA-SAC",
        "kind": "project",
        "actor": "outputs/train/priority1_sac_eq_K3_k4best_lr0p01_rs1_alr0p0001_1m/data_kaynov_3/actor__000000522752_-0018.467.pt",
    },
    {
        "job": "k4_eq_manifest",
        "pair": "K4_RA-SAC",
        "role": "baseline",
        "case": "K4",
        "benchmark": "RA-SAC",
        "kind": "project",
        "actor": "outputs/train/sac_eq_K4_relaxed_rdc2_23_rdc3_6/data_kaynov_4/actor__000000952832_-0025.588.pt",
    },
    {
        "job": "k4_eq_new",
        "pair": "K4_RA-SAC",
        "role": "candidate",
        "case": "K4",
        "benchmark": "RA-SAC",
        "kind": "project",
        "actor": "outputs/train/k4_alpha_old_lr0p01_rs1_raw_w512_alr0p0001/data_kaynov_4/actor__000000870912_-0024.617.pt",
    },
    {
        "job": "2_1_ineq_manifest",
        "pair": "2-1_RA-SAC_ineq",
        "role": "baseline",
        "case": "2-1",
        "benchmark": "RA-SAC ineq.",
        "kind": "table",
        "actor": "outputs/train/sac_ineq_2_1_exactalloc_lr0p0005_rs0p00195312_g0p96_rt0p5_1m/2-1/actor__000000573952_-0030.730.pt",
    },
    {
        "job": "2_1_ineq_new",
        "pair": "2-1_RA-SAC_ineq",
        "role": "candidate",
        "case": "2-1",
        "benchmark": "RA-SAC ineq.",
        "kind": "table",
        "actor": "outputs/train/priority2a_grid_sac_ineq_2_1_lr0p01_rs0p25_alr0p0001_1m/2-1/actor__000000973312_-0030.590.pt",
    },
    {
        "job": "2_3_ineq_manifest",
        "pair": "2-3_RA-SAC_ineq",
        "role": "baseline",
        "case": "2-3",
        "benchmark": "RA-SAC ineq.",
        "kind": "table",
        "actor": "outputs/train/sac_ineq_2_3_exactalloc_lr0p0005_rs0p00195312_g0p96_rt0p5_1m/2-3/actor__000000430592_-0058.548.pt",
    },
    {
        "job": "2_3_ineq_new",
        "pair": "2-3_RA-SAC_ineq",
        "role": "candidate",
        "case": "2-3",
        "benchmark": "RA-SAC ineq.",
        "kind": "table",
        "actor": "outputs/train/priority2b_transfer_sac_ineq_2_3_lr0p01_rs0p25_alr0p0001_1m/2-3/actor__000000051712_-0058.475.pt",
    },
    {
        "job": "2_5_ineq_manifest",
        "pair": "2-5_RA-SAC_ineq",
        "role": "baseline",
        "case": "2-5",
        "benchmark": "RA-SAC ineq.",
        "kind": "table",
        "actor": "outputs/train/sac_ineq_2_5_exactalloc_lr0p0005_rs0p00195312_g0p96_rt0p5_1m/2-5/actor__000000266752_-0085.112.pt",
    },
    {
        "job": "2_5_ineq_new",
        "pair": "2-5_RA-SAC_ineq",
        "role": "candidate",
        "case": "2-5",
        "benchmark": "RA-SAC ineq.",
        "kind": "table",
        "actor": "outputs/train/priority2b_transfer_sac_ineq_2_5_lr0p01_rs0p25_alr0p0001_1m/2-5/actor__000000686592_-0084.931.pt",
    },
    # S1/S2: compare the selected 1M checkpoint with the continued 2M state.
    {
        "job": "s1_eq_1m",
        "pair": "S1_RA-SAC_2M_vs_1M",
        "role": "baseline",
        "case": "S1",
        "benchmark": "RA-SAC",
        "kind": "project",
        "actor": "outputs/train/ra_sac_exact_S1_1m/data_small_rvi_0/actor__000000748032_-0007.774.pt",
    },
    {
        "job": "s1_eq_2m",
        "pair": "S1_RA-SAC_2M_vs_1M",
        "role": "candidate",
        "case": "S1",
        "benchmark": "RA-SAC",
        "kind": "project",
        "actor": "outputs/train/ra_sac_exact_S1_resume2m/data_small_rvi_0/actor__000001464832_-0007.751.pt",
    },
    {
        "job": "s1_ineq_1m",
        "pair": "S1_RA-SAC_ineq_2M_vs_1M",
        "role": "baseline",
        "case": "S1",
        "benchmark": "RA-SAC ineq.",
        "kind": "project",
        "actor": "outputs/train/ra_sac_ineq_exact_S1_1m/data_small_rvi_0/actor__000000748032_-0007.805.pt",
    },
    {
        "job": "s1_ineq_2m",
        "pair": "S1_RA-SAC_ineq_2M_vs_1M",
        "role": "candidate",
        "case": "S1",
        "benchmark": "RA-SAC ineq.",
        "kind": "project",
        "actor": "outputs/train/ra_sac_ineq_exact_S1_resume2m/data_small_rvi_0/actor__000001464832_-0007.772.pt",
    },
    {
        "job": "s2_eq_1m",
        "pair": "S2_RA-SAC_2M_vs_1M",
        "role": "baseline",
        "case": "S2",
        "benchmark": "RA-SAC",
        "kind": "project",
        "actor": "outputs/train/ra_sac_exact_S2_1m/data_small_rvi_1/actor__000000748032_-0011.820.pt",
    },
    {
        "job": "s2_eq_2m_final",
        "pair": "S2_RA-SAC_2M_vs_1M",
        "role": "candidate",
        "case": "S2",
        "benchmark": "RA-SAC",
        "kind": "project",
        "actor": "outputs/train/ra_sac_exact_S2_resume2m/data_small_rvi_1/actor__000001976832.pt",
    },
    {
        "job": "s2_ineq_1m",
        "pair": "S2_RA-SAC_ineq_2M_vs_1M",
        "role": "baseline",
        "case": "S2",
        "benchmark": "RA-SAC ineq.",
        "kind": "project",
        "actor": "outputs/train/ra_sac_ineq_exact_S2_1m/data_small_rvi_1/actor__000000748032_-0011.877.pt",
    },
    {
        "job": "s2_ineq_2m_final",
        "pair": "S2_RA-SAC_ineq_2M_vs_1M",
        "role": "candidate",
        "case": "S2",
        "benchmark": "RA-SAC ineq.",
        "kind": "project",
        "actor": "outputs/train/ra_sac_ineq_exact_S2_resume2m/data_small_rvi_1/actor__000001976832.pt",
    },
)


def table_case_index(case: str) -> int:
    metadata = data_random_table_benchmark_metadata()
    return next(i for i, row in enumerate(metadata) if row["case"] == case)


def bank_path(case: str) -> Path:
    if case in {"S1", "S2"}:
        return ROOT / "outputs/evaluation/s1s2_13bench_rvi_1000ep_20260728/demand_banks" / f"{case}.npz"
    return OUTPUT_DIR / "demand_banks" / f"{case}.npz"


def build_table_bank(case: str) -> dict[int, np.ndarray]:
    data = data_random_table_benchmarks()[table_case_index(case)]
    info = env_input_initialize(*data)[0]
    rng = np.random.default_rng(SEED + table_case_index(case) * 1009)
    bank: dict[int, np.ndarray] = {}
    for rdc in info.rdc_dict:
        values = np.asarray(
            sample_demand(
                info.demand_feature[rdc],
                size=(EPISODES, EPISODE_STEPS),
                random_state=rng,
            ),
            dtype=np.float64,
        )
        values[:, 0] = 0.0
        bank[int(rdc)] = values
    return bank


def save_local_bank(path: Path, bank: dict[int, np.ndarray], case: str) -> dict[str, Any]:
    if "-" not in case:
        return save_demand_bank(path, bank, case, SEED)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"rdc_{key}": value for key, value in bank.items()})
    first = next(iter(bank.values()))
    return {
        "case": case,
        "seed": SEED + table_case_index(case) * 1009,
        "episodes": int(first.shape[0]),
        "episode_steps": int(first.shape[1]),
        "rdcs": sorted(bank),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "path": str(path.relative_to(ROOT)),
    }


def parse_table_training_logs() -> dict[str, tuple[Path, dict[str, Any]]]:
    index: dict[str, tuple[Path, dict[str, Any]]] = {}
    patterns = (
        re.compile(r"^Training table case (\S+) index=(\d+),\s*(.*)$"),
        re.compile(r"^Training dataset=table15 case (\S+) index=(\d+),\s*(.*)$"),
    )
    for log_path in (ROOT / "outputs/logs").glob("**/*.log"):
        protocol: dict[str, Any] | None = None
        try:
            with log_path.open(encoding="utf-8", errors="ignore") as file:
                for raw in file:
                    line = raw.strip()
                    match = next((pattern.match(line) for pattern in patterns if pattern.match(line)), None)
                    if match:
                        protocol = {"case": match.group(1), "case_index": int(match.group(2))}
                        for part in split_top_level(match.group(3)):
                            if "=" in part:
                                key, value = part.split("=", 1)
                                protocol[key.strip()] = parse_value(value)
                    elif protocol is not None and line.startswith("Action-space config:"):
                        action_config: dict[str, Any] = {}
                        for part in split_top_level(line.split(":", 1)[1]):
                            if "=" in part:
                                key, value = part.split("=", 1)
                                action_config[key.strip()] = parse_value(value)
                        protocol.update(
                            {
                                "rdc_bsl_min_override": action_config.get("rdc_min"),
                                "cdc_bsl_min_override": action_config.get("cdc_min"),
                                "rdc_bsl_max_override": action_config.get("rdc_max"),
                                "cdc_bsl_max_override": action_config.get("cdc_max"),
                                "action_space_profile": action_config.get("profile"),
                            }
                        )
                    if protocol is not None and protocol.get("cwd"):
                        cwd = Path(str(protocol["cwd"]))
                        index[str(cwd.resolve())] = (log_path, protocol.copy())
        except OSError:
            continue
    return index


def prepare() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "parts").mkdir(exist_ok=True)
    (OUTPUT_DIR / "episode_means").mkdir(exist_ok=True)
    (OUTPUT_DIR / "job_logs").mkdir(exist_ok=True)
    metadata: list[dict[str, Any]] = []
    for case in sorted({spec["case"] for spec in SPECS}):
        path = bank_path(case)
        if case in {"S1", "S2"}:
            bank = load_demand_bank(path)
            first = next(iter(bank.values()))
            if first.shape != (EPISODES, EPISODE_STEPS):
                raise ValueError(f"Unexpected existing demand bank shape for {case}: {first.shape}")
            metadata.append(
                {
                    "case": case,
                    "path": str(path.relative_to(ROOT)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "source": "existing_common_1000ep_bank",
                }
            )
            continue
        bank = build_table_bank(case) if "-" in case else build_demand_bank(case, EPISODES, EPISODE_STEPS, SEED)
        metadata.append(save_local_bank(path, bank, case))

    project_index = index_training_logs()
    table_index = parse_table_training_logs()
    prepared_specs: list[dict[str, Any]] = []
    for raw_spec in SPECS:
        spec: dict[str, Any] = dict(raw_spec)
        actor = (ROOT / spec["actor"]).resolve()
        if not actor.exists():
            raise FileNotFoundError(actor)
        lookup = table_index if spec["kind"] == "table" else project_index
        entry = lookup.get(str(actor.parent.resolve()))
        if entry is None:
            raise KeyError(f"No training protocol found for {actor.parent}")
        log_path, protocol = entry
        spec["training_log"] = str(log_path.relative_to(ROOT))
        spec["protocol"] = json_safe(protocol)
        spec["demand_bank"] = str(bank_path(spec["case"]).relative_to(ROOT))
        prepared_specs.append(spec)

    (OUTPUT_DIR / "demand_banks.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "specs.json").write_text(
        json.dumps(prepared_specs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_specs() -> list[dict[str, Any]]:
    return json.loads((OUTPUT_DIR / "specs.json").read_text(encoding="utf-8"))


def restore_bound_key_types(protocol: dict[str, Any]) -> dict[str, Any]:
    protocol = dict(protocol)
    for key in ("rdc_bsl_min_override", "rdc_bsl_max_override", "adaptive_rdc_bsl_min", "adaptive_rdc_bsl_max"):
        value = protocol.get(key)
        if isinstance(value, dict):
            protocol[key] = {int(rdc): bound for rdc, bound in value.items()}
    return protocol


def build_table_env(case: str, protocol: dict[str, Any], bank: dict[int, np.ndarray]):
    data = data_random_table_benchmarks()[table_case_index(case)]
    info = env_input_initialize(*data)[0]
    _, _, _, holding_weight, penalty_weight, _, _ = auto_hyperparameter(info, 100)
    input_args = {
        "all_demand_list": bank,
        "holding_weight": holding_weight,
        "penalty_weight": penalty_weight,
        "cost_plus": 0.0,
        "info_input": info,
        "if_generate_demand": False,
        "seed": SEED,
        "rho": float(protocol.get("rho", 1.0)),
        "allocation_rule": protocol.get("allocation_rule", "prioritized_greedy_inequality"),
        "demand_fulfillment": protocol.get("demand_fulfillment", "partial_backorder"),
        "emergency_shipment_prob": protocol.get("emergency_shipment_prob", 1.0),
        "rdc_bsl_min_override": protocol.get("rdc_bsl_min_override"),
        "cdc_bsl_min_override": protocol.get("cdc_bsl_min_override"),
        "rdc_bsl_max_override": protocol.get("rdc_bsl_max_override"),
        "cdc_bsl_max_override": protocol.get("cdc_bsl_max_override"),
    }
    env = generate_env(OWMRInventoryEnv, get_env_args(input_args))
    env.total_periods = EPISODE_STEPS
    env.episode = -1
    return env, input_args


def evaluate_job(job_name: str) -> None:
    torch.set_num_threads(1)
    spec = next(item for item in load_specs() if item["job"] == job_name)
    bank = load_demand_bank(ROOT / spec["demand_bank"])
    protocol = restore_bound_key_types(spec["protocol"])
    if spec["kind"] == "table":
        env, input_args = build_table_env(spec["case"], protocol, bank)
    else:
        env, _, input_args = build_project_env(spec["case"], protocol, bank)
    costs = evaluate_project_actor(
        spec["benchmark"],
        ROOT / spec["actor"],
        env,
        input_args,
        protocol,
        EPISODES,
        EPISODE_STEPS,
        WARMUP_STEPS,
    )
    episode_means = costs.mean(axis=1)
    np.savez_compressed(OUTPUT_DIR / "episode_means" / f"{job_name}.npz", episode_means=episode_means)
    result = {
        "job": job_name,
        "pair": spec["pair"],
        "role": spec["role"],
        "case": spec["case"],
        "benchmark": spec["benchmark"],
        **summarize_costs(costs),
        "episodes": EPISODES,
        "episode_steps": EPISODE_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "actor": spec["actor"],
        "training_log": spec["training_log"],
        "demand_bank": spec["demand_bank"],
    }
    (OUTPUT_DIR / "parts" / f"{job_name}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


def paired_se(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def merge() -> None:
    rows = [
        json.loads((OUTPUT_DIR / "parts" / f"{spec['job']}.json").read_text(encoding="utf-8"))
        for spec in load_specs()
    ]
    with (OUTPUT_DIR / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    comparisons: list[dict[str, Any]] = []
    for pair in sorted({row["pair"] for row in rows}):
        baseline = next(row for row in rows if row["pair"] == pair and row["role"] == "baseline")
        candidate = next(row for row in rows if row["pair"] == pair and row["role"] == "candidate")
        with np.load(OUTPUT_DIR / "episode_means" / f"{baseline['job']}.npz") as data:
            base_means = np.asarray(data["episode_means"], dtype=float)
        with np.load(OUTPUT_DIR / "episode_means" / f"{candidate['job']}.npz") as data:
            candidate_means = np.asarray(data["episode_means"], dtype=float)
        delta = candidate_means - base_means
        mean_delta, _, median_delta = mean_se_median(delta)
        comparisons.append(
            {
                "pair": pair,
                "case": candidate["case"],
                "benchmark": candidate["benchmark"],
                "baseline_mean": baseline["mean_cost"],
                "baseline_se": baseline["se_cost"],
                "baseline_median": baseline["median_cost"],
                "candidate_mean": candidate["mean_cost"],
                "candidate_se": candidate["se_cost"],
                "candidate_median": candidate["median_cost"],
                "candidate_minus_baseline": mean_delta,
                "paired_se": paired_se(delta),
                "median_paired_delta": median_delta,
                "candidate_better_episode_fraction": float(np.mean(delta < 0.0)),
                "candidate_actor": candidate["actor"],
                "baseline_actor": baseline["actor"],
            }
        )
    (OUTPUT_DIR / "comparisons.json").write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (OUTPUT_DIR / "comparisons.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    lines = [
        "# Recent SAC Candidate Common-Demand Evaluation",
        "",
        f"Protocol: {EPISODES} common episodes, {EPISODE_STEPS} periods per episode, "
        f"costs averaged over periods {WARMUP_STEPS + 1}-{EPISODE_STEPS}. Negative paired delta is better.",
        "",
        "| Pair | Baseline mean (SE) | Candidate mean (SE) | Baseline median | Candidate median | Paired delta (SE) | Better episodes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['pair']} | {row['baseline_mean']:.6f} ({row['baseline_se']:.6f}) | "
            f"{row['candidate_mean']:.6f} ({row['candidate_se']:.6f}) | "
            f"{row['baseline_median']:.6f} | {row['candidate_median']:.6f} | "
            f"{row['candidate_minus_baseline']:+.6f} ({row['paired_se']:.6f}) | "
            f"{100.0 * row['candidate_better_episode_fraction']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- K/Table15 candidate rows compare the newly trained best checkpoint against the current manifest checkpoint.",
            "- S1 uses the best saved continuation checkpoint; S2 uses the final saved continuation state because no post-1M checkpoint beat the inherited 1M global best.",
            "- This report is validation evidence only and does not update the manifest.",
        ]
    )
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(workers: int) -> None:
    prepare()
    specs = load_specs()
    pending = list(specs)
    active: dict[subprocess.Popen[Any], tuple[dict[str, Any], Any]] = {}
    failures: list[str] = []
    while pending or active:
        while pending and len(active) < workers:
            spec = pending.pop(0)
            log_file = (OUTPUT_DIR / "job_logs" / f"{spec['job']}.log").open("w", encoding="utf-8")
            command = [sys.executable, str(Path(__file__).resolve()), "--job", spec["job"]]
            process = subprocess.Popen(command, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT)
            active[process] = (spec, log_file)
        time.sleep(1.0)
        for process, (spec, log_file) in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_file.close()
            del active[process]
            if return_code != 0:
                failures.append(spec["job"])
                print(f"FAIL {spec['job']} rc={return_code}", flush=True)
            else:
                print(f"DONE {spec['job']}", flush=True)
    if failures:
        raise RuntimeError(f"Evaluation jobs failed: {failures}")
    merge()
    print(f"Wrote {OUTPUT_DIR / 'summary.md'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    cli = parse_args()
    if cli.job:
        evaluate_job(cli.job)
    else:
        run_all(max(1, cli.workers))
