#!/usr/bin/env python3
"""Common-random-number evaluation for four methods on all Table 15 cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

from Benchmark import (
    OWMRInventoryEnv,
    auto_hyperparameter,
    env_input_initialize,
    generate_env,
    heuristic_order_qty,
    recursive_heuristic_tail_fixed,
    relaxation_based_heuristic_distribution_aware_capped,
)
from dataset import data_random_table_benchmark_metadata, data_random_table_benchmarks
from demand_utils import sample_demand
from evaluate_manifest_kaynov_small import (
    evaluate_project_actor,
    json_safe,
    load_demand_bank,
    summarize_costs,
)
from evaluate_recent_sac_candidates_common_1000ep import (
    parse_table_training_logs,
    restore_bound_key_types,
)
from main_DRL_agents import get_env_args


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs/evaluation/table15_four_methods_common1000_20260731"
DEFAULT_LOG_DIR = ROOT / "outputs/logs/table15_four_methods_common1000_20260731_logs"
DEFAULT_EPISODES = 1000
DEFAULT_EPISODE_STEPS = 1000
DEFAULT_WARMUP_STEPS = 500
DEFAULT_SEED = 20260731

METHODS = ("RA-SAC", "RA-SAC ineq.", "RO", "RB")
METHOD_SLUGS = {
    "RA-SAC": "ra_sac",
    "RA-SAC ineq.": "ra_sac_ineq",
    "RO": "ro",
    "RB": "rb",
}
SLUG_METHODS = {value: key for key, value in METHOD_SLUGS.items()}
CASES = tuple(row["case"] for row in data_random_table_benchmark_metadata())

# These checkpoints improved the corresponding manifest training score during
# the completed 2M continuation. Other cells retain the manifest selection,
# including candidates already selected by a prior independent evaluation.
ACTOR_OVERRIDES = {
    ("RA-SAC", "2-1"): "outputs/train/sac_eq_2_1_exactalloc_lr0p0005_rs0p00195312_g0p96_rt1_1m_stable_resume2m_20260731_rngfixed_queue/2-1/actor__000001884672_-0030.751.pt",
    ("RA-SAC", "6-1"): "outputs/train/sac_eq_6_1_exactalloc_lr0p0002_rs0p00195312_g0p96_rt1_1m_stable_resume2m_20260731_rngfixed_queue/6-1/actor__000001372672_-0307.535.pt",
    ("RA-SAC", "6-2"): "outputs/train/sac_eq_6_2_exactalloc_lr0p0002_rs0p00195312_g0p96_rt1_1m_stable_resume2m_20260731_rngfixed_queue/6-2/actor__000001137152_-0248.497.pt",
    ("RA-SAC", "6-3"): "outputs/train/sac_eq_6_3_exactalloc_lowlr_combo03_lr0p0001_g0p96_rt0p5_1p5m_stable_resume2m_20260731_rngfixed_queue/6-3/actor__000001628672_-0403.138.pt",
    ("RA-SAC", "6-4"): "outputs/train/sac_eq_6_4_exactalloc_lr0p0002_rs0p00195312_g0p96_rt1_1m_stable_resume2m_20260731_rngfixed_queue/6-4/actor__000001065472_-0268.925.pt",
    ("RA-SAC", "6-5"): "outputs/train/sac_eq_6_5_exactalloc_lr0p0002_rs0p00195312_g0p96_rt1_1m_stable_resume2m_20260731_rngfixed_queue/6-5/actor__000001024512_-0225.034.pt",
    ("RA-SAC ineq.", "6-1"): "outputs/train/sac_ineq_6_1_exactalloc_lr0p0003_rs0p00195312_g0p96_rt0p5_1m_stable_resume2m_20260731_rngfixed_queue/6-1/actor__000001096192_-0306.250.pt",
    ("RA-SAC ineq.", "6-4"): "outputs/train/sac_ineq_6_4_exactalloc_lr0p0003_rs0p00195312_g0p96_rt0p5_1m_stable_resume2m_20260731_rngfixed_queue/6-4/actor__000001075712_-0266.037.pt",
}


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def table_case_index(case: str) -> int:
    return CASES.index(case)


def demand_bank_path(output_dir: Path, case: str) -> Path:
    return output_dir / "demand_banks" / f"{case}.npz"


def build_demand_bank(case: str, episodes: int, steps: int, seed: int) -> dict[int, np.ndarray]:
    data = data_random_table_benchmarks()[table_case_index(case)]
    info = env_input_initialize(*data)[0]
    rng = np.random.default_rng(seed + table_case_index(case) * 1009)
    bank: dict[int, np.ndarray] = {}
    for rdc in info.rdc_dict:
        values = np.asarray(
            sample_demand(
                info.demand_feature[rdc],
                size=(episodes, steps),
                random_state=rng,
            ),
            dtype=np.float64,
        )
        values[:, 0] = 0.0
        bank[int(rdc)] = values
    return bank


def save_demand_bank(path: Path, bank: dict[int, np.ndarray], case: str, seed: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"rdc_{rdc}": values for rdc, values in bank.items()})
    first = next(iter(bank.values()))
    return {
        "case": case,
        "seed": seed + table_case_index(case) * 1009,
        "episodes": int(first.shape[0]),
        "episode_steps": int(first.shape[1]),
        "rdcs": sorted(bank),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "path": relative(path),
    }


def selected_manifest_rows() -> dict[tuple[str, str], dict[str, str]]:
    path = ROOT / "outputs/valid_result_manifest_13x24.csv"
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {
        (row["benchmark"], row["case"]): row
        for row in rows
        if row["benchmark"] in METHODS and row["case"] in CASES
    }


def build_specs(cases: tuple[str, ...], methods: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = selected_manifest_rows()
    training_index = parse_table_training_logs()
    specs: list[dict[str, Any]] = []
    for case in cases:
        for benchmark in methods:
            job = f"{METHOD_SLUGS[benchmark]}__{case.replace('-', '_')}"
            spec: dict[str, Any] = {
                "job": job,
                "case": case,
                "benchmark": benchmark,
                "kind": "actor" if benchmark.startswith("RA-SAC") else "heuristic",
            }
            if spec["kind"] == "actor":
                row = rows.get((benchmark, case))
                if row is None:
                    raise KeyError(f"Manifest has no {benchmark} / {case} row")
                actor_rel = ACTOR_OVERRIDES.get((benchmark, case), row["checkpoint_or_policy"])
                actor = (ROOT / actor_rel).resolve()
                if not actor.exists():
                    raise FileNotFoundError(actor)
                entry = training_index.get(str(actor.parent.resolve()))
                if entry is None:
                    raise KeyError(f"No training protocol found for {actor.parent}")
                training_log, protocol = entry
                spec.update(
                    {
                        "actor": relative(actor),
                        "training_log": relative(training_log),
                        "protocol": json_safe(protocol),
                        "selection_source": row["source"],
                        "manifest_best_avgR": row["best_avgR"],
                        "overrides_manifest_actor": (benchmark, case) in ACTOR_OVERRIDES,
                    }
                )
            specs.append(spec)
    expected = len(cases) * len(methods)
    if len(specs) != expected:
        raise RuntimeError(f"Expected {expected} specs, found {len(specs)}")
    return specs


def prepare(
    output_dir: Path,
    cases: tuple[str, ...],
    methods: tuple[str, ...],
    episodes: int,
    steps: int,
    seed: int,
) -> None:
    for name in ("parts", "episode_means", "job_logs", "demand_banks"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    specs = build_specs(cases, methods)
    metadata: list[dict[str, Any]] = []
    for case in cases:
        path = demand_bank_path(output_dir, case)
        if path.exists():
            bank = load_demand_bank(path)
            shape = next(iter(bank.values())).shape
            if shape != (episodes, steps):
                raise ValueError(f"Existing {case} demand bank has shape {shape}, expected {(episodes, steps)}")
            metadata.append(
                {
                    "case": case,
                    "seed": seed + table_case_index(case) * 1009,
                    "episodes": episodes,
                    "episode_steps": steps,
                    "rdcs": sorted(bank),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "path": relative(path),
                }
            )
        else:
            metadata.append(save_demand_bank(path, build_demand_bank(case, episodes, steps, seed), case, seed))
    (output_dir / "specs.json").write_text(
        json.dumps(specs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "demand_banks.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(specs)} jobs and {len(metadata)} common demand banks", flush=True)


def load_specs(output_dir: Path) -> list[dict[str, Any]]:
    return json.loads((output_dir / "specs.json").read_text(encoding="utf-8"))


def build_env(case: str, protocol: dict[str, Any], bank: dict[int, np.ndarray], seed: int, steps: int):
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
        "seed": seed,
        "rho": float(protocol.get("rho", 1.0)),
        "allocation_rule": protocol.get("allocation_rule", "prioritized_greedy"),
        "demand_fulfillment": protocol.get("demand_fulfillment", "partial_backorder"),
        "emergency_shipment_prob": protocol.get("emergency_shipment_prob", 1.0),
        "rdc_bsl_min_override": protocol.get("rdc_bsl_min_override"),
        "cdc_bsl_min_override": protocol.get("cdc_bsl_min_override"),
        "rdc_bsl_max_override": protocol.get("rdc_bsl_max_override"),
        "cdc_bsl_max_override": protocol.get("cdc_bsl_max_override"),
    }
    env = generate_env(OWMRInventoryEnv, get_env_args(input_args))
    env.total_periods = steps
    env.episode = -1
    return env, info, input_args


def evaluate_heuristic(
    benchmark: str,
    case: str,
    bank: dict[int, np.ndarray],
    episodes: int,
    steps: int,
    warmup: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    protocol = {
        "allocation_rule": "proportion",
        "rho": 1.0,
        "demand_fulfillment": "partial_backorder",
        "emergency_shipment_prob": 1.0,
    }
    env, info, _ = build_env(case, protocol, bank, seed, steps)
    env.method = "heu"
    if benchmark == "RO":
        rdc_bsl, _, cdc_bsl = recursive_heuristic_tail_fixed(info)
        method_alias = "RO_tail_fixed"
    elif benchmark == "RB":
        _, rdc_bsl, cdc_bsl = relaxation_based_heuristic_distribution_aware_capped(info)
        method_alias = "RB_distribution_aware_capped"
    else:
        raise ValueError(benchmark)
    costs = np.empty((episodes, steps - warmup), dtype=np.float64)
    for episode in range(episodes):
        env.reset()
        for period in range(steps):
            action = heuristic_order_qty(env, info, rdc_bsl, cdc_bsl, method=method_alias)
            _, _, done, info_out = env.step(action)
            if period >= warmup:
                costs[episode, period - warmup] = sum(float(value) for value in info_out[0].values())
            if done and period + 1 != steps:
                raise RuntimeError(f"{benchmark} {case} ended at period {period + 1}")
    policy = {
        "method_alias": method_alias,
        "rdc_bsl": json_safe(rdc_bsl),
        "cdc_bsl": float(cdc_bsl),
        **protocol,
    }
    return costs, policy


def evaluate_job(
    output_dir: Path,
    job_name: str,
    episodes: int,
    steps: int,
    warmup: int,
    seed: int,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    spec = next(item for item in load_specs(output_dir) if item["job"] == job_name)
    bank_rel = demand_bank_path(output_dir, spec["case"])
    bank = load_demand_bank(bank_rel)
    if spec["kind"] == "actor":
        protocol = restore_bound_key_types(spec["protocol"])
        env, _, input_args = build_env(spec["case"], protocol, bank, seed, steps)
        costs = evaluate_project_actor(
            spec["benchmark"],
            ROOT / spec["actor"],
            env,
            input_args,
            protocol,
            episodes,
            steps,
            warmup,
        )
        policy: dict[str, Any] = {
            "actor": spec["actor"],
            "training_log": spec["training_log"],
            "allocation_rule": protocol.get("allocation_rule"),
        }
    else:
        costs, policy = evaluate_heuristic(
            spec["benchmark"], spec["case"], bank, episodes, steps, warmup, seed
        )
    episode_means = costs.mean(axis=1)
    np.savez_compressed(output_dir / "episode_means" / f"{job_name}.npz", episode_means=episode_means)
    result = {
        "job": job_name,
        "case": spec["case"],
        "benchmark": spec["benchmark"],
        **summarize_costs(costs),
        "best_avgR_equivalent": -float(costs.mean()),
        "episodes": episodes,
        "episode_steps": steps,
        "warmup_steps": warmup,
        "demand_bank": relative(bank_rel),
        "policy": policy,
    }
    (output_dir / "parts" / f"{job_name}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def paired_se(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(values.size))


def merge(output_dir: Path) -> None:
    specs = load_specs(output_dir)
    active_cases = tuple(case for case in CASES if any(spec["case"] == case for spec in specs))
    active_methods = tuple(method for method in METHODS if any(spec["benchmark"] == method for spec in specs))
    results: list[dict[str, Any]] = []
    for spec in specs:
        part = output_dir / "parts" / f"{spec['job']}.json"
        if not part.exists():
            raise FileNotFoundError(part)
        results.append(json.loads(part.read_text(encoding="utf-8")))
    order = {(case, method): index for index, (case, method) in enumerate(
        (pair for case in active_cases for pair in ((case, method) for method in active_methods))
    )}
    results.sort(key=lambda row: order[(row["case"], row["benchmark"])])
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("case", "benchmark", "mean_cost", "se_cost", "median_cost", "best_avgR_equivalent", "job"),
        )
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    result_index = {(row["case"], row["benchmark"]): row for row in results}
    paired_rows: list[dict[str, Any]] = []
    comparisons = (("RA-SAC", "RO"), ("RA-SAC", "RB"), ("RA-SAC ineq.", "RO"), ("RA-SAC ineq.", "RB"), ("RA-SAC ineq.", "RA-SAC"))
    for case in active_cases:
        for method, reference in comparisons:
            if method not in active_methods or reference not in active_methods:
                continue
            left_job = result_index[(case, method)]["job"]
            right_job = result_index[(case, reference)]["job"]
            left = np.load(output_dir / "episode_means" / f"{left_job}.npz")["episode_means"]
            right = np.load(output_dir / "episode_means" / f"{right_job}.npz")["episode_means"]
            delta = left - right
            paired_rows.append(
                {
                    "case": case,
                    "method": method,
                    "reference": reference,
                    "mean_cost_delta": float(delta.mean()),
                    "paired_se": paired_se(delta),
                    "win_fraction": float(np.mean(delta < 0.0)),
                }
            )
    with (output_dir / "paired_comparisons.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    lines = [
        "# Table 15 Four-Method Common-Demand Evaluation",
        "",
        f"Protocol: {results[0]['episodes']} common episodes per case, {results[0]['episode_steps']} periods per episode, "
        f"statistics over periods {results[0]['warmup_steps'] + 1}-{results[0]['episode_steps']}.",
        "",
        "| Case | Method | Mean cost | SE | Median cost |",
        "|---|---|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['case']} | {row['benchmark']} | {row['mean_cost']:.6f} | {row['se_cost']:.6f} | {row['median_cost']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Negative paired cost differences favor the listed method.",
            "",
            "| Case | Method | Reference | Mean paired delta | Paired SE | Win fraction |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in paired_rows:
        lines.append(
            f"| {row['case']} | {row['method']} | {row['reference']} | "
            f"{row['mean_cost_delta']:.6f} | {row['paired_se']:.6f} | {row['win_fraction']:.3f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Merged {len(results)} results into {output_dir / 'summary.md'}", flush=True)


def append_status(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")
        file.flush()


def run_all(args: argparse.Namespace) -> None:
    prepare(args.output_dir, args.cases, args.methods, args.episodes, args.episode_steps, args.seed)
    specs = load_specs(args.output_dir)
    status_path = args.log_dir / "parallel_status.log"
    append_status(status_path, f"PLAN total={len(specs)} workers={args.workers}")

    def run_spec(spec: dict[str, Any]) -> tuple[str, int]:
        job = spec["job"]
        part = args.output_dir / "parts" / f"{job}.json"
        if part.exists():
            return job, 0
        log_path = args.output_dir / "job_logs" / f"{job}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--job",
            job,
            "--output-dir",
            str(args.output_dir),
            "--log-dir",
            str(args.log_dir),
            "--episodes",
            str(args.episodes),
            "--episode-steps",
            str(args.episode_steps),
            "--warmup-steps",
            str(args.warmup_steps),
            "--seed",
            str(args.seed),
        ]
        append_status(status_path, f"START {job}")
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(command, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT, check=False)
        return job, completed.returncode

    done = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_spec, spec): spec for spec in specs}
        for future in as_completed(futures):
            job, returncode = future.result()
            if returncode == 0:
                done += 1
                append_status(status_path, f"DONE {job} done={done}/{len(specs)}")
            else:
                failures += 1
                append_status(status_path, f"FAIL {job} rc={returncode}")
    append_status(status_path, f"SUMMARY total={len(specs)} done={done} failures={failures}")
    if failures:
        raise SystemExit(1)
    merge(args.output_dir)


def parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--job")
    mode.add_argument("--merge", action="store_true")
    mode.add_argument("--run-all", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--episode-steps", type=int, default=DEFAULT_EPISODE_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--methods", default=",".join(METHOD_SLUGS.values()))
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.log_dir = args.log_dir.resolve()
    args.cases = parse_csv_tuple(args.cases)
    method_slugs = parse_csv_tuple(args.methods)
    unknown_cases = set(args.cases) - set(CASES)
    unknown_methods = set(method_slugs) - set(SLUG_METHODS)
    if unknown_cases or unknown_methods:
        parser.error(f"Unknown cases={sorted(unknown_cases)} methods={sorted(unknown_methods)}")
    args.methods = tuple(SLUG_METHODS[slug] for slug in method_slugs)
    if not 0 <= args.warmup_steps < args.episode_steps:
        parser.error("warmup-steps must be in [0, episode-steps)")
    return args


def main() -> None:
    args = parse_args()
    if args.prepare:
        prepare(args.output_dir, args.cases, args.methods, args.episodes, args.episode_steps, args.seed)
    elif args.job:
        evaluate_job(args.output_dir, args.job, args.episodes, args.episode_steps, args.warmup_steps, args.seed)
    elif args.merge:
        merge(args.output_dir)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
