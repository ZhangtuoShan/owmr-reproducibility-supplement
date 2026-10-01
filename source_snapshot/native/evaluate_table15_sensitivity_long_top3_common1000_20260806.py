#!/usr/bin/env python3
"""Reevaluate the top three checkpoints from every Table15/Sensitivity13 long run."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

import evaluate_manifest_kaynov_small as evaluator
import evaluate_recent_sac_candidates_common_1000ep as common
import evaluate_stationary_manifest_dataset as stationary


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs/evaluation/table15_sensitivity13_long_top3_common1000_20260806"
TRAIN_ROOT = ROOT / "outputs/train"
PREFIX = "rasac_all_cases_long_from_scratch_20260803"
MANIFEST = ROOT / "outputs/valid_result_manifest_13x24.csv"
TABLE_BANKS = ROOT / "outputs/evaluation/manifest_pending_common1000_20260802/demand_banks"
SENSITIVITY_BANKS = ROOT / "outputs/evaluation/sensitivity13_rasac_ro_common1000_20260802/demand_banks"
ACTOR_RE = re.compile(r"actor__(\d+)_([+-]?\d+(?:\.\d+)?)\.pt$")
TABLE_CASE_RE = re.compile(r"^(2|6|20)-([1-5])$")
SENSITIVITY_CASE_RE = re.compile(r"^sensitivity-(\d+)$")
EPISODES = 1000
EPISODE_STEPS = 1000
WARMUP_STEPS = 500


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def parse_experiment(directory: Path) -> tuple[str, str, str, str] | None:
    for method in ("eq", "ineq"):
        marker = f"{PREFIX}_{method}_"
        if not directory.name.startswith(marker):
            continue
        suffix = directory.name[len(marker):]
        sensitivity = re.match(r"^(sensitivity-\d+)_(.+)$", suffix)
        if sensitivity:
            return "sensitivity13", method, sensitivity.group(1), sensitivity.group(2)
        table = re.match(r"^((?:2|6|20)-[1-5])_(.+)$", suffix)
        if table:
            return "table15", method, table.group(1), table.group(2)
    return None


def ranked_actors(directory: Path) -> list[tuple[float, int, Path]]:
    actors: list[tuple[float, int, Path]] = []
    for actor in directory.rglob("actor__*.pt"):
        match = ACTOR_RE.fullmatch(actor.name)
        if match:
            actors.append((float(match.group(2)), int(match.group(1)), actor))
    actors.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if len(actors) < 3:
        raise RuntimeError(f"Fewer than three scored actors under {directory}: {len(actors)}")
    return actors


def discover_specs() -> list[dict[str, Any]]:
    training_index = evaluator.index_training_logs()
    specs: list[dict[str, Any]] = []
    experiment_counts = {"table15": 0, "sensitivity13": 0}
    for directory in sorted(TRAIN_ROOT.glob(f"{PREFIX}_*")):
        parsed = parse_experiment(directory)
        if parsed is None:
            continue
        family, method, case, variant = parsed
        experiment_counts[family] += 1
        benchmark = "RA-SAC" if method == "eq" else "RA-SAC ineq."
        bank = (TABLE_BANKS if family == "table15" else SENSITIVITY_BANKS) / f"{case}.npz"
        if not bank.exists():
            raise FileNotFoundError(bank)
        for rank, (avg_r, step, actor) in enumerate(ranked_actors(directory)[:3], start=1):
            entry = training_index.get(str(actor.parent.resolve()))
            if entry is None:
                raise KeyError(f"Training protocol not found for {actor.parent}")
            log_path, protocol = entry
            dataset = str(protocol.get("dataset", family))
            if family == "sensitivity13" and dataset not in {
                "sensitivity13",
                "sensitivity13_corrected",
            }:
                dataset = "sensitivity13_corrected"
            job = safe_name(f"{family}_{method}_{case}_{variant}_rank{rank}")
            specs.append(
                {
                    "job": job,
                    "family": family,
                    "dataset": dataset,
                    "benchmark": benchmark,
                    "method": method,
                    "case": case,
                    "variant": variant,
                    "experiment": directory.name,
                    "training_rank": rank,
                    "training_avgR": avg_r,
                    "training_step": step,
                    "actor": str(actor.relative_to(ROOT)),
                    "training_log": str(log_path.relative_to(ROOT)),
                    "protocol": evaluator.json_safe(protocol),
                    "demand_bank": str(bank.relative_to(ROOT)),
                }
            )
    expected_experiments = {"table15": 36, "sensitivity13": 32}
    if experiment_counts != expected_experiments:
        raise RuntimeError(
            f"Unexpected experiment counts: {experiment_counts}; expected {expected_experiments}"
        )
    if len(specs) != 204:
        raise RuntimeError(f"Expected 204 top-three jobs, found {len(specs)}")
    jobs = [spec["job"] for spec in specs]
    actors = [spec["actor"] for spec in specs]
    if len(jobs) != len(set(jobs)):
        raise RuntimeError("Duplicate evaluation job names discovered")
    if len(actors) != len(set(actors)):
        raise RuntimeError("Duplicate actor checkpoints discovered")
    return specs


def prepare() -> list[dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("parts", "episode_means", "job_logs"):
        (OUTPUT_DIR / name).mkdir(exist_ok=True)
    specs = discover_specs()
    (OUTPUT_DIR / "specs.json").write_text(
        json.dumps(specs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return specs


def load_specs() -> list[dict[str, Any]]:
    return json.loads((OUTPUT_DIR / "specs.json").read_text(encoding="utf-8"))


def evaluate_job(job_name: str) -> None:
    torch.set_num_threads(1)
    spec = next(item for item in load_specs() if item["job"] == job_name)
    bank = evaluator.load_demand_bank(ROOT / spec["demand_bank"])
    first = next(iter(bank.values()))
    if first.shape != (EPISODES, EPISODE_STEPS):
        raise ValueError(f"Unexpected bank shape for {spec['case']}: {first.shape}")
    protocol = common.restore_bound_key_types(spec["protocol"])
    if spec["family"] == "table15":
        env, input_args = common.build_table_env(spec["case"], protocol, bank)
    else:
        stationary.configure(spec["dataset"], MANIFEST)
        env, _, input_args = evaluator.build_project_env(spec["case"], protocol, bank)
    costs = evaluator.evaluate_project_actor(
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
    np.savez_compressed(
        OUTPUT_DIR / "episode_means" / f"{job_name}.npz",
        episode_means=episode_means,
    )
    result = {
        **{key: spec[key] for key in (
            "job", "family", "dataset", "benchmark", "method", "case", "variant",
            "experiment", "training_rank", "training_avgR", "training_step", "actor",
            "training_log", "demand_bank",
        )},
        **evaluator.summarize_costs(costs),
        "episodes": EPISODES,
        "episode_steps": EPISODE_STEPS,
        "warmup_steps": WARMUP_STEPS,
    }
    (OUTPUT_DIR / "parts" / f"{job_name}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


def merge() -> None:
    specs = load_specs()
    missing = [
        spec["job"] for spec in specs
        if not (OUTPUT_DIR / "parts" / f"{spec['job']}.json").exists()
    ]
    if missing:
        raise RuntimeError(f"Cannot merge with {len(missing)} missing parts")
    rows = [
        json.loads((OUTPUT_DIR / "parts" / f"{spec['job']}.json").read_text(encoding="utf-8"))
        for spec in specs
    ]
    fields = list(rows[0])
    with (OUTPUT_DIR / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    winners: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for experiment in sorted({row["experiment"] for row in rows}):
        candidates = [row for row in rows if row["experiment"] == experiment]
        winner = min(candidates, key=lambda row: (row["mean_cost"], row["training_rank"]))
        winners.append(winner)
        rank1 = next(row for row in candidates if row["training_rank"] == 1)
        with np.load(OUTPUT_DIR / "episode_means" / f"{rank1['job']}.npz") as data:
            rank1_means = np.asarray(data["episode_means"], dtype=float)
        for candidate in sorted(candidates, key=lambda row: row["training_rank"]):
            with np.load(OUTPUT_DIR / "episode_means" / f"{candidate['job']}.npz") as data:
                means = np.asarray(data["episode_means"], dtype=float)
            delta = means - rank1_means
            comparisons.append(
                {
                    "experiment": experiment,
                    "case": candidate["case"],
                    "benchmark": candidate["benchmark"],
                    "training_rank": candidate["training_rank"],
                    "training_step": candidate["training_step"],
                    "training_avgR": candidate["training_avgR"],
                    "mean_cost": candidate["mean_cost"],
                    "se_cost": candidate["se_cost"],
                    "median_cost": candidate["median_cost"],
                    "candidate_minus_rank1": float(delta.mean()),
                    "paired_se": common.paired_se(delta),
                    "better_episode_fraction": float(np.mean(delta < 0.0)),
                    "actor": candidate["actor"],
                }
            )
    with (OUTPUT_DIR / "winners.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(winners[0]))
        writer.writeheader()
        writer.writerows(winners)
    (OUTPUT_DIR / "winners.json").write_text(
        json.dumps(winners, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (OUTPUT_DIR / "comparisons.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    refreshed = [row for row in winners if row["training_rank"] != 1]
    lines = [
        "# Table15 and Sensitivity13 Long-Run Top-3 Reevaluation",
        "",
        f"Protocol: {EPISODES} common-demand episodes, {EPISODE_STEPS} periods, "
        f"statistics over periods {WARMUP_STEPS + 1}-{EPISODE_STEPS}.",
        "",
        f"- Experiments: {len(winners)}",
        f"- Evaluated checkpoints: {len(rows)}",
        f"- Formal winners not equal to training rank 1: {len(refreshed)}",
        "",
        "| Family | Method | Case | Variant | Formal winner rank | Step | Mean | SE | Median |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in winners:
        lines.append(
            f"| {row['family']} | {row['benchmark']} | {row['case']} | {row['variant']} | "
            f"{row['training_rank']} | {row['training_step']} | {row['mean_cost']:.6f} | "
            f"{row['se_cost']:.6f} | {row['median_cost']:.6f} |"
        )
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--job")
    mode.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        specs = prepare()
        print(json.dumps({"experiments": 68, "jobs": len(specs)}, ensure_ascii=False))
    elif args.job:
        evaluate_job(args.job)
    else:
        merge()


if __name__ == "__main__":
    main()
