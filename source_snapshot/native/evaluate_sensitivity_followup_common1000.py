#!/usr/bin/env python3
"""Common-demand reevaluation for the Sensitivity-4/12 follow-up actors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

import evaluate_manifest_kaynov_small as evaluator
import evaluate_stationary_manifest_dataset as stationary
from evaluation_statistics import mean_se_median


ROOT = Path(__file__).resolve().parent
PYTHON = "/usr/local/bin/python3"
SOURCE_EVAL = ROOT / "outputs/evaluation/sensitivity13_rasac_ro_common1000_20260802"
OUTPUT_DIR = ROOT / "outputs/evaluation/sensitivity_followup_common1000_20260802"
LOG_DIR = ROOT / "outputs/logs/sensitivity_followup_common1000_20260802_logs"

CANDIDATES = {
    "s4_cdc650": {
        "label": "RA-SAC cdc-min=650",
        "case": "sensitivity-4",
        "cwd": ROOT / "outputs/train/rasac_sensitivity_4_cdcmin650_600k_v2_20260802/sensitivity-4",
        "log": ROOT / "outputs/logs/sensitivity_followup_s4_s12_s13_v2_20260802_logs/rasac_sensitivity_4_cdcmin650_600k_v2_20260802.log",
    },
    "s4_cdc500": {
        "label": "RA-SAC cdc-min=500",
        "case": "sensitivity-4",
        "cwd": ROOT / "outputs/train/rasac_sensitivity_4_cdcmin500_600k_v2_20260802/sensitivity-4",
        "log": ROOT / "outputs/logs/sensitivity_followup_s4_s12_s13_v2_20260802_logs/rasac_sensitivity_4_cdcmin500_600k_v2_20260802.log",
    },
    "s12_g099_lr5e4": {
        "label": "RA-SAC gamma=.99 lr=5e-4",
        "case": "sensitivity-12",
        "cwd": ROOT / "outputs/train/rasac_sensitivity_12_gamma0p99_lr5em04_1m_v2_20260802/sensitivity-12",
        "log": ROOT / "outputs/logs/sensitivity_followup_s4_s12_s13_v2_20260802_logs/rasac_sensitivity_12_gamma0p99_lr5em04_1m_v2_20260802.log",
    },
    "s12_g099_lr2e4": {
        "label": "RA-SAC gamma=.99 lr=2e-4",
        "case": "sensitivity-12",
        "cwd": ROOT / "outputs/train/rasac_sensitivity_12_gamma0p99_lr2em04_1m_v2_20260802/sensitivity-12",
        "log": ROOT / "outputs/logs/sensitivity_followup_s4_s12_s13_v2_20260802_logs/rasac_sensitivity_12_gamma0p99_lr2em04_1m_v2_20260802.log",
    },
}

EPISODES = 1000
EPISODE_STEPS = 1000
WARMUP = 500


def best_actor(cwd: Path) -> tuple[Path, int, float]:
    pattern = re.compile(r"actor__([0-9]+)_(-?[0-9.]+)\.pt$")
    found = []
    for path in cwd.glob("actor__*.pt"):
        match = pattern.match(path.name)
        if match:
            found.append((float(match.group(2)), int(match.group(1)), path))
    if not found:
        raise FileNotFoundError(f"No scored actor under {cwd}")
    reward, step, path = max(found, key=lambda item: item[0])
    return path, step, reward


def training_protocol(log_path: Path) -> dict:
    protocol = None
    with log_path.open(encoding="utf-8", errors="ignore") as file:
        for raw in file:
            line = raw.strip()
            if line.startswith("Training dataset="):
                protocol = evaluator.parse_training_line(line)
            elif line.startswith("Action-space config:") and protocol is not None:
                protocol.update(evaluator.parse_action_space_line(line))
    if protocol is None:
        raise RuntimeError(f"Training protocol not found in {log_path}")
    return protocol


def prepare() -> None:
    for name in ("parts", "episode_means", "demand_banks", "job_logs"):
        (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for case in {spec["case"] for spec in CANDIDATES.values()}:
        source = SOURCE_EVAL / "demand_banks" / f"{case}.npz"
        target = OUTPUT_DIR / "demand_banks" / f"{case}.npz"
        if not target.exists():
            shutil.copy2(source, target)
    manifest = {
        "episodes": EPISODES,
        "episode_steps": EPISODE_STEPS,
        "warmup": WARMUP,
        "source_demand_bank": str(SOURCE_EVAL.relative_to(ROOT)),
        "candidates": {},
    }
    for key, spec in CANDIDATES.items():
        actor, step, reward = best_actor(spec["cwd"])
        manifest["candidates"][key] = {
            "label": spec["label"],
            "case": spec["case"],
            "actor": str(actor.relative_to(ROOT)),
            "best_step": step,
            "training_best_avgR": reward,
            "training_log": str(spec["log"].relative_to(ROOT)),
        }
    (OUTPUT_DIR / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def evaluate_one(key: str) -> None:
    spec = CANDIDATES[key]
    actor_path, step, training_reward = best_actor(spec["cwd"])
    protocol = training_protocol(spec["log"])
    stationary.configure("sensitivity13", ROOT / "outputs/valid_result_manifest_13x24.csv")
    bank_path = OUTPUT_DIR / "demand_banks" / f"{spec['case']}.npz"
    bank = evaluator.load_demand_bank(bank_path)
    env, _, input_args = evaluator.build_project_env(spec["case"], protocol, bank)
    costs = evaluator.evaluate_project_actor(
        spec["label"],
        actor_path,
        env,
        input_args,
        protocol,
        EPISODES,
        EPISODE_STEPS,
        WARMUP,
    )
    episode_means = costs.mean(axis=1)
    mean, se, median = mean_se_median(episode_means)
    np.savez_compressed(
        OUTPUT_DIR / "episode_means" / f"{key}.npz",
        episode_mean_cost=episode_means,
    )
    result = {
        "key": key,
        "label": spec["label"],
        "case": spec["case"],
        "mean_cost": mean,
        "se_cost": se,
        "median_cost": median,
        "episodes": EPISODES,
        "episode_steps": EPISODE_STEPS,
        "warmup_steps": WARMUP,
        "actor_path": str(actor_path.relative_to(ROOT)),
        "training_best_step": step,
        "training_best_avgR": training_reward,
        "training_log": str(spec["log"].relative_to(ROOT)),
        "demand_bank": str(bank_path.relative_to(ROOT)),
        "demand_bank_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
        "action_bounds": {
            "rdc_min": input_args["rdc_bsl_min_override"],
            "rdc_max": input_args["rdc_bsl_max_override"],
            "cdc_min": input_args["cdc_bsl_min_override"],
            "cdc_max": input_args["cdc_bsl_max_override"],
        },
    }
    (OUTPUT_DIR / "parts" / f"{key}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


def paired_comparison(left_key: str, right_key: str) -> dict:
    left = np.load(OUTPUT_DIR / "episode_means" / f"{left_key}.npz")["episode_mean_cost"]
    right = np.load(OUTPUT_DIR / "episode_means" / f"{right_key}.npz")["episode_mean_cost"]
    delta = left - right
    return {
        "left": left_key,
        "right": right_key,
        "mean_delta_left_minus_right": float(delta.mean()),
        "se_delta": float(delta.std(ddof=1) / np.sqrt(delta.size)),
        "left_better_fraction": float(np.mean(left < right)),
    }


def merge() -> None:
    rows = [
        json.loads((OUTPUT_DIR / "parts" / f"{key}.json").read_text(encoding="utf-8"))
        for key in CANDIDATES
    ]
    with (OUTPUT_DIR / "results.csv").open("w", newline="", encoding="utf-8") as file:
        fields = [
            "key", "label", "case", "mean_cost", "se_cost", "median_cost",
            "training_best_step", "training_best_avgR", "actor_path", "demand_bank_sha256",
        ]
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    comparisons = [
        paired_comparison("s4_cdc650", "s4_cdc500"),
        paired_comparison("s12_g099_lr5e4", "s12_g099_lr2e4"),
    ]
    (OUTPUT_DIR / "comparisons.json").write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    old_results = {
        (row["benchmark"], row["case"]): row
        for row in csv.DictReader((SOURCE_EVAL / "results.csv").open(encoding="utf-8"))
    }
    lines = [
        "# Sensitivity Follow-up Common-1000 Evaluation",
        "",
        "All candidate actors use the existing common-demand bank: 1,000 episodes, 1,000 periods per episode, first 500 periods discarded.",
        "",
        "| Candidate | Case | Mean cost | SE | Median | Training best step | Delta vs RO |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ro_mean = float(old_results[("RO", row["case"])]["mean_cost"])
        lines.append(
            f"| {row['label']} | {row['case']} | {row['mean_cost']:.6f} | "
            f"{row['se_cost']:.6f} | {row['median_cost']:.6f} | "
            f"{row['training_best_step']} | {row['mean_cost'] - ro_mean:+.6f} |"
        )
    lines += ["", "## Paired Candidate Comparisons", ""]
    for row in comparisons:
        lines.append(
            f"- `{row['left']}` minus `{row['right']}`: mean delta "
            f"{row['mean_delta_left_minus_right']:+.6f} (SE {row['se_delta']:.6f}); "
            f"left policy is better on {row['left_better_fraction']:.1%} of episodes."
        )
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_parallel(workers: int) -> None:
    prepare()
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }

    def run(key: str) -> tuple[str, int]:
        log_path = OUTPUT_DIR / "job_logs" / f"{key}.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [PYTHON, "-u", str(Path(__file__).resolve()), "--one", key],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
        return key, completed.returncode

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run, key): key for key in CANDIDATES}
        for future in as_completed(futures):
            key, returncode = future.result()
            print(f"{'DONE' if returncode == 0 else 'FAIL'} {key} rc={returncode}", flush=True)
            if returncode:
                failures.append(key)
    if failures:
        raise RuntimeError(f"Failed evaluations: {', '.join(failures)}")
    merge()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--one", choices=tuple(CANDIDATES))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.one:
        evaluate_one(args.one)
    else:
        run_parallel(args.workers)


if __name__ == "__main__":
    main()
