#!/usr/bin/env python3
"""Run corrected RO, then RA-SAC, on the 13 sensitivity instances."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from Benchmark import compute_table15_size_aware_action_space_overrides, env_input_initialize
from dataset import data_sensitivity_analysis_13, data_sensitivity_analysis_13_metadata
from output_paths import output_path


ROOT = Path(__file__).resolve().parent
PYTHON = "/usr/local/bin/python3"
TRAIN_RUNNER = ROOT / "run_table_single_combo.py"
HEURISTIC_RUNNER = ROOT / "run_table_heuristic_eval.py"
SUMMARIZER = ROOT / "summarize_grid_logs.py"
RUN_TAG = "sensitivity13_rasac_ro_20260801"
DEFAULT_LOG_DIR = f"{RUN_TAG}_logs"

BASE_REWARD_SCALE = 2.0 ** -9
MIN_REWARD_EXPONENT = -18
MAX_REWARD_EXPONENT = -9


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def append_status(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(f"[{now()}] {message.rstrip()}\n")
        file.flush()


def training_env() -> dict[str, str]:
    cache = ROOT / "outputs" / "cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    return {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "MPLCONFIGDIR": str(cache),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def ro_tag(case_index: int) -> str:
    return f"{RUN_TAG}_ro_parts/case{case_index:02d}_RO_tail_fixed"


def ro_csv(case_index: int) -> Path:
    return output_path("evaluation", f"{ro_tag(case_index)}.csv")


def ro_completed(case_index: int) -> bool:
    path = ro_csv(case_index)
    if not path.exists():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as file:
            row = next(csv.DictReader(file))
        return bool(row.get("avg_cost")) and bool(row.get("se_cost"))
    except (OSError, StopIteration):
        return False


def build_ro_jobs(args: argparse.Namespace) -> list[dict]:
    jobs = []
    for case_index, metadata in enumerate(data_sensitivity_analysis_13_metadata()):
        name = f"ro_{metadata['case'].replace('-', '_')}"
        command = [
            PYTHON, "-u", str(HEURISTIC_RUNNER),
            "--dataset", "sensitivity13",
            "--method", "RO_tail_fixed",
            "--case-index", str(case_index),
            "--eval-times", str(args.eval_times),
            "--total-periods", str(args.total_periods),
            "--warmup", str(args.warmup),
            "--allocation-rule", "proportion",
            "--seed", str(args.seed),
            "--tag", ro_tag(case_index),
        ]
        jobs.append({
            "name": name,
            "case_index": case_index,
            "case": metadata["case"],
            "command": command,
        })
    return jobs


def run_job(job: dict, log_dir: Path, status_path: Path, stage: str) -> dict:
    log_path = log_dir / f"{job['name']}.log"
    append_status(status_path, f"START stage={stage} job={job['name']} log={log_path}")
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("CMD: " + " ".join(job["command"]) + "\n")
        log_file.flush()
        proc = subprocess.run(
            job["command"],
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=training_env(),
        )
    result = {
        **job,
        "returncode": proc.returncode,
        "elapsed_sec": time.time() - started,
        "log": str(log_path),
    }
    label = "DONE" if proc.returncode == 0 else "FAIL"
    append_status(
        status_path,
        f"{label} stage={stage} job={job['name']} rc={proc.returncode} "
        f"elapsed_sec={result['elapsed_sec']:.1f}",
    )
    return result


def run_parallel(
        jobs: list[dict], workers: int, log_dir: Path,
        status_path: Path, stage: str) -> list[dict]:
    if not jobs:
        return []
    results = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {
            executor.submit(run_job, job, log_dir, status_path, stage): job
            for job in jobs
        }
        for future in as_completed(futures):
            results.append(future.result())
    failures = [item for item in results if item["returncode"] != 0]
    if failures:
        names = ", ".join(item["name"] for item in failures)
        raise RuntimeError(f"{stage} failed for: {names}")
    return results


def read_ro_rows() -> list[dict]:
    rows = []
    for case_index in range(13):
        with ro_csv(case_index).open(newline="", encoding="utf-8") as file:
            rows.append(next(csv.DictReader(file)))
    return rows


def merge_ro_results(rows: list[dict]) -> None:
    csv_path = output_path("evaluation", f"{RUN_TAG}_ro.csv")
    summary_path = output_path("summaries", f"{RUN_TAG}_ro.md")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Sensitivity13 corrected RO evaluation",
        "",
        "RO_tail_fixed with proportional allocation; uncertainty is the SE of episode-average cost.",
        "",
        "| Case | Mean cost | SE | Median | CDC BSL | RDC BSL |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {float(row['avg_cost']):.3f} | "
            f"{float(row['se_cost']):.3f} | {float(row['median_cost']):.3f} | "
            f"{float(row['warehouse_bsl']):.3f} | `{row['retailer_bsl']}` |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_reward_scale(base_cost: float, case_cost: float) -> tuple[float, int]:
    """Keep scaled RO cost near the baseline 2-RDC training scale."""
    if not math.isfinite(case_cost) or case_cost <= 0.0:
        raise ValueError(f"Invalid RO cost for reward scaling: {case_cost}")
    target_scaled_cost = BASE_REWARD_SCALE * base_cost
    raw_scale = target_scaled_cost / case_cost
    exponent = int(round(math.log2(raw_scale)))
    exponent = min(MAX_REWARD_EXPONENT, max(MIN_REWARD_EXPONENT, exponent))
    return 2.0 ** exponent, exponent


def rasac_cwd_name(case_name: str) -> str:
    return f"rasac_{case_name.replace('-', '_')}_2rdc_transfer_scaled_1m_20260801"


def rasac_cwd(case_name: str) -> Path:
    return output_path("train", rasac_cwd_name(case_name), case_name)


def rasac_completed(case_name: str, break_step: int) -> bool:
    cwd = rasac_cwd(case_name)
    recorder_path = cwd / "recorder.npy"
    if not recorder_path.exists() or not (cwd / "act.pth").exists():
        return False
    try:
        recorder = np.load(recorder_path)
        return bool(recorder.size) and float(recorder[-1, 0]) >= 0.98 * break_step
    except (OSError, ValueError, IndexError):
        return False


def build_rasac_jobs(args: argparse.Namespace, ro_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    cases = data_sensitivity_analysis_13()
    metadata_rows = data_sensitivity_analysis_13_metadata()
    base_cost = float(ro_rows[0]["avg_cost"])
    jobs = []
    manifest_rows = []
    for case_index, (data, metadata, ro_row) in enumerate(zip(cases, metadata_rows, ro_rows)):
        info_input, _, _ = env_input_initialize(*data)
        rdc_min, cdc_min, rdc_max, cdc_max, action_metadata = (
            compute_table15_size_aware_action_space_overrides(
                info_input,
                return_metadata=True,
            )
        )
        reward_scale, reward_exponent = choose_reward_scale(
            base_cost,
            float(ro_row["avg_cost"]),
        )
        case_name = metadata["case"]
        cwd_name = rasac_cwd_name(case_name)
        command = [
            PYTHON, "-u", str(TRAIN_RUNNER),
            "--dataset", "sensitivity13",
            "--benchmark", "ra_sac",
            "--case-index", str(case_index),
            "--lr", "0.0005",
            "--reward-scale", str(reward_scale),
            "--gamma", "0.96",
            "--repeat-times", "1.0",
            "--net-dims", "64,32",
            "--batch-size", "64",
            "--horizon-len", "512",
            "--buffer-size", "100000",
            "--soft-update-tau", "0.005",
            "--clip-grad-norm", "3.0",
            "--replay-warmup-steps", "10000",
            "--freeze-state-normalization-from-warmup",
            "--alpha-learning-rate", "2.5e-5",
            "--break-step", str(args.break_step),
            "--training-eval-profile", "short",
            "--eval-per-step", "10000",
            "--eval-times", str(args.eval_times),
            "--eval-episode-steps", "300",
            "--eval-burn-in-steps", "150",
            "--random-seed", "0",
            "--num-threads", "1",
            "--device", "cpu",
            "--allocation-rule", "prioritized_greedy",
            "--table15-action-space-profile", "size_aware_v1",
            "--save-best-training-state",
            "--run-tag", RUN_TAG,
            "--cwd-name", cwd_name,
        ]
        job = {
            "name": f"rasac_{case_name.replace('-', '_')}",
            "case_index": case_index,
            "case": case_name,
            "command": command,
        }
        jobs.append(job)
        manifest_rows.append({
            **metadata,
            "ro_mean_cost": float(ro_row["avg_cost"]),
            "reward_scale": reward_scale,
            "reward_scale_exponent": reward_exponent,
            "scaled_ro_cost": reward_scale * float(ro_row["avg_cost"]),
            "hyperparameters": {
                "learning_rate": 5e-4,
                "gamma": 0.96,
                "repeat_times": 1.0,
                "net_dims": [64, 32],
                "batch_size": 64,
                "horizon_len": 512,
                "buffer_size": 100000,
                "soft_update_tau": 0.005,
                "clip_grad_norm": 3.0,
                "replay_warmup_steps": 10000,
                "fixed_state_normalization": True,
                "alpha_learning_rate": 2.5e-5,
                "break_step": args.break_step,
            },
            "action_space": {
                "rdc_min": rdc_min,
                "rdc_max": rdc_max,
                "cdc_min": cdc_min,
                "cdc_max": cdc_max,
                **action_metadata,
            },
            "cwd": str(rasac_cwd(case_name)),
            "command": command,
        })
    return jobs, manifest_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ro-workers", type=int, default=6)
    parser.add_argument("--break-step", type=int, default=1_000_000)
    parser.add_argument("--eval-times", type=int, default=20)
    parser.add_argument("--total-periods", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_dir = output_path("logs", args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "parallel_status.log"
    launch_path = log_dir / "launch_manifest.json"

    if args.dry_run:
        print(json.dumps({
            "dataset": data_sensitivity_analysis_13_metadata(),
            "ro_jobs": [job["name"] for job in build_ro_jobs(args)],
            "reward_scale_rule": "2-RDC 2^-9 anchored to corrected RO mean cost",
        }, indent=2))
        return 0

    previous_launch = None
    if launch_path.exists():
        previous_launch = json.loads(launch_path.read_text(encoding="utf-8"))
        previous_pid = int(previous_launch.get("pid", -1))
        if previous_launch.get("status") == "complete":
            print(launch_path.read_text(encoding="utf-8"), end="")
            return 0
        try:
            os.kill(previous_pid, 0)
        except (OSError, ValueError):
            pass
        else:
            raise RuntimeError(
                f"Refusing duplicate launch: active pid={previous_pid} in {launch_path}"
            )

    launch = {
        "created": now(),
        "pid": os.getpid(),
        "run_tag": RUN_TAG,
        "workers": args.workers,
        "ro_workers": args.ro_workers,
        "break_step": args.break_step,
        "status": "running_ro",
    }
    if previous_launch is not None:
        launch["restart_count"] = int(previous_launch.get("restart_count", 0)) + 1
        launch["previous_pid"] = previous_launch.get("pid")
        launch["restarted"] = now()
    launch_path.write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
    append_status(status_path, f"MASTER pid={os.getpid()} stage=RO")

    ro_jobs = [job for job in build_ro_jobs(args) if not ro_completed(job["case_index"])]
    run_parallel(ro_jobs, args.ro_workers, log_dir, status_path, "RO")
    ro_rows = read_ro_rows()
    merge_ro_results(ro_rows)

    rasac_jobs, scale_manifest = build_rasac_jobs(args, ro_rows)
    scale_path = log_dir / "experiment_manifest.json"
    scale_path.write_text(json.dumps({
        "created": now(),
        "reward_scale_policy": {
            "baseline_scale": BASE_REWARD_SCALE,
            "baseline_case": "sensitivity-1",
            "selection": "nearest power of two preserving the baseline RO-scaled cost",
            "exponent_bounds": [MIN_REWARD_EXPONENT, MAX_REWARD_EXPONENT],
        },
        "cases": scale_manifest,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    launch["status"] = "running_rasac"
    launch["rasac_started"] = now()
    launch_path.write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
    append_status(status_path, "MASTER stage=RA-SAC")

    pending = [
        job for job in rasac_jobs
        if not rasac_completed(job["case"], args.break_step)
    ]
    run_parallel(pending, args.workers, log_dir, status_path, "RA-SAC")

    summary_path = output_path("summaries", f"{RUN_TAG}_rasac.md")
    subprocess.run(
        [
            PYTHON, str(SUMMARIZER),
            "--log-dir", str(log_dir),
            "--output", str(summary_path),
            "--title", "Sensitivity13 RA-SAC (2-RDC parameters, cost-scaled rewards)",
        ],
        cwd=ROOT,
        check=True,
        env=training_env(),
    )
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    launch["status"] = "complete"
    launch["completed"] = now()
    launch_path.write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
    append_status(status_path, "SUMMARY total_ro=13 total_rasac=13 failures=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
