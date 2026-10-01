#!/usr/bin/env python3
"""Run the two additional-seed training campaign used for compute reporting.

The campaign contains 54 project-native jobs and 10 Kaynov source-PPO jobs.
Project-native jobs run eight at a time with one CPU thread each. Source PPO
jobs run one at a time and use eight rollout workers. Runtime and realized
environment steps are written incrementally so an interrupted campaign can be
resumed without repeating completed jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT_PYTHON = Path("/usr/local/bin/python3")
SOURCE_PYTHON = Path("/Users/zhangtuo_shan/anaconda3/envs/scmrl-py39/bin/python")
TABLE_RUNNER = ROOT / "run_table_single_combo.py"
KAYNOV_RUNNER = ROOT / "run_kaynov_single_combo.py"
SCMRL_ROOT = ROOT / "scm_rl-master"
SOURCE_RUNNER = SCMRL_ROOT / "scsim/train_ppo.py"
SOURCE_CONFIGS = SCMRL_ROOT / "configs"
SOURCE_EXPERIMENTS = SCMRL_ROOT / "experiments"
MANIFEST_PATH = ROOT / "outputs/valid_result_manifest_13x24.csv"
LOG_DIR = ROOT / "outputs/logs/manifest_two_seed_training_20260918"
JOBS_LOG_DIR = LOG_DIR / "jobs"

SEEDS = (1, 2)
PROJECT_PARALLEL = 8
SOURCE_ROLLOUT_WORKERS = 8
MIN_FREE_BYTES = 2 * 1024**3
RUN_TAG = "manifest_two_seed_20260918"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def rel(path: str | Path | None) -> str | None:
    if path is None:
        return None
    value = Path(path)
    try:
        return str(value.resolve().relative_to(ROOT))
    except ValueError:
        return str(value)


def append_status(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {message}"
    with (LOG_DIR / "parallel_status.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
    print(line, flush=True)


def compact_json(data: dict[int, float] | dict[str, float]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def add_flag(command: list[str], flag: str, value: Any = None) -> None:
    command.append(flag)
    if value is not None:
        command.append(str(value))


def table_job(
    *,
    method: str,
    benchmark: str,
    case: str,
    case_index: int,
    seed: int,
    target_steps: int,
    lr: float,
    reward_scale: float,
    gamma: float,
    net_dims: str,
    batch_size: int,
    repeat_times: float,
    horizon_len: int,
    eval_per_step: int,
    eval_times: int,
    allocation_rule: str,
    extras: list[str] | None = None,
) -> dict[str, Any]:
    slug = f"{RUN_TAG}_{method}_{case.replace('-', '_')}_s{seed}"
    output_dir = ROOT / "outputs/train" / slug / case
    command = [
        str(PROJECT_PYTHON), "-u", str(TABLE_RUNNER),
        "--benchmark", benchmark,
        "--dataset", "table15",
        "--case-index", str(case_index),
        "--lr", str(lr),
        "--reward-scale", str(reward_scale),
        "--gamma", str(gamma),
        "--net-dims", net_dims,
        "--batch-size", str(batch_size),
        "--repeat-times", str(repeat_times),
        "--horizon-len", str(horizon_len),
        "--break-step", str(target_steps),
        "--eval-per-step", str(eval_per_step),
        "--training-eval-profile", "short",
        "--eval-times", str(eval_times),
        "--eval-episode-steps", "300",
        "--eval-burn-in-steps", "150",
        "--random-seed", str(seed),
        "--num-threads", "1",
        "--device", "cpu",
        "--gpu-id", "-1",
        "--allocation-rule", allocation_rule,
        "--rho", "1.0",
        "--if-generate-demand",
        "--table15-action-space-profile", "size_aware_v1",
        "--run-tag", RUN_TAG,
        "--cwd-name", slug,
    ]
    command.extend(extras or [])
    return {
        "job_id": f"project_{method}_{case}_s{seed}",
        "family": "project",
        "method": method,
        "case": case,
        "seed": seed,
        "target_steps": target_steps,
        "horizon_len": horizon_len,
        "output_dir": str(output_dir),
        "command": command,
    }


def kaynov_job(
    *,
    method: str,
    agent: str,
    kaynov_index: int,
    case: str,
    seed: int,
    target_steps: int,
    lr: float,
    reward_scale: float,
    gamma: float,
    net_dims: str,
    batch_size: int,
    repeat_times: float,
    horizon_len: int,
    clip_grad_norm: float,
    allocation_rule: str,
    eval_per_step: int,
    eval_times: int,
    extras: list[str] | None = None,
) -> dict[str, Any]:
    slug = f"{RUN_TAG}_{method}_{case}_s{seed}"
    instance = f"data_kaynov_{kaynov_index}"
    output_dir = ROOT / "outputs/train" / slug / instance
    command = [
        str(PROJECT_PYTHON), "-u", str(KAYNOV_RUNNER),
        "--agent", agent,
        "--kaynov-index", str(kaynov_index),
        "--batch", f"{case}_s{seed}",
        "--lr", str(lr),
        "--reward-scale", str(reward_scale),
        "--gamma", str(gamma),
        "--net-dims", net_dims,
        "--batch-size", str(batch_size),
        "--repeat-times", str(repeat_times),
        "--horizon-len", str(horizon_len),
        "--clip-grad-norm", str(clip_grad_norm),
        "--break-step", str(target_steps),
        "--eval-per-step", str(eval_per_step),
        "--training-eval-profile", "short",
        "--eval-times", str(eval_times),
        "--eval-episode-steps", "300",
        "--eval-burn-in-steps", "150",
        "--random-seed", str(seed),
        "--num-threads", "1",
        "--device", "cpu",
        "--gpu-id", "-1",
        "--allocation-rule", allocation_rule,
        "--if-generate-demand",
        "--run-tag", RUN_TAG,
        "--cwd-name", slug,
    ]
    if kaynov_index == 5:
        command.extend([
            "--rho", "0.8",
            "--demand-fulfillment", "kaynov_partial_backlogging",
            "--emergency-shipment-prob", "0.8",
        ])
    else:
        command.extend(["--rho", "1.0", "--demand-fulfillment", "partial_backorder"])
    command.extend(extras or [])
    return {
        "job_id": f"project_{method}_{case}_s{seed}",
        "family": "project",
        "method": method,
        "case": case,
        "seed": seed,
        "target_steps": target_steps,
        "horizon_len": horizon_len,
        "output_dir": str(output_dir),
        "command": command,
    }


def build_project_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    buffer_sac = [
        "--buffer-size", "100000", "--soft-update-tau", "0.005",
        "--replay-warmup-steps", "10000",
        "--freeze-state-normalization-from-warmup",
        "--alpha-learning-rate", "2.5e-05",
    ]
    buffer_td3 = [
        "--buffer-size", "100000", "--soft-update-tau", "0.005",
        "--replay-warmup-steps", "10000",
        "--freeze-state-normalization-from-warmup",
    ]
    ppo_extras = [
        "--lambda-gae-adv", "0.98", "--lambda-entropy", "0.0",
        "--ratio-clip", "0.25", "--policy-activation", "tanh",
    ]
    table_cases = {
        "2-1": {"case_index": 0},
        "6-1": {"case_index": 5},
        "20-1": {"case_index": 10},
    }

    for seed in SEEDS:
        # RA-SAC: the 20-retailer row uses the validated delayed-actor mechanism.
        for case, lr, batch, target in (
            ("2-1", 5e-4, 64, 2_000_000),
            ("6-1", 2e-4, 64, 2_000_000),
        ):
            jobs.append(table_job(
                method="rasac", benchmark="ra_sac", case=case,
                case_index=table_cases[case]["case_index"], seed=seed,
                target_steps=target, lr=lr, reward_scale=2**-9, gamma=0.96,
                net_dims="64,32", batch_size=batch, repeat_times=1.0,
                horizon_len=512, eval_per_step=10_000, eval_times=100,
                allocation_rule="prioritized_greedy",
                extras=[*buffer_sac, "--clip-grad-norm", "3.0"],
            ))
        rdc20 = {1: 87, 2: 102, 3: 116, 4: 77, 5: 141, 6: 100, 7: 102,
                 8: 81, 9: 79, 10: 36, 11: 79, 12: 116, 13: 106, 14: 85,
                 15: 106, 16: 89, 17: 173, 18: 79, 19: 98, 20: 27}
        jobs.append(table_job(
            method="rasac", benchmark="ra_sac", case="20-1", case_index=10,
            seed=seed, target_steps=2_000_000, lr=1e-4,
            reward_scale=2**-9, gamma=0.94, net_dims="64,32",
            batch_size=256, repeat_times=0.25, horizon_len=512,
            eval_per_step=10_000, eval_times=20,
            allocation_rule="prioritized_greedy",
            extras=[
                *buffer_sac, "--clip-grad-norm", "3.0",
                "--sac-agent", "mechanism", "--actor-lr", "2.5e-05",
                "--critic-lr", "0.0001", "--actor-update-interval", "2",
                "--actor-net-dims", "64,32", "--critic-net-dims", "64,32",
                "--critic-ensemble-size", "2", "--critic-target-min-size", "2",
                "--replay-sampler", "paired_uniform", "--cdc-bsl-min-override", "2749",
                "--rdc-bsl-max-override", compact_json(rdc20),
                "--cdc-bsl-max-override", "2936", "--save-best-training-state",
            ],
        ))

        # RA-PPO.
        for case, gamma, batch, repeat, horizon, target in (
            ("2-1", 0.95, 4096, 8.0, 32768, 100_000_000),
            ("6-1", 0.90, 8192, 16.0, 65536, 100_000_000),
            ("20-1", 0.90, 4096, 8.0, 32768, 150_000_000),
        ):
            jobs.append(table_job(
                method="rappo", benchmark="ra_ppo", case=case,
                case_index=table_cases[case]["case_index"], seed=seed,
                target_steps=target, lr=3e-4,
                reward_scale=1.0 if case != "20-1" else 2**-6,
                gamma=gamma, net_dims="64,32", batch_size=batch,
                repeat_times=repeat, horizon_len=horizon,
                eval_per_step=500_000, eval_times=20,
                allocation_rule="prioritized_greedy",
                extras=["--clip-grad-norm", "10.0", *ppo_extras],
            ))

        # RA-TD3.
        td3_table = {
            "2-1": (1e-4, 2**-6, 2.0, 0.05, 1_000_000),
            "6-1": (1e-3, 2**-9, 2.0, 0.10, 1_000_000),
            "20-1": (3e-4, 2**-9, 1.0, 0.10, 2_000_000),
        }
        for case, (lr, rs, repeat, noise, target) in td3_table.items():
            jobs.append(table_job(
                method="ratd3", benchmark="ra_td3", case=case,
                case_index=table_cases[case]["case_index"], seed=seed,
                target_steps=target, lr=lr, reward_scale=rs, gamma=0.96,
                net_dims="64,64", batch_size=512, repeat_times=repeat,
                horizon_len=1024, eval_per_step=10_000, eval_times=100,
                allocation_rule="prioritized_greedy",
                extras=[
                    *buffer_td3, "--clip-grad-norm", "3.0",
                    "--explore-noise-std", str(noise),
                    "--policy-noise-std", "0.1", "--update-freq", "2",
                ],
            ))

        # PA-SAC and Seq-SAC (manifest label SA-SAC).
        pa_seq = (
            ("pasac", "pa_sac", "2-1", 3e-4, 2**-9, 0.94, 1.0, "proportion", 1_000_000),
            ("pasac", "pa_sac", "20-1", 2e-4, 2**-9, 0.94, 0.25, "proportion", 2_000_000),
            ("seqsac", "sa_sac", "2-1", 3e-4, 2**-6, 0.96, 0.5, "random_sequential", 1_000_000),
            ("seqsac", "sa_sac", "20-1", 1e-4, 2**-9, 0.94, 0.25, "random_sequential", 2_000_000),
        )
        for method, benchmark, case, lr, rs, gamma, repeat, allocation, target in pa_seq:
            jobs.append(table_job(
                method=method, benchmark=benchmark, case=case,
                case_index=table_cases[case]["case_index"], seed=seed,
                target_steps=target, lr=lr, reward_scale=rs, gamma=gamma,
                net_dims="64,32", batch_size=64 if case == "2-1" else 256,
                repeat_times=repeat, horizon_len=512, eval_per_step=10_000,
                eval_times=100, allocation_rule=allocation,
                extras=[*buffer_sac, "--clip-grad-norm", "3.0"],
            ))

        # AN-SAC family.
        for method, benchmark in (
            ("ansac", "an_sac"),
            ("penaltyansac", "penalty_an_sac"),
            ("lagansac", "lag_an_sac"),
        ):
            for case in ("2-1", "20-1"):
                lr = 5e-4 if case == "2-1" else 1e-4
                batch = 64 if case == "2-1" else 256
                repeat = 1.0 if case == "2-1" else 0.25
                extras = [
                    "--buffer-size", "100000", "--soft-update-tau", "0.001",
                    "--replay-warmup-steps", "10000",
                    "--freeze-state-normalization-from-warmup",
                    "--alpha-learning-rate", "2.5e-05", "--clip-grad-norm", "1.0",
                ]
                if method == "penaltyansac":
                    extras.extend([
                        "--an-penalty-coef", "1.0" if case == "2-1" else "10.0",
                        "--an-penalty-power", "1.0", "--an-violation-eps", "1.0",
                        "--an-positive-tau", "0.0",
                    ])
                elif method == "lagansac":
                    extras.extend([
                        "--lag-lambda-lr", "0.0003" if case == "2-1" else "0.003",
                    ])
                jobs.append(table_job(
                    method=method, benchmark=benchmark, case=case,
                    case_index=table_cases[case]["case_index"], seed=seed,
                    target_steps=1_000_000, lr=lr, reward_scale=2**-9,
                    gamma=0.96 if case == "2-1" else 0.94,
                    net_dims="64,32", batch_size=batch, repeat_times=repeat,
                    horizon_len=512, eval_per_step=10_000, eval_times=100,
                    allocation_rule="proportion", extras=extras,
                ))

        # Literature cases: Kaynov-1 (K0) and PB-1 (K5).
        k0_rdc_min = {1: 1, 2: 1, 3: 1}
        k0_rdc_max = {1: 12, 2: 12, 3: 12}
        k0_common = [
            "--buffer-size", "100000", "--soft-update-tau", "0.005",
            "--rdc-bsl-min-override", compact_json(k0_rdc_min),
            "--cdc-bsl-min-override", "16", "--rdc-bsl-max-override", compact_json(k0_rdc_max),
            "--cdc-bsl-max-override", "66",
        ]
        jobs.append(kaynov_job(
            method="rasac", agent="sac", kaynov_index=0, case="K0", seed=seed,
            target_steps=1_000_000, lr=0.01, reward_scale=1.0, gamma=0.96,
            net_dims="64,32", batch_size=64, repeat_times=1.0, horizon_len=512,
            clip_grad_norm=3.0, allocation_rule="prioritized_greedy",
            eval_per_step=10_000, eval_times=20, extras=k0_common,
        ))
        jobs.append(kaynov_job(
            method="rasac", agent="sac", kaynov_index=5, case="K5", seed=seed,
            target_steps=2_000_000, lr=1e-4, reward_scale=2**-6, gamma=0.90,
            net_dims="64,32", batch_size=256, repeat_times=0.5, horizon_len=512,
            clip_grad_norm=1.0, allocation_rule="prioritized_greedy",
            eval_per_step=10_000, eval_times=20,
            extras=[
                "--buffer-size", "100000", "--soft-update-tau", "0.005",
                "--replay-warmup-steps", "10000",
                "--freeze-state-normalization-from-warmup", "--alpha-lr", "2.5e-05",
                "--rdc-bsl-max-override", "50", "--cdc-bsl-max-override", "500",
            ],
        ))

        # K0/K5 RA-PPO and RA-TD3 use the stored manifest action envelopes.
        for case, index, max_rdc, max_cdc, rs, rho_tag in (
            ("K0", 0, {1: 10, 2: 10, 3: 10}, 66, 1.0, "k0"),
            ("K5", 5, 50, 500, 2**-6, "k5"),
        ):
            jobs.append(kaynov_job(
                method="rappo", agent="ppo_continuous", kaynov_index=index,
                case=case, seed=seed, target_steps=50_000_000, lr=7e-4,
                reward_scale=rs, gamma=0.90 if rho_tag == "k0" else 0.88,
                net_dims="64,64", batch_size=4096, repeat_times=8.0,
                horizon_len=32768, clip_grad_norm=10.0,
                allocation_rule="prioritized_greedy", eval_per_step=10_000,
                eval_times=100, extras=[
                    *ppo_extras, "--rdc-bsl-max-override", compact_json(max_rdc) if isinstance(max_rdc, dict) else str(max_rdc),
                    "--cdc-bsl-max-override", str(max_cdc),
                ],
            ))
            jobs.append(kaynov_job(
                method="ratd3", agent="td3", kaynov_index=index,
                case=case, seed=seed,
                target_steps=1_000_000 if rho_tag == "k0" else 2_000_000,
                lr=3e-4, reward_scale=2**-6, gamma=0.96,
                net_dims="64,64", batch_size=512,
                repeat_times=1.0 if rho_tag == "k0" else 2.0,
                horizon_len=1024, clip_grad_norm=3.0,
                allocation_rule="prioritized_greedy", eval_per_step=10_000,
                eval_times=100, extras=[
                    "--buffer-size", "100000", "--soft-update-tau", "0.005",
                    "--explore-noise-std", "0.05" if rho_tag == "k0" else "0.1",
                    "--policy-noise-std", "0.1", "--update-freq", "2",
                    "--rdc-bsl-max-override", compact_json(max_rdc) if isinstance(max_rdc, dict) else str(max_rdc),
                    "--cdc-bsl-max-override", str(max_cdc),
                ],
            ))

        # RA-SAC-Ineq is requested only for K0 and K5.
        jobs.append(kaynov_job(
            method="rasacineq", agent="sac", kaynov_index=0, case="K0", seed=seed,
            target_steps=1_000_000, lr=3e-4, reward_scale=2**-6, gamma=0.90,
            net_dims="64,32", batch_size=128, repeat_times=0.5, horizon_len=512,
            clip_grad_norm=3.0, allocation_rule="prioritized_greedy_inequality",
            eval_per_step=10_000, eval_times=20, extras=k0_common,
        ))
        jobs.append(kaynov_job(
            method="rasacineq", agent="sac", kaynov_index=5, case="K5", seed=seed,
            target_steps=10_000_000, lr=5e-5, reward_scale=2**-12, gamma=0.96,
            net_dims="64,32", batch_size=256, repeat_times=0.35, horizon_len=512,
            clip_grad_norm=1.0, allocation_rule="prioritized_greedy_inequality",
            eval_per_step=10_000, eval_times=20, extras=[
                "--alpha-lr", "1e-05", "--buffer-size", "100000",
                "--soft-update-tau", "0.005", "--replay-warmup-steps", "10000",
                "--freeze-state-normalization-from-warmup", "--rdc-bsl-max-override", "50",
                "--adaptive-action-space", "--adaptive-cdc-horizon-mode", "joint",
                "--adaptive-action-mode", "cost_aware", "--adaptive-action-quantile", "0.92",
                "--adaptive-action-cost-margin", "0.03", "--adaptive-action-min-quantile", "0.85",
                "--adaptive-action-max-quantile", "0.95", "--adaptive-cdc-multiplier", "1.0",
                "--adaptive-cdc-buffer-alpha", "0.0", "--adaptive-rdc-lower-fraction", "0.0",
                "--adaptive-cdc-lower-fraction", "0.25", "--adaptive-small-case-beta", "0.0",
                "--adaptive-small-case-threshold", "3",
            ],
        ))

    if len(jobs) != 54:
        raise AssertionError(f"Expected 54 project jobs, built {len(jobs)}")
    if len({job["job_id"] for job in jobs}) != len(jobs):
        raise AssertionError("Project job IDs are not unique")
    # Short jobs first; very long RA-PPO jobs naturally occupy the final wave.
    return sorted(jobs, key=lambda job: (job["target_steps"], job["seed"], job["job_id"]))


SOURCE_BASE = {
    "K0": "scenario_1",
    "K5": "scenario_13",
    "2-1": "srcppotable15i021t150m",
    "6-1": "srcppo61q999b32m16lr1s30h1000cont100m",
    "20-1": "srcppo201rqu999b32m16lr1s30h1000cont200m",
}


def build_source_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for seed in SEEDS:
        for case in ("K0", "K5", "2-1", "6-1", "20-1"):
            scenario = f"ms2seed20260918{case.lower().replace('-', 'r')}s{seed}"
            if case in {"K0", "K5", "2-1"}:
                target = 150_000_000
                checkpoint_freq = 20
                overrides = [
                    "--vf-clip-param", "1000.0", "--vf-loss-coeff", "1e-07",
                    "--train-batch-size", "327680", "--sgd-minibatch-size", "32768",
                    "--learning-rate", "0.0001", "--num-sgd-iter", "30",
                ]
            elif case == "6-1":
                target = 100_000_000
                checkpoint_freq = 60
                overrides = [
                    "--vf-clip-param", "10.0", "--vf-loss-coeff", "0.1",
                    "--train-batch-size", "32768", "--sgd-minibatch-size", "16384",
                    "--learning-rate", "0.0001", "--num-sgd-iter", "30",
                    "--clip-param", "0.5", "--kl-coeff", "0.0", "--kl-target", "0.01",
                    "--simple-optimizer",
                ]
            else:
                target = 200_000_000
                checkpoint_freq = 120
                overrides = [
                    "--vf-clip-param", "100.0", "--vf-loss-coeff", "1.0",
                    "--train-batch-size", "32768", "--sgd-minibatch-size", "16384",
                    "--learning-rate", "0.0001", "--num-sgd-iter", "30",
                    "--clip-param", "0.5", "--kl-coeff", "0.0", "--kl-target", "0.01",
                    "--simple-optimizer",
                ]
            command = [
                str(SOURCE_PYTHON), "-u", "scsim/train_ppo.py", "-s", scenario,
                "--timesteps-total", str(target), "--num-workers", str(SOURCE_ROLLOUT_WORKERS),
                "--seed", str(seed), "--checkpoint-freq", str(checkpoint_freq), *overrides,
            ]
            jobs.append({
                "job_id": f"sourceppo_{case}_s{seed}",
                "family": "source_ppo",
                "method": "kaynov_source_ppo",
                "case": case,
                "seed": seed,
                "target_steps": target,
                "scenario": scenario,
                "source_scenario": SOURCE_BASE[case],
                "output_dir": str(SOURCE_EXPERIMENTS / f"scenario_{scenario}"),
                "command": command,
            })
    if len(jobs) != 10:
        raise AssertionError(f"Expected 10 source-PPO jobs, built {len(jobs)}")
    return jobs


def manifest_rows() -> dict[tuple[str, str], dict[str, str]]:
    mapping = {
        "rasac": "RA-SAC", "rappo": "RA-PPO", "ratd3": "RA-TD3",
        "pasac": "PA-SAC", "seqsac": "SA-SAC", "ansac": "AN-SAC",
        "penaltyansac": "Penalty-AN-SAC", "lagansac": "Lag-AN-SAC",
        "rasacineq": "RA-SAC ineq.", "kaynov_source_ppo": "PPO (Kaynov source)",
    }
    with MANIFEST_PATH.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected: dict[tuple[str, str], dict[str, str]] = {}
    for method, benchmark in mapping.items():
        for case in ("K0", "K5", "2-1", "6-1", "20-1"):
            matches = [row for row in rows if row["benchmark"] == benchmark and row["case"] == case]
            if matches:
                row = matches[0]
                selected[(method, case)] = {
                    "benchmark": benchmark,
                    "case": case,
                    "validated_mean_cost": row.get("validated_mean_cost", ""),
                    "best_step": row.get("best_step", ""),
                    "checkpoint_or_policy": row.get("checkpoint_or_policy", ""),
                    "source": row.get("source", ""),
                }
    return selected


def prepare_source_scenarios(jobs: list[dict[str, Any]]) -> None:
    for job in jobs:
        source = SOURCE_CONFIGS / job["source_scenario"]
        target = SOURCE_CONFIGS / job["scenario"]
        if not source.is_dir():
            raise FileNotFoundError(source)
        target.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            destination = target / item.name
            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(item, destination)
        protocol = {
            "created_at": now(), "seed": job["seed"], "case": job["case"],
            "source_scenario": job["source_scenario"], "target_scenario": job["scenario"],
            "target_steps": job["target_steps"], "command": job["command"],
        }
        (target / "multiseed_protocol.json").write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def write_launch_manifest(project: list[dict[str, Any]], source: list[dict[str, Any]]) -> None:
    sources = manifest_rows()
    all_jobs = project + source
    for job in all_jobs:
        job["manifest_reference"] = sources.get((job["method"], job["case"]))
    payload = {
        "created_at": now(),
        "purpose": "Two additional training seeds for algorithmic and compute-cost reporting",
        "seeds": list(SEEDS),
        "total_jobs": len(all_jobs),
        "project_jobs": len(project),
        "source_ppo_jobs": len(source),
        "execution": {
            "project_parallel": PROJECT_PARALLEL,
            "project_threads_per_job": 1,
            "source_parallel": 1,
            "source_rollout_workers": SOURCE_ROLLOUT_WORKERS,
            "phase_order": ["project", "source_ppo"],
            "minimum_free_disk_bytes": MIN_FREE_BYTES,
            "runtime_definition": "End-to-end wall time including in-training checkpoint-selection evaluation",
        },
        "manifest_source": rel(MANIFEST_PATH),
        "jobs": all_jobs,
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "launch_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def project_actual_steps(job: dict[str, Any]) -> int:
    recorder = Path(job["output_dir"]) / "recorder.npy"
    if not recorder.is_file():
        return 0
    data = np.load(recorder)
    return int(np.max(data[:, 0])) if data.ndim == 2 and data.shape[0] else 0


def project_step_tolerance(job: dict[str, Any]) -> int:
    """Allow normal on-policy stopping granularity without accepting partial runs."""
    return max(
        10_000,
        int(job.get("horizon_len", 0)) * 2,
        (int(job["target_steps"]) + 99) // 100,
    )


def source_states(job: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(job["output_dir"]) / "ray_results/PPO"
    states: list[dict[str, Any]] = []
    if not root.is_dir():
        return states
    for progress in root.glob("PPO_OWMREnvironment_*/progress.csv"):
        try:
            with progress.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
        except OSError:
            continue
        if not rows:
            continue
        checkpoints = []
        for checkpoint in progress.parent.glob("checkpoint_*/checkpoint-*"):
            if checkpoint.name.endswith(".tune_metadata") or not checkpoint.is_file():
                continue
            if checkpoint.with_name(checkpoint.name + ".tune_metadata").is_file():
                checkpoints.append(checkpoint)
        checkpoints.sort(key=lambda path: path.stat().st_mtime)
        states.append({
            "steps": int(float(rows[-1].get("timesteps_total") or 0)),
            "checkpoint": checkpoints[-1] if checkpoints else None,
            "trial_dir": progress.parent,
        })
    return sorted(states, key=lambda item: item["steps"])


def free_bytes() -> int:
    return int(shutil.disk_usage(ROOT).free)


def wait_for_space(job_id: str) -> None:
    last_notice = 0.0
    while free_bytes() < MIN_FREE_BYTES:
        if time.monotonic() - last_notice > 300:
            append_status(
                f"WAIT_STORAGE job={job_id} free_gib={free_bytes() / 1024**3:.2f} "
                f"required_gib={MIN_FREE_BYTES / 1024**3:.2f}"
            )
            last_notice = time.monotonic()
        time.sleep(60)


RESULT_FIELDS = [
    "job_id", "family", "method", "case", "seed", "status", "returncode",
    "target_steps", "actual_steps", "elapsed_seconds", "started_at", "finished_at",
    "output_dir", "log_path", "final_checkpoint",
]


def load_results() -> dict[str, dict[str, Any]]:
    path = LOG_DIR / "results.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["job_id"]: item for item in data.get("results", [])}


def save_results(results: dict[str, dict[str, Any]], total: int) -> None:
    ordered = sorted(results.values(), key=lambda row: row["job_id"])
    complete = sum(row.get("status") == "completed" for row in ordered)
    failed = sum(row.get("status") == "failed" for row in ordered)
    payload = {
        "updated_at": now(), "total": total, "completed": complete,
        "failed": failed, "remaining": total - complete - failed, "results": ordered,
    }
    temp = LOG_DIR / "results.json.tmp"
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(LOG_DIR / "results.json")
    with (LOG_DIR / "training_runtime.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in ordered:
            writer.writerow({key: row.get(key, "") for key in RESULT_FIELDS})


def project_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "MPLCONFIGDIR": str(ROOT / "outputs/cache/matplotlib"),
    })
    return env


def run_project_job(job: dict[str, Any]) -> dict[str, Any]:
    wait_for_space(job["job_id"])
    log_path = JOBS_LOG_DIR / f"{job['job_id']}.log"
    started_at = now()
    started = time.monotonic()
    append_status(
        f"START family=project job={job['job_id']} target={job['target_steps']} "
        f"free_gib={free_bytes() / 1024**3:.2f}"
    )
    with log_path.open("a", encoding="utf-8") as output:
        output.write(f"[{started_at}] CMD {' '.join(job['command'])}\n")
        output.flush()
        completed = subprocess.run(
            job["command"], cwd=ROOT, env=project_environment(),
            stdout=output, stderr=subprocess.STDOUT,
        )
    elapsed = time.monotonic() - started
    actual = project_actual_steps(job)
    tolerance = project_step_tolerance(job)
    successful = completed.returncode == 0 and actual >= job["target_steps"] - tolerance
    result = {
        **{key: job[key] for key in ("job_id", "family", "method", "case", "seed", "target_steps")},
        "status": "completed" if successful else "failed",
        "returncode": completed.returncode, "actual_steps": actual,
        "elapsed_seconds": round(elapsed, 3), "started_at": started_at,
        "finished_at": now(), "output_dir": rel(job["output_dir"]),
        "log_path": rel(log_path), "final_checkpoint": None,
    }
    append_status(
        f"{'DONE' if successful else 'FAIL'} family=project job={job['job_id']} "
        f"actual={actual} elapsed={elapsed:.1f}s rc={completed.returncode}"
    )
    return result


def run_project_phase(
    jobs: list[dict[str, Any]], results: dict[str, dict[str, Any]], total: int
) -> None:
    pending = []
    for job in jobs:
        prior = results.get(job["job_id"])
        if prior and prior.get("status") == "completed":
            continue
        actual = project_actual_steps(job)
        tolerance = project_step_tolerance(job)
        if actual >= job["target_steps"] - tolerance:
            results[job["job_id"]] = {
                **{key: job[key] for key in ("job_id", "family", "method", "case", "seed", "target_steps")},
                "status": "completed", "returncode": 0, "actual_steps": actual,
                "elapsed_seconds": prior.get("elapsed_seconds", "") if prior else "",
                "started_at": prior.get("started_at", "") if prior else "",
                "finished_at": now(), "output_dir": rel(job["output_dir"]),
                "log_path": rel(JOBS_LOG_DIR / f"{job['job_id']}.log"), "final_checkpoint": None,
            }
        else:
            pending.append(job)
    save_results(results, total)
    append_status(
        f"PHASE_START project total={len(jobs)} pending={len(pending)} parallel={PROJECT_PARALLEL}"
    )
    with ThreadPoolExecutor(max_workers=PROJECT_PARALLEL) as executor:
        futures = {executor.submit(run_project_job, job): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as error:  # keep the rest of the campaign alive
                result = {
                    **{key: job[key] for key in ("job_id", "family", "method", "case", "seed", "target_steps")},
                    "status": "failed", "returncode": -1, "actual_steps": project_actual_steps(job),
                    "elapsed_seconds": "", "started_at": "", "finished_at": now(),
                    "output_dir": rel(job["output_dir"]),
                    "log_path": rel(JOBS_LOG_DIR / f"{job['job_id']}.log"),
                    "final_checkpoint": None, "error": repr(error),
                }
                append_status(f"FAIL family=project job={job['job_id']} exception={error!r}")
            results[job["job_id"]] = result
            save_results(results, total)
    completed = sum(results.get(job["job_id"], {}).get("status") == "completed" for job in jobs)
    failed = sum(results.get(job["job_id"], {}).get("status") == "failed" for job in jobs)
    append_status(f"PHASE_SUMMARY project total={len(jobs)} completed={completed} failed={failed}")


def source_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{SCMRL_ROOT}:{ROOT}:{env.get('PYTHONPATH', '')}"
    env["MPLCONFIGDIR"] = "/tmp/mplconfig_scmrl_multiseed"
    env.pop("RAY_ADDRESS", None)
    return env


def run_source_job(job: dict[str, Any]) -> dict[str, Any]:
    wait_for_space(job["job_id"])
    states = source_states(job)
    state = states[-1] if states else {}
    command = list(job["command"])
    checkpoint = state.get("checkpoint")
    if checkpoint is not None and int(state.get("steps") or 0) < job["target_steps"]:
        command.extend(["--restore", str(Path(checkpoint).resolve())])
    log_path = JOBS_LOG_DIR / f"{job['job_id']}.log"
    started_at = now()
    started = time.monotonic()
    append_status(
        f"START family=source_ppo job={job['job_id']} source={state.get('steps', 0)} "
        f"target={job['target_steps']} rollout_workers={SOURCE_ROLLOUT_WORKERS}"
    )
    with log_path.open("a", encoding="utf-8") as output:
        output.write(f"[{started_at}] CMD {' '.join(command)}\n")
        output.flush()
        completed = subprocess.run(
            command, cwd=SCMRL_ROOT, env=source_environment(),
            stdout=output, stderr=subprocess.STDOUT,
        )
    elapsed = time.monotonic() - started
    final_states = source_states(job)
    final = final_states[-1] if final_states else {}
    actual = int(final.get("steps") or 0)
    successful = completed.returncode == 0 and actual >= job["target_steps"]
    result = {
        **{key: job[key] for key in ("job_id", "family", "method", "case", "seed", "target_steps")},
        "status": "completed" if successful else "failed",
        "returncode": completed.returncode, "actual_steps": actual,
        "elapsed_seconds": round(elapsed, 3), "started_at": started_at,
        "finished_at": now(), "output_dir": rel(job["output_dir"]),
        "log_path": rel(log_path), "final_checkpoint": rel(final.get("checkpoint")),
    }
    append_status(
        f"{'DONE' if successful else 'FAIL'} family=source_ppo job={job['job_id']} "
        f"actual={actual} elapsed={elapsed:.1f}s rc={completed.returncode}"
    )
    return result


def run_source_phase(
    jobs: list[dict[str, Any]], results: dict[str, dict[str, Any]], total: int
) -> None:
    append_status(f"PHASE_START source_ppo total={len(jobs)} parallel=1 rollout_workers={SOURCE_ROLLOUT_WORKERS}")
    for job in jobs:
        prior = results.get(job["job_id"])
        if prior and prior.get("status") == "completed":
            continue
        states = source_states(job)
        state = states[-1] if states else {}
        if int(state.get("steps") or 0) >= job["target_steps"]:
            result = {
                **{key: job[key] for key in ("job_id", "family", "method", "case", "seed", "target_steps")},
                "status": "completed", "returncode": 0, "actual_steps": int(state["steps"]),
                "elapsed_seconds": prior.get("elapsed_seconds", "") if prior else "",
                "started_at": prior.get("started_at", "") if prior else "",
                "finished_at": now(), "output_dir": rel(job["output_dir"]),
                "log_path": rel(JOBS_LOG_DIR / f"{job['job_id']}.log"),
                "final_checkpoint": rel(state.get("checkpoint")),
            }
        else:
            result = run_source_job(job)
        results[job["job_id"]] = result
        save_results(results, total)
        if result["status"] != "completed":
            append_status(f"QUEUE_ABORT source_ppo_failure={job['job_id']}")
            break
        time.sleep(5)
    completed = sum(results.get(job["job_id"], {}).get("status") == "completed" for job in jobs)
    failed = sum(results.get(job["job_id"], {}).get("status") == "failed" for job in jobs)
    append_status(f"PHASE_SUMMARY source_ppo total={len(jobs)} completed={completed} failed={failed}")


def validate_environment() -> None:
    for path in (
        PROJECT_PYTHON, SOURCE_PYTHON, TABLE_RUNNER, KAYNOV_RUNNER,
        SOURCE_RUNNER, MANIFEST_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (os.cpu_count() or 0) < 8:
        raise RuntimeError(f"This launcher expects at least 8 logical CPUs, found {os.cpu_count()}")
    (ROOT / "outputs/cache/matplotlib").mkdir(parents=True, exist_ok=True)
    JOBS_LOG_DIR.mkdir(parents=True, exist_ok=True)


def prepare(*, write_manifest: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validate_environment()
    project = build_project_jobs()
    source = build_source_jobs()
    prepare_source_scenarios(source)
    if write_manifest:
        write_launch_manifest(project, source)
    return project, source


def run_campaign() -> int:
    project, source = prepare(write_manifest=False)
    total = len(project) + len(source)
    results = load_results()
    append_status(
        f"QUEUE_START total={total} project={len(project)} source_ppo={len(source)} "
        f"project_parallel={PROJECT_PARALLEL} source_parallel=1 free_gib={free_bytes() / 1024**3:.2f}"
    )
    run_project_phase(project, results, total)
    run_source_phase(source, results, total)
    complete = sum(row.get("status") == "completed" for row in results.values())
    failed = sum(row.get("status") == "failed" for row in results.values())
    remaining = total - complete - failed
    save_results(results, total)
    append_status(
        f"SUMMARY total={total} completed={complete} failures={failed} remaining={remaining}"
    )
    return 0 if complete == total and failed == 0 else 1


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def launch() -> int:
    validate_environment()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pid_path = LOG_DIR / "manager.pid"
    if pid_path.is_file():
        text = pid_path.read_text(encoding="utf-8").strip()
        if text and process_alive(int(text)):
            print(json.dumps({"status": "already_running", "manager_pid": int(text)}, indent=2))
            return 0
    project, source = prepare()
    command = [str(PROJECT_PYTHON), "-u", str(Path(__file__).resolve()), "--run"]
    with (LOG_DIR / "master.log").open("a", encoding="utf-8") as master:
        process = subprocess.Popen(
            command, cwd=ROOT, env=project_environment(), stdout=master,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    marker = json.loads((LOG_DIR / "launch_manifest.json").read_text(encoding="utf-8"))
    marker["manager_pid"] = process.pid
    marker["launched_at"] = now()
    (LOG_DIR / "launch_manifest.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "launched", "manager_pid": process.pid,
        "total_jobs": len(project) + len(source),
        "project_jobs": len(project), "source_ppo_jobs": len(source),
        "project_parallel": PROJECT_PARALLEL,
        "source_rollout_workers": SOURCE_ROLLOUT_WORKERS,
        "log_dir": rel(LOG_DIR),
    }, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run the prepared queue in this process")
    parser.add_argument("--prepare-only", action="store_true", help="Validate and write the launch manifest")
    args = parser.parse_args()
    if args.prepare_only:
        project, source = prepare()
        print(json.dumps({
            "status": "prepared", "total": len(project) + len(source),
            "project": len(project), "source_ppo": len(source),
            "seeds": list(SEEDS), "free_gib": round(free_bytes() / 1024**3, 3),
        }, ensure_ascii=False, indent=2))
        return 0
    return run_campaign() if args.run else launch()


if __name__ == "__main__":
    raise SystemExit(main())
