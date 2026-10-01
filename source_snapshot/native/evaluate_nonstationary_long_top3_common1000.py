#!/usr/bin/env python3
"""Reevaluate the top-3 checkpoints from each nonstationary long run."""

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

import numpy as np
import torch

from Benchmark import (
    FiniteHorizonNonstationaryOWMREnv,
    decomposition_aggregation_rolling_distribution_aware,
    env_input_initialize,
    rolling_da_order_qty,
)
from call_agents_results import load_actor
from dataset import data_nonstationary_quantile_skus
from demand_utils import sample_quantile_series_demand
from evaluation_statistics import mean_se_median


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/nonstationary_long_top3_common1000_20260806"
RUN_PREFIX = "SAC_rasac_all_cases_long_from_scratch_20260803"
HORIZON = 13
ITEM_CODES = (
    "50102914",
    "98011894",
    "02351RFH-005",
    "98011941",
    "02352QGG-004",
)
ACTOR_PATTERN = re.compile(r"actor__(\d+)(?:_([+-]?\d+(?:\.\d+)?))?\.pt$")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def portable_path(path: Path) -> str:
    """Prefer project-relative paths while allowing external smoke outputs."""
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bank(bank: dict[int, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(bank):
        digest.update(str(key).encode("ascii"))
        digest.update(np.ascontiguousarray(bank[key]).tobytes())
    return digest.hexdigest()


def summarize(values: np.ndarray) -> dict[str, float]:
    mean, se, median = mean_se_median(values)
    return {"mean": float(mean), "se": float(se), "median": float(median)}


def paired_stats(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    return {
        "mean_delta": float(delta.mean()),
        "se_delta": float(delta.std(ddof=1) / math.sqrt(delta.size)),
        "median_delta": float(np.median(delta)),
        "candidate_better_fraction": float(np.mean(delta < 0.0)),
    }


def dataset_for(item_code: str):
    datasets, metadata = data_nonstationary_quantile_skus(item_codes=(item_code,))
    return env_input_initialize(*datasets[0])[0], metadata[0]


def build_bank(info, episodes: int, seed: int) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    bank = {k: np.empty((episodes, HORIZON), dtype=float) for k in info.rdc_dict}
    for k in info.rdc_dict:
        for period in range(HORIZON):
            bank[k][:, period] = sample_quantile_series_demand(
                info.demand_feature[k], period, size=episodes, random_state=rng
            )
    return bank


def load_bank(path: Path) -> dict[int, np.ndarray]:
    with np.load(path) as archive:
        return {
            int(key.removeprefix("rdc_")): np.asarray(archive[key], dtype=float)
            for key in archive.files
        }


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


def evaluate_actor(
    info,
    bank: dict[int, np.ndarray],
    actor_path: Path,
    allocation_rule: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    torch.set_num_threads(1)
    actor = load_actor(actor_path, torch.device("cpu"))
    env = build_env(info, bank, allocation_rule, seed)
    episodes = len(next(iter(bank.values())))
    full = np.zeros(episodes, dtype=np.float64)
    controllable = np.zeros(episodes, dtype=np.float64)
    with torch.no_grad():
        for episode in range(episodes):
            state = env.reset()
            for period in range(HORIZON):
                action = actor(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
                if isinstance(action, tuple):
                    action = action[0]
                state, _, done, _ = env.step(action.squeeze(0).cpu().numpy())
                breakdown = env.last_cost_breakdown
                full[episode] += sum(breakdown["full_cost"].values())
                controllable[episode] += sum(breakdown["train_cost"].values())
                if bool(done) != (period == HORIZON - 1):
                    raise RuntimeError(f"Unexpected done={done} at period={period}.")
    return full / HORIZON, controllable / HORIZON


def evaluate_da(info, bank: dict[int, np.ndarray], seed: int):
    schedule = decomposition_aggregation_rolling_distribution_aware(info, horizon=HORIZON)
    env = build_env(info, bank, "proportion", seed)
    env.method = "heu"
    episodes = len(next(iter(bank.values())))
    full = np.zeros(episodes, dtype=np.float64)
    controllable = np.zeros(episodes, dtype=np.float64)
    for episode in range(episodes):
        env.reset()
        for period in range(HORIZON):
            _, _, done, _ = env.step(rolling_da_order_qty(env, schedule))
            breakdown = env.last_cost_breakdown
            full[episode] += sum(breakdown["full_cost"].values())
            controllable[episode] += sum(breakdown["train_cost"].values())
            if bool(done) != (period == HORIZON - 1):
                raise RuntimeError(f"Unexpected done={done} at period={period}.")
    return full / HORIZON, controllable / HORIZON


def locate_run(method: str, item_code: str) -> Path:
    benchmark = "eq" if method == "ra_sac" else "ineq"
    parent = ROOT / "outputs/train" / f"{RUN_PREFIX}_{benchmark}_nonstationary_{item_code}"
    matches = sorted(parent.glob(f"nonstationary_*_{item_code}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one run directory for {method}/{item_code}: {matches}")
    return matches[0]


def available_actor_ranking(run_dir: Path) -> list[dict]:
    recorder = np.load(run_dir / "recorder.npy")
    score_by_step = {int(round(row[0])): float(row[1]) for row in recorder}
    candidates = []
    for actor_path in sorted(run_dir.glob("actor__*.pt")):
        match = ACTOR_PATTERN.match(actor_path.name)
        if match is None:
            continue
        step = int(match.group(1))
        score = score_by_step.get(step)
        if score is None and match.group(2) is not None:
            score = float(match.group(2))
        if score is None:
            raise ValueError(f"No recorder score for {actor_path}")
        candidates.append({
            "step": step,
            "training_avgR": score,
            "actor_path": str(actor_path.relative_to(ROOT)),
        })
    candidates.sort(key=lambda row: (-row["training_avgR"], row["step"]))
    for rank, row in enumerate(candidates, start=1):
        row["training_rank"] = rank
    return candidates


def prepare(output_dir: Path, episodes: int, base_seed: int) -> tuple[list[dict], list[dict]]:
    bank_dir = output_dir / "demand_banks"
    bank_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    bank_metadata = []
    for case_index, item_code in enumerate(ITEM_CODES):
        info, metadata = dataset_for(item_code)
        case_seed = base_seed + 1009 * case_index
        bank = build_bank(info, episodes, case_seed)
        bank_path = bank_dir / f"case{case_index}_{item_code}_common{episodes}.npz"
        np.savez_compressed(bank_path, **{f"rdc_{k}": values for k, values in bank.items()})
        bank_metadata.append({
            "case_index": case_index,
            "item_code": item_code,
            "seed": case_seed,
            "episodes": episodes,
            "sha256": sha256_bank(bank),
            "path": portable_path(bank_path),
            "initial_inventory_protocol": metadata["initial_inventory_protocol"],
        })
        jobs.append({
            "job_id": f"case{case_index}_{item_code}_rolling_da",
            "case_index": case_index,
            "item_code": item_code,
            "method": "rolling_da",
            "allocation_rule": "proportion",
            "seed": case_seed,
            "bank_path": portable_path(bank_path),
            "episodes": episodes,
        })
        for method, allocation_rule in (
            ("ra_sac", "prioritized_greedy"),
            ("ra_sac_ineq", "prioritized_greedy_inequality"),
        ):
            ranking = available_actor_ranking(locate_run(method, item_code))
            if len(ranking) < 3:
                raise RuntimeError(f"Only {len(ranking)} actors found for {method}/{item_code}.")
            for actor in ranking[:3]:
                jobs.append({
                    "job_id": f"case{case_index}_{item_code}_{method}_rank{actor['training_rank']}",
                    "case_index": case_index,
                    "item_code": item_code,
                    "method": method,
                    "allocation_rule": allocation_rule,
                    "seed": case_seed,
                    "bank_path": portable_path(bank_path),
                    "episodes": episodes,
                    **actor,
                })
    return jobs, bank_metadata


def run_job(job: dict, output_dir_text: str) -> dict:
    output_dir = Path(output_dir_text)
    part_path = output_dir / "parts" / f"{job['job_id']}.json"
    if part_path.exists():
        return json.loads(part_path.read_text(encoding="utf-8"))
    info, _ = dataset_for(job["item_code"])
    bank = load_bank(ROOT / job["bank_path"])
    if job["method"] == "rolling_da":
        full, controllable = evaluate_da(info, bank, job["seed"])
        actor_path = None
        actor_hash = None
    else:
        actor_path = ROOT / job["actor_path"]
        full, controllable = evaluate_actor(
            info, bank, actor_path, job["allocation_rule"], job["seed"]
        )
        actor_hash = sha256_file(actor_path)
    episode_path = output_dir / "episode_costs" / f"{job['job_id']}.npz"
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        episode_path,
        full_weekly_cost=full,
        controllable_weekly_cost=controllable,
    )
    result = {
        **job,
        "actor_sha256": actor_hash,
        "episode_cost_path": portable_path(episode_path),
        "full_weekly_cost": summarize(full),
        "controllable_weekly_cost": summarize(controllable),
    }
    atomic_json(part_path, result)
    return result


def merge(output_dir: Path, jobs: list[dict], bank_metadata: list[dict]) -> None:
    results = [
        json.loads((output_dir / "parts" / f"{job['job_id']}.json").read_text(encoding="utf-8"))
        for job in jobs
    ]
    results.sort(key=lambda row: (
        row["case_index"], row["method"], row.get("training_rank", 0)
    ))
    selections = []
    comparisons = []
    for case_index, item_code in enumerate(ITEM_CODES):
        da = next(row for row in results if row["case_index"] == case_index and row["method"] == "rolling_da")
        da_cost = np.load(ROOT / da["episode_cost_path"])["controllable_weekly_cost"]
        for method in ("ra_sac", "ra_sac_ineq"):
            candidates = [
                row for row in results
                if row["case_index"] == case_index and row["method"] == method
            ]
            selected = min(
                candidates,
                key=lambda row: (
                    row["controllable_weekly_cost"]["mean"], row["training_rank"]
                ),
            )
            selections.append(selected)
            selected_cost = np.load(ROOT / selected["episode_cost_path"])["controllable_weekly_cost"]
            comparisons.append({
                "case_index": case_index,
                "item_code": item_code,
                "method": method,
                "selected_training_rank": selected["training_rank"],
                "selected_step": selected["step"],
                "selected_minus_rolling_da": paired_stats(selected_cost, da_cost),
            })

    payload = {
        "protocol": {
            "episodes": jobs[0]["episodes"],
            "horizon": HORIZON,
            "common_demand_within_item": True,
            "primary_metric": "controllable_weekly_cost",
            "action_bound_profile": "joint_quantile_v2",
            "action_lower_quantile": 0.5,
            "action_upper_quantile": 0.99,
            "checkpoint_candidates": "top 3 available actors ranked by training avgR",
        },
        "demand_banks": bank_metadata,
        "results": results,
        "formal_selections": selections,
        "comparisons": comparisons,
    }
    atomic_json(output_dir / "results.json", payload)

    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        fields = [
            "case_index", "item_code", "method", "training_rank", "step", "training_avgR",
            "controllable_mean", "controllable_se", "controllable_median",
            "full_mean", "full_se", "full_median", "actor_path",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in results:
            ctl = row["controllable_weekly_cost"]
            full = row["full_weekly_cost"]
            writer.writerow({
                "case_index": row["case_index"],
                "item_code": row["item_code"],
                "method": row["method"],
                "training_rank": row.get("training_rank"),
                "step": row.get("step"),
                "training_avgR": row.get("training_avgR"),
                "controllable_mean": ctl["mean"],
                "controllable_se": ctl["se"],
                "controllable_median": ctl["median"],
                "full_mean": full["mean"],
                "full_se": full["se"],
                "full_median": full["median"],
                "actor_path": row.get("actor_path"),
            })

    lines = [
        "# Nonstationary long-run top-3 reassessment",
        "",
        f"Protocol: {jobs[0]['episodes']} common-demand episodes per item; 13 weeks per episode.",
        "Primary metric: controllable weekly cost. All actors use joint_quantile_v2 bounds.",
        "",
        "| Item | Method | Train rank | Step | Train avgR | Controllable mean | SE | Median | Full mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        ctl = row["controllable_weekly_cost"]
        full = row["full_weekly_cost"]
        lines.append(
            f"| {row['item_code']} | {row['method']} | {row.get('training_rank', '-')} | "
            f"{row.get('step', '-')} | {row.get('training_avgR', float('nan')):.4f} | "
            f"{ctl['mean']:.4f} | {ctl['se']:.4f} | {ctl['median']:.4f} | {full['mean']:.4f} |"
        )
    lines.extend(["", "## Formal selections", ""])
    for selected, comparison in zip(selections, comparisons):
        ctl = selected["controllable_weekly_cost"]
        delta = comparison["selected_minus_rolling_da"]
        lines.append(
            f"- {selected['item_code']} {selected['method']}: training rank "
            f"{selected['training_rank']} at step {selected['step']}; mean {ctl['mean']:.4f} "
            f"(SE {ctl['se']:.4f}, median {ctl['median']:.4f}); versus rolling DA "
            f"delta {delta['mean_delta']:+.4f} (paired SE {delta['se_delta']:.4f})."
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--skip-da", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker-index", type=int)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.worker_index is not None:
        launch = json.loads((output_dir / "launch_manifest.json").read_text(encoding="utf-8"))
        job = launch["jobs"][args.worker_index]
        result = run_job(job, str(output_dir))
        ctl = result["controllable_weekly_cost"]
        print(
            f"DONE {job['job_id']} mean={ctl['mean']:.6f} se={ctl['se']:.6f}",
            flush=True,
        )
        return

    jobs, bank_metadata = prepare(output_dir, args.episodes, args.seed)
    if args.skip_da:
        jobs = [job for job in jobs if job["method"] != "rolling_da"]
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    launch = {
        "workers": args.workers,
        "episodes": args.episodes,
        "jobs": jobs,
        "demand_banks": bank_metadata,
    }
    atomic_json(output_dir / "launch_manifest.json", launch)
    status_path = output_dir / "parallel_status.log"
    failures = []
    completed = 0
    status_path.write_text(f"START total={len(jobs)} workers={args.workers}\n", encoding="utf-8")
    job_log_dir = output_dir / "job_logs"
    job_log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(enumerate(jobs))
    active = {}
    while pending or active:
        while pending and len(active) < args.workers:
            index, job = pending.pop(0)
            log_path = job_log_dir / f"{job['job_id']}.log"
            log_file = log_path.open("w", encoding="utf-8")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output-dir",
                str(output_dir),
                "--worker-index",
                str(index),
            ]
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            active[process.pid] = (process, job, log_file, log_path)
        finished = []
        for pid, (process, job, log_file, log_path) in active.items():
            return_code = process.poll()
            if return_code is None:
                continue
            log_file.close()
            finished.append(pid)
            if return_code == 0:
                part_path = output_dir / "parts" / f"{job['job_id']}.json"
                result = json.loads(part_path.read_text(encoding="utf-8"))
                completed += 1
                ctl = result["controllable_weekly_cost"]
                line = (
                    f"DONE {completed}/{len(jobs)} {job['job_id']} "
                    f"mean={ctl['mean']:.6f} se={ctl['se']:.6f}\n"
                )
            else:
                error_tail = "\n".join(
                    log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
                )
                failures.append({
                    "job_id": job["job_id"],
                    "return_code": return_code,
                    "log_path": portable_path(log_path),
                    "error_tail": error_tail,
                })
                line = f"FAIL {job['job_id']} return_code={return_code} log={log_path}\n"
            with status_path.open("a", encoding="utf-8") as file:
                file.write(line)
        for pid in finished:
            del active[pid]
        if active:
            time.sleep(0.2)
    if failures:
        atomic_json(output_dir / "failures.json", failures)
        with status_path.open("a", encoding="utf-8") as file:
            file.write(f"SUMMARY total={len(jobs)} failures={len(failures)}\n")
        raise RuntimeError(f"{len(failures)} evaluation jobs failed.")
    if args.max_jobs is None and not args.skip_da:
        merge(output_dir, jobs, bank_metadata)
    with status_path.open("a", encoding="utf-8") as file:
        file.write(f"SUMMARY total={len(jobs)} failures=0\n")


if __name__ == "__main__":
    main()
