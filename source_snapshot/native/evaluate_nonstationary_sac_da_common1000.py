#!/usr/bin/env python3
"""Paired common-demand evaluation for nonstationary SAC and rolling DA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/nonstationary_sac_da_common1000_20260801"
HORIZON = 13
ACTORS = {
    (0, "ra_sac"): ROOT / (
        "outputs/train/SAC_nonstationary_sac_two_cases_20260801_ra_sac_50102914_500k/"
        "nonstationary_0_50102914/actor__000000260520_-0031.774.pt"
    ),
    (0, "ra_sac_ineq"): ROOT / (
        "outputs/train/SAC_nonstationary_sac_two_cases_20260801_ra_sac_ineq_50102914_500k/"
        "nonstationary_0_50102914/actor__000000260520_-0031.774.pt"
    ),
    (1, "ra_sac"): ROOT / (
        "outputs/train/SAC_nonstationary_sac_two_cases_20260801_ra_sac_98011894_500k/"
        "nonstationary_1_98011894/actor__000000229320_-0935.135.pt"
    ),
    (1, "ra_sac_ineq"): ROOT / (
        "outputs/train/SAC_nonstationary_sac_two_cases_20260801_ra_sac_ineq_98011894_500k/"
        "nonstationary_1_98011894/actor__000000229320_-0935.135.pt"
    ),
}
ALLOCATION_RULES = {
    "ra_sac": "prioritized_greedy",
    "ra_sac_ineq": "prioritized_greedy_inequality",
    "rolling_da": "proportion",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bank_sha256(bank: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(bank):
        digest.update(str(key).encode("utf-8"))
        digest.update(np.ascontiguousarray(bank[key]).tobytes())
    return digest.hexdigest()


def build_fixed_demands(info, episodes: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    bank = {k: np.empty((episodes, HORIZON), dtype=float) for k in info.rdc_dict}
    for k in info.rdc_dict:
        for period in range(HORIZON):
            bank[k][:, period] = sample_quantile_series_demand(
                info.demand_feature[k], period, size=episodes, random_state=rng
            )
    return bank


def build_env(info, bank: dict, allocation_rule: str, seed: int):
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
        seed=seed,
        method="SAC",
        # Actors referenced by this historical evaluator were trained with the
        # original additive action-space mapping.
        action_bound_profile="legacy_v1",
    )
    env.episode = 0
    return env


def evaluate_actor(info, bank, actor_path: Path, allocation_rule: str, seed: int):
    actor = load_actor(actor_path, torch.device("cpu"))
    env = build_env(info, bank, allocation_rule, seed)
    env.method = "SAC"
    episodes = len(next(iter(bank.values())))
    full_weekly = np.zeros(episodes, dtype=np.float64)
    controllable_weekly = np.zeros(episodes, dtype=np.float64)

    with torch.no_grad():
        for episode in range(episodes):
            state = env.reset()
            for period in range(HORIZON):
                state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                action = actor(state_tensor)
                if isinstance(action, tuple):
                    action = action[0]
                action = action.squeeze(0).detach().cpu().numpy()
                state, _, done, _ = env.step(action)
                breakdown = env.last_cost_breakdown
                full_weekly[episode] += sum(breakdown["full_cost"].values())
                controllable_weekly[episode] += sum(breakdown["train_cost"].values())
                if done != (period == HORIZON - 1):
                    raise RuntimeError(f"Unexpected done={done} at period {period}.")
    return full_weekly / HORIZON, controllable_weekly / HORIZON


def evaluate_da(info, bank, seed: int):
    schedule = decomposition_aggregation_rolling_distribution_aware(info, horizon=HORIZON)
    env = build_env(info, bank, "proportion", seed)
    env.method = "heu"
    episodes = len(next(iter(bank.values())))
    full_weekly = np.zeros(episodes, dtype=np.float64)
    controllable_weekly = np.zeros(episodes, dtype=np.float64)
    for episode in range(episodes):
        env.reset()
        for period in range(HORIZON):
            _, _, done, _ = env.step(rolling_da_order_qty(env, schedule))
            breakdown = env.last_cost_breakdown
            full_weekly[episode] += sum(breakdown["full_cost"].values())
            controllable_weekly[episode] += sum(breakdown["train_cost"].values())
            if done != (period == HORIZON - 1):
                raise RuntimeError(f"Unexpected done={done} at period {period}.")
    return full_weekly / HORIZON, controllable_weekly / HORIZON


def paired_stats(candidate: np.ndarray, reference: np.ndarray) -> dict:
    delta = np.asarray(candidate) - np.asarray(reference)
    return {
        "mean_delta": float(delta.mean()),
        "se_delta": float(delta.std(ddof=1) / math.sqrt(delta.size)),
        "median_delta": float(np.median(delta)),
        "candidate_better_fraction": float(np.mean(delta < 0.0)),
    }


def summarize(values: np.ndarray) -> dict:
    mean, se, median = mean_se_median(values)
    return {"mean": mean, "se": se, "median": median}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    episode_dir = output_dir / "episode_costs"
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_dir.mkdir(parents=True, exist_ok=True)

    missing = [str(path) for path in ACTORS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing best actors:\n" + "\n".join(missing))

    datasets, metadata = data_nonstationary_quantile_skus(candidate_count=2)
    results = []
    comparisons = []
    bank_metadata = []
    for case_index, (data, case_metadata) in enumerate(zip(datasets, metadata)):
        info = env_input_initialize(*data)[0]
        case_seed = args.seed + case_index * 1009
        bank = build_fixed_demands(info, args.episodes, case_seed)
        item_code = str(case_metadata["item_code"])
        np.savez_compressed(
            output_dir / f"case{case_index}_{item_code}_demand_bank.npz",
            **{f"rdc_{k}": values for k, values in bank.items()},
        )
        bank_metadata.append({
            "case_index": case_index,
            "item_code": item_code,
            "seed": case_seed,
            "episodes": args.episodes,
            "sha256": bank_sha256(bank),
        })

        costs = {}
        for method in ("ra_sac", "ra_sac_ineq", "rolling_da"):
            if method == "rolling_da":
                full, controllable = evaluate_da(info, bank, case_seed)
                actor_path = None
                actor_hash = None
            else:
                actor_path = ACTORS[(case_index, method)]
                full, controllable = evaluate_actor(
                    info, bank, actor_path, ALLOCATION_RULES[method], case_seed
                )
                actor_hash = file_sha256(actor_path)
            costs[method] = {"full": full, "controllable": controllable}
            np.savez_compressed(
                episode_dir / f"case{case_index}_{item_code}_{method}.npz",
                full_weekly_cost=full,
                controllable_weekly_cost=controllable,
            )
            row = {
                "case_index": case_index,
                "item_code": item_code,
                "method": method,
                "allocation_rule": ALLOCATION_RULES[method],
                "episodes": args.episodes,
                "actor_path": None if actor_path is None else str(actor_path.relative_to(ROOT)),
                "actor_sha256": actor_hash,
                "full_weekly_cost": summarize(full),
                "controllable_weekly_cost": summarize(controllable),
            }
            results.append(row)
            print(
                f"case={case_index} item={item_code} method={method} "
                f"controllable={row['controllable_weekly_cost']['mean']:.6f} "
                f"full={row['full_weekly_cost']['mean']:.6f}",
                flush=True,
            )

        for candidate, reference in (
            ("ra_sac", "rolling_da"),
            ("ra_sac_ineq", "rolling_da"),
            ("ra_sac_ineq", "ra_sac"),
        ):
            comparisons.append({
                "case_index": case_index,
                "item_code": item_code,
                "candidate": candidate,
                "reference": reference,
                "full_weekly_cost": paired_stats(
                    costs[candidate]["full"], costs[reference]["full"]
                ),
                "controllable_weekly_cost": paired_stats(
                    costs[candidate]["controllable"], costs[reference]["controllable"]
                ),
            })

    payload = {
        "protocol": {
            "episodes": args.episodes,
            "horizon": HORIZON,
            "base_seed": args.seed,
            "common_demand_within_case": True,
            "primary_metric": "controllable_weekly_cost",
        },
        "demand_banks": bank_metadata,
        "results": results,
        "comparisons": comparisons,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "case_index", "item_code", "method", "allocation_rule", "episodes",
            "controllable_mean", "controllable_se", "controllable_median",
            "full_mean", "full_se", "full_median", "actor_path", "actor_sha256",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({
                "case_index": row["case_index"],
                "item_code": row["item_code"],
                "method": row["method"],
                "allocation_rule": row["allocation_rule"],
                "episodes": row["episodes"],
                "controllable_mean": row["controllable_weekly_cost"]["mean"],
                "controllable_se": row["controllable_weekly_cost"]["se"],
                "controllable_median": row["controllable_weekly_cost"]["median"],
                "full_mean": row["full_weekly_cost"]["mean"],
                "full_se": row["full_weekly_cost"]["se"],
                "full_median": row["full_weekly_cost"]["median"],
                "actor_path": row["actor_path"],
                "actor_sha256": row["actor_sha256"],
            })

    lines = [
        "# Nonstationary SAC and rolling-DA paired evaluation",
        "",
        f"Protocol: {args.episodes} common-demand episodes per case, {HORIZON} weeks per episode.",
        "",
        "| Case | Item | Method | Controllable mean | SE | Median | Full mean | SE | Median |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        ctl = row["controllable_weekly_cost"]
        full = row["full_weekly_cost"]
        lines.append(
            f"| {row['case_index']} | {row['item_code']} | {row['method']} | "
            f"{ctl['mean']:.4f} | {ctl['se']:.4f} | {ctl['median']:.4f} | "
            f"{full['mean']:.4f} | {full['se']:.4f} | {full['median']:.4f} |"
        )
    lines.extend(["", "## Paired controllable-cost deltas", ""])
    for row in comparisons:
        stats = row["controllable_weekly_cost"]
        lines.append(
            f"- Case {row['case_index']} {row['candidate']} - {row['reference']}: "
            f"mean delta {stats['mean_delta']:.4f} (SE {stats['se_delta']:.4f}), "
            f"better on {stats['candidate_better_fraction']:.1%} of episodes."
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
