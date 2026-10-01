#!/usr/bin/env python3
"""Evaluate Rolling-DA with proportional and cost-aware ReAllocation.

The experiment reuses the five common-demand banks from the formal
nonstationary evaluation.  Rolling-DA generates the same retailer requests in
both arms; only the scarce-CDC allocation rule changes.  Allocation is called
explicitly so its runtime can be separated from request generation and the
inventory-system transition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from Benchmark import (
    FiniteHorizonNonstationaryOWMREnv,
    allocation_prioritized_greedy,
    allocation_proportion,
    decomposition_aggregation_rolling_distribution_aware,
    env_input_initialize,
    rolling_da_order_qty,
)
from dataset import data_nonstationary_quantile_skus
from evaluation_statistics import mean_se_median


ROOT = Path(__file__).resolve().parent
SOURCE_RESULTS = (
    ROOT / "outputs/evaluation/nonstationary_long_top3_common1000_20260806/results.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/evaluation/nonstationary5_rolling_da_reallocation_common1000_20260917"
)
HORIZON = 13
ITEM_CODES = (
    "50102914",
    "98011894",
    "02351RFH-005",
    "98011941",
    "02352QGG-004",
)
METHODS = (
    ("rolling_da_proportional", "proportion"),
    ("rolling_da_reallocation", "prioritized_greedy"),
)


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


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
        "relative_mean_delta": float(
            delta.mean() / max(float(np.asarray(reference, dtype=float).mean()), 1e-12)
        ),
    }


def load_source_jobs(source_results: Path, episodes: int) -> list[dict]:
    payload = json.loads(source_results.read_text(encoding="utf-8"))
    banks = {str(row["item_code"]): row for row in payload["demand_banks"]}
    jobs = []
    for case_index, item_code in enumerate(ITEM_CODES):
        if item_code not in banks:
            raise KeyError(f"Missing formal demand bank for {item_code}")
        source = banks[item_code]
        if int(source["episodes"]) < episodes:
            raise ValueError(
                f"Demand bank for {item_code} has {source['episodes']} episodes, "
                f"but {episodes} were requested."
            )
        jobs.append(
            {
                "case_index": case_index,
                "item_code": item_code,
                "seed": int(source["seed"]),
                "episodes": int(episodes),
                "bank_path": str(source["path"]),
                "bank_sha256": str(source["sha256"]),
                "initial_inventory_protocol": str(
                    source.get("initial_inventory_protocol", "cold_start_aligned_windows_v2")
                ),
            }
        )
    return jobs


def load_bank(path: Path, episodes: int) -> dict[int, np.ndarray]:
    with np.load(path) as archive:
        bank = {
            int(key.removeprefix("rdc_")): np.asarray(archive[key], dtype=float)[:episodes]
            for key in archive.files
        }
    expected = (episodes, HORIZON)
    invalid = {key: values.shape for key, values in bank.items() if values.shape != expected}
    if invalid:
        raise ValueError(f"Demand bank {path} has invalid shapes: {invalid}; expected {expected}")
    return bank


def build_info(item_code: str):
    datasets, metadata = data_nonstationary_quantile_skus(item_codes=(item_code,))
    return env_input_initialize(*datasets[0])[0], metadata[0]


def build_env(info, bank: dict[int, np.ndarray], seed: int):
    env = FiniteHorizonNonstationaryOWMREnv(
        all_demand_list=bank,
        info_input=info,
        horizon=HORIZON,
        if_generate_demand=False,
        cost_plus=0.0,
        holding_weight=1.0,
        penalty_weight=0.0,
        allocation_rule="proportion",
        train_cost_mode="controllable",
        include_period_in_state=True,
        include_action_mask_in_state=True,
        action_bound_profile="joint_quantile_v2",
        action_lower_quantile=0.5,
        action_upper_quantile=0.99,
        seed=seed,
        method="SAC",
    )
    env.method = "heu"
    env.episode = 0
    return env


def allocate_requests(
    env, action: np.ndarray, allocation_rule: str
) -> tuple[float, np.ndarray]:
    requests = np.maximum(np.asarray(action[:-1], dtype=float), 0.0)
    available = round(float(env.CDC_inventory[0]), 0)
    if allocation_rule == "proportion":
        remaining, allocated = allocation_proportion(available, requests)
    elif allocation_rule == "prioritized_greedy":
        demand_model = {
            k: env._period_lead_time_pmf_model(
                k, env.period, env.rdcs[k].lead_time + 1
            )
            for k in env.rdcs
        }
        positions = {k: float(env.state_echelon_rdc[k][-1]) for k in env.rdcs}
        remaining, allocated = allocation_prioritized_greedy(
            available,
            requests,
            demand_model,
            positions,
            env.holding_cost,
            env.backorder_cost,
        )
    else:
        raise ValueError(f"Unsupported allocation rule: {allocation_rule}")
    return float(remaining), np.hstack(
        (np.asarray(allocated, dtype=float), float(action[-1]))
    )


def evaluate_method(
    info,
    bank: dict[int, np.ndarray],
    schedule: dict,
    allocation_rule: str,
    seed: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    episodes = len(next(iter(bank.values())))
    env = build_env(info, bank, seed)
    full = np.zeros(episodes, dtype=np.float64)
    controllable = np.zeros(episodes, dtype=np.float64)
    components = {
        key: np.zeros(episodes, dtype=np.float64)
        for key in ("WHC", "RHC", "BC", "LC", "penalty")
    }
    demand = np.zeros(episodes, dtype=np.float64)
    satisfied = np.zeros(episodes, dtype=np.float64)
    request_time = 0.0
    allocation_time = 0.0
    transition_time = 0.0
    requested_units = 0.0
    allocated_units = 0.0
    binding_events = 0
    total_events = episodes * HORIZON

    simulation_start = time.perf_counter()
    for episode in range(episodes):
        env.reset()
        for period in range(HORIZON):
            request_start = time.perf_counter()
            action = rolling_da_order_qty(env, schedule)
            request_time += time.perf_counter() - request_start

            requests = np.maximum(np.asarray(action[:-1], dtype=float), 0.0)
            available = round(float(env.CDC_inventory[0]), 0)
            requested_units += float(requests.sum())
            binding_events += int(float(requests.sum()) > available + 1e-9)

            allocation_start = time.perf_counter()
            remaining, feasible_action = allocate_requests(env, action, allocation_rule)
            allocation_time += time.perf_counter() - allocation_start
            allocated_units += float(feasible_action[:-1].sum())

            transition_start = time.perf_counter()
            # Mirror the established heuristic branch exactly after the
            # explicitly timed allocation.  ``step_feasible_action`` rounds
            # retailer quantities to three decimals, while the historical
            # Rolling-DA evaluator preserves the allocator output unchanged.
            env.CDC_inventory[0] = remaining
            for k, quantity in zip(env.rdcs, feasible_action[:-1]):
                env.RDC_inventory[k][-1] = float(quantity)
            env.CDC_inventory[-1] = round(float(feasible_action[-1]), 3)
            _, _, done, _ = env._step_after_orders()
            transition_time += time.perf_counter() - transition_start

            breakdown = env.last_cost_breakdown
            full_cost = breakdown["full_cost"]
            train_cost = breakdown["train_cost"]
            full[episode] += sum(full_cost.values())
            controllable[episode] += sum(train_cost.values())
            for key in components:
                components[key][episode] += float(full_cost[key])
            for demand_info in breakdown["demand_info"].values():
                demand[episode] += float(demand_info["demand"])
                satisfied[episode] += float(demand_info["satisfied"])
            if bool(done) != (period == HORIZON - 1):
                raise RuntimeError(f"Unexpected done={done} at period={period}")
    simulation_time = time.perf_counter() - simulation_start

    arrays = {
        "full_weekly_cost": full / HORIZON,
        "controllable_weekly_cost": controllable / HORIZON,
        "demand": demand,
        "satisfied": satisfied,
    }
    for key, values in components.items():
        arrays[f"full_weekly_{key.lower()}"] = values / HORIZON

    total_demand = float(demand.sum())
    result = {
        "allocation_rule": allocation_rule,
        "episodes": episodes,
        "periods": total_events,
        "full_weekly_cost": summarize(arrays["full_weekly_cost"]),
        "controllable_weekly_cost": summarize(arrays["controllable_weekly_cost"]),
        "full_weekly_components": {
            key: summarize(arrays[f"full_weekly_{key.lower()}"])
            for key in components
        },
        "demand_weighted_fill_rate": float(
            satisfied.sum() / max(total_demand, 1e-12)
        ),
        "allocation": {
            "activation_rate": float(binding_events / total_events),
            "binding_events": int(binding_events),
            "total_events": int(total_events),
            "requested_units": float(requested_units),
            "allocated_units": float(allocated_units),
            "unfilled_units": float(requested_units - allocated_units),
        },
        "timing": {
            "simulation_wall_seconds": float(simulation_time),
            "rolling_da_request_total_seconds": float(request_time),
            "allocation_total_seconds": float(allocation_time),
            "transition_total_seconds": float(transition_time),
            "rolling_da_request_mean_ms": float(1e3 * request_time / total_events),
            "allocation_mean_ms": float(1e3 * allocation_time / total_events),
            "transition_mean_ms": float(1e3 * transition_time / total_events),
            "end_to_end_mean_ms": float(1e3 * simulation_time / total_events),
        },
    }
    return result, arrays


def run_case(job: dict, output_dir_text: str) -> dict:
    logging.getLogger("Benchmark").setLevel(logging.WARNING)
    output_dir = Path(output_dir_text)
    part_path = output_dir / "parts" / f"case{job['case_index']}_{job['item_code']}.json"
    info, metadata = build_info(job["item_code"])
    bank = load_bank(resolve_path(job["bank_path"]), job["episodes"])
    actual_hash = sha256_bank(bank)
    if job["episodes"] == 1000 and actual_hash != job["bank_sha256"]:
        raise ValueError(
            f"Demand-bank hash mismatch for {job['item_code']}: "
            f"expected {job['bank_sha256']}, got {actual_hash}"
        )

    schedule_start = time.perf_counter()
    schedule = decomposition_aggregation_rolling_distribution_aware(
        info, horizon=HORIZON
    )
    schedule_time = time.perf_counter() - schedule_start

    methods = {}
    arrays_by_method = {}
    episode_dir = output_dir / "episode_costs"
    episode_dir.mkdir(parents=True, exist_ok=True)
    for method, allocation_rule in METHODS:
        result, arrays = evaluate_method(
            info, bank, schedule, allocation_rule, job["seed"]
        )
        methods[method] = result
        arrays_by_method[method] = arrays
        np.savez_compressed(
            episode_dir / f"case{job['case_index']}_{job['item_code']}_{method}.npz",
            **arrays,
        )

    reallocation = arrays_by_method["rolling_da_reallocation"]
    proportional = arrays_by_method["rolling_da_proportional"]
    comparisons = {
        "full_weekly_cost": paired_stats(
            reallocation["full_weekly_cost"], proportional["full_weekly_cost"]
        ),
        "controllable_weekly_cost": paired_stats(
            reallocation["controllable_weekly_cost"],
            proportional["controllable_weekly_cost"],
        ),
    }
    for key in ("whc", "rhc", "bc", "lc"):
        comparisons[f"full_weekly_{key}"] = paired_stats(
            reallocation[f"full_weekly_{key}"], proportional[f"full_weekly_{key}"]
        )

    payload = {
        **job,
        "verified_bank_sha256": actual_hash,
        "dataset_metadata": metadata,
        "rolling_da_schedule_seconds": float(schedule_time),
        "methods": methods,
        "reallocation_minus_proportional": comparisons,
    }
    atomic_json(part_path, payload)
    return payload


def write_outputs(output_dir: Path, jobs: list[dict], results: list[dict]) -> None:
    results.sort(key=lambda row: int(row["case_index"]))
    result_rows = []
    comparison_rows = []
    for case in results:
        for method, result in case["methods"].items():
            components = result["full_weekly_components"]
            result_rows.append(
                {
                    "case_index": case["case_index"],
                    "item_code": case["item_code"],
                    "method": method,
                    "allocation_rule": result["allocation_rule"],
                    "full_mean": result["full_weekly_cost"]["mean"],
                    "full_se": result["full_weekly_cost"]["se"],
                    "full_median": result["full_weekly_cost"]["median"],
                    "controllable_mean": result["controllable_weekly_cost"]["mean"],
                    "controllable_se": result["controllable_weekly_cost"]["se"],
                    "controllable_median": result["controllable_weekly_cost"]["median"],
                    "cdc_holding_mean": components["WHC"]["mean"],
                    "rdc_holding_mean": components["RHC"]["mean"],
                    "backorder_mean": components["BC"]["mean"],
                    "lost_sales_mean": components["LC"]["mean"],
                    "fill_rate": result["demand_weighted_fill_rate"],
                    "activation_rate": result["allocation"]["activation_rate"],
                    "allocation_mean_ms": result["timing"]["allocation_mean_ms"],
                    "request_mean_ms": result["timing"]["rolling_da_request_mean_ms"],
                    "end_to_end_mean_ms": result["timing"]["end_to_end_mean_ms"],
                }
            )
        comparison = case["reallocation_minus_proportional"]
        comparison_rows.append(
            {
                "case_index": case["case_index"],
                "item_code": case["item_code"],
                "full_mean_delta": comparison["full_weekly_cost"]["mean_delta"],
                "full_relative_delta": comparison["full_weekly_cost"]["relative_mean_delta"],
                "full_se_delta": comparison["full_weekly_cost"]["se_delta"],
                "controllable_mean_delta": comparison["controllable_weekly_cost"]["mean_delta"],
                "controllable_relative_delta": comparison["controllable_weekly_cost"]["relative_mean_delta"],
                "candidate_better_fraction": comparison["full_weekly_cost"]["candidate_better_fraction"],
            }
        )

    payload = {
        "protocol": {
            "episodes": jobs[0]["episodes"],
            "horizon": HORIZON,
            "items": list(ITEM_CODES),
            "initial_inventory": "cold q0.9 (cost-derived service level)",
            "common_demand_across_allocation_rules": True,
            "rolling_da_request_policy": "decomposition_aggregation_rolling_distribution_aware",
            "allocation_rules": {
                "rolling_da_proportional": "deterministic largest-remainder proportional allocation",
                "rolling_da_reallocation": "exact unit-marginal equality ReAllocation",
            },
            "source_results": portable_path(SOURCE_RESULTS),
        },
        "demand_banks": jobs,
        "results": results,
    }
    atomic_json(output_dir / "results.json", payload)

    for filename, rows in (
        ("results.csv", result_rows),
        ("comparisons.csv", comparison_rows),
    ):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Rolling-DA Allocation A/B on Five Nonstationary SKUs",
        "",
        (
            f"Protocol: {jobs[0]['episodes']} common-demand episodes per SKU, "
            f"{HORIZON} weeks, cold q0.9 initialization."
        ),
        "",
        "| SKU | Allocation | Full cost | Controllable cost | Fill rate | CDC hold | RDC hold | Backorder | Active | Allocation ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result_rows:
        allocation = "ReAllocation" if row["method"].endswith("reallocation") else "Proportional"
        lines.append(
            f"| {row['item_code']} | {allocation} | {row['full_mean']:.4f} | "
            f"{row['controllable_mean']:.4f} | {row['fill_rate']:.2%} | "
            f"{row['cdc_holding_mean']:.4f} | {row['rdc_holding_mean']:.4f} | "
            f"{row['backorder_mean']:.4f} | {row['activation_rate']:.2%} | "
            f"{row['allocation_mean_ms']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## ReAllocation minus proportional allocation",
            "",
            "| SKU | Full-cost delta | Relative delta | Controllable delta | Relative delta | ReAllocation better |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison_rows:
        lines.append(
            f"| {row['item_code']} | {row['full_mean_delta']:+.4f} | "
            f"{row['full_relative_delta']:+.2%} | "
            f"{row['controllable_mean_delta']:+.4f} | "
            f"{row['controllable_relative_delta']:+.2%} | "
            f"{row['candidate_better_fraction']:.2%} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--source-results", type=Path, default=SOURCE_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")

    source_results = resolve_path(args.source_results)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = load_source_jobs(source_results, args.episodes)
    atomic_json(
        output_dir / "launch_manifest.json",
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "running",
            "episodes": args.episodes,
            "max_workers": min(args.max_workers, len(jobs)),
            "source_results": portable_path(source_results),
            "output_dir": portable_path(output_dir),
            "jobs": jobs,
        },
    )

    started = time.perf_counter()
    results = []
    failures = []
    max_workers = min(max(int(args.max_workers), 1), len(jobs))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_case, job, str(output_dir)): job for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"DONE {job['item_code']} "
                    f"{len(results)}/{len(jobs)} elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            except Exception as exc:  # preserve every completed part before failing
                failures.append({"item_code": job["item_code"], "error": repr(exc)})
                print(f"FAIL {job['item_code']}: {exc!r}", flush=True)

    manifest = json.loads((output_dir / "launch_manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_seconds": float(time.perf_counter() - started),
            "completed": len(results),
            "failures": failures,
            "status": "complete" if not failures else "failed",
        }
    )
    atomic_json(output_dir / "launch_manifest.json", manifest)
    if failures:
        raise RuntimeError(f"{len(failures)} case(s) failed: {failures}")
    write_outputs(output_dir, jobs, results)
    print(f"WROTE {output_dir}", flush=True)


if __name__ == "__main__":
    main()
