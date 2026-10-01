#!/usr/bin/env python3
"""Tune a low-dimensional Rolling-DA base-stock policy with ReAllocation.

The search never reads the formal common-1000 test demand banks.  It first
uses independent fixed tuning and validation banks, writes an immutable
selection manifest for all five SKUs, and only then evaluates the locked
parameters on the existing formal test banks.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from Benchmark import decomposition_aggregation_rolling_distribution_aware
from demand_utils import sample_quantile_series_demand
from evaluate_nonstationary_rolling_da_reallocation import (
    HORIZON,
    ITEM_CODES,
    ROOT,
    SOURCE_RESULTS,
    atomic_json,
    build_info,
    evaluate_method,
    load_bank,
    paired_stats,
    portable_path,
    resolve_path,
    sha256_bank,
    summarize,
)


DEFAULT_OUTPUT = (
    ROOT
    / "outputs/evaluation/nonstationary5_tuned_base_stock_reallocation_20260917"
)
BASELINE_OUTPUT = (
    ROOT
    / "outputs/evaluation/nonstationary5_rolling_da_reallocation_common1000_20260917"
)
DEFAULT_COARSE_GRID = (0.60, 0.80, 1.00, 1.25, 1.50)
DEFAULT_STEPS = (0.20, 0.10, 0.05)


def build_bank(info, episodes: int, seed: int) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    bank = {
        int(k): np.empty((episodes, HORIZON), dtype=float)
        for k in info.rdc_dict
    }
    for k in info.rdc_dict:
        for period in range(HORIZON):
            bank[int(k)][:, period] = sample_quantile_series_demand(
                info.demand_feature[k],
                period,
                size=episodes,
                random_state=rng,
            )
    return bank


def save_bank(path: Path, bank: dict[int, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"rdc_{k}": values for k, values in bank.items()})


def parameter_names(rdc_keys: tuple[int, ...]) -> tuple[str, ...]:
    return ("cdc_local_scale",) + tuple(f"rdc_{k}_scale" for k in rdc_keys)


def normalize_vector(values, lower: float, upper: float) -> tuple[float, ...]:
    return tuple(round(float(np.clip(value, lower, upper)), 6) for value in values)


def adjusted_schedule(
    base_schedule: dict,
    info,
    scales: tuple[float, ...],
) -> dict:
    """Scale node targets while preserving the DA echelon decomposition."""
    rdc_keys = tuple(base_schedule["rdc_base_stock"].keys())
    if len(scales) != len(rdc_keys) + 1:
        raise ValueError(f"Expected {len(rdc_keys) + 1} scales, got {len(scales)}")

    rdc_levels = {
        k: np.asarray(base_schedule["rdc_base_stock"][k], dtype=float).copy()
        * float(scales[index + 1])
        for index, k in enumerate(rdc_keys)
    }
    cdc_local = (
        np.asarray(base_schedule["cdc_local_base_stock"], dtype=float).copy()
        * float(scales[0])
    )
    cdc_echelon = np.full_like(cdc_local, np.nan, dtype=float)
    l0 = int(info.item_lead_time)
    for period in range(int(base_schedule["horizon"])):
        if not bool(base_schedule["cdc_active_mask"][period]):
            continue
        arrival_period = period + l0
        downstream = sum(
            float(rdc_levels[k][arrival_period])
            for k in base_schedule["reachable_rdcs"][period]
        )
        cdc_echelon[period] = float(cdc_local[period]) + downstream

    schedule = dict(base_schedule)
    schedule["rdc_base_stock"] = rdc_levels
    schedule["cdc_local_base_stock"] = cdc_local
    schedule["cdc_echelon_base_stock"] = cdc_echelon
    schedule["tuning_scales"] = {
        name: float(value)
        for name, value in zip(parameter_names(rdc_keys), scales)
    }
    return schedule


def assert_identity_schedule(base_schedule: dict, adjusted: dict) -> None:
    for k in base_schedule["rdc_base_stock"]:
        if not np.array_equal(
            base_schedule["rdc_base_stock"][k],
            adjusted["rdc_base_stock"][k],
            equal_nan=True,
        ):
            raise AssertionError(f"Unit scaling changed RDC {k} targets")
    for key in ("cdc_local_base_stock", "cdc_echelon_base_stock"):
        if not np.array_equal(base_schedule[key], adjusted[key], equal_nan=True):
            raise AssertionError(f"Unit scaling changed {key}")


def candidate_record(
    info,
    bank: dict[int, np.ndarray],
    base_schedule: dict,
    scales: tuple[float, ...],
    seed: int,
    phase: str,
) -> dict:
    schedule = adjusted_schedule(base_schedule, info, scales)
    result, _ = evaluate_method(
        info,
        bank,
        schedule,
        allocation_rule="prioritized_greedy",
        seed=seed,
    )
    return {
        "phase": phase,
        "scales": list(scales),
        "controllable_mean": float(result["controllable_weekly_cost"]["mean"]),
        "controllable_se": float(result["controllable_weekly_cost"]["se"]),
        "full_mean": float(result["full_weekly_cost"]["mean"]),
        "fill_rate": float(result["demand_weighted_fill_rate"]),
        "simulation_wall_seconds": float(result["timing"]["simulation_wall_seconds"]),
    }


def tune_case(job: dict, output_dir_text: str, search: dict) -> dict:
    logging.getLogger("Benchmark").setLevel(logging.WARNING)
    started = time.perf_counter()
    output_dir = Path(output_dir_text)
    info, metadata = build_info(job["item_code"])
    rdc_keys = tuple(info.rdc_dict.keys())
    names = parameter_names(rdc_keys)
    dimension = len(names)
    base_schedule = decomposition_aggregation_rolling_distribution_aware(
        info, horizon=HORIZON
    )
    identity = adjusted_schedule(base_schedule, info, (1.0,) * dimension)
    assert_identity_schedule(base_schedule, identity)

    tuning_bank = build_bank(info, int(search["tuning_episodes"]), int(job["tuning_seed"]))
    validation_bank = build_bank(
        info, int(search["validation_episodes"]), int(job["validation_seed"])
    )
    tuning_bank_path = (
        output_dir / "demand_banks" / f"case{job['case_index']}_{job['item_code']}_tuning.npz"
    )
    validation_bank_path = (
        output_dir
        / "demand_banks"
        / f"case{job['case_index']}_{job['item_code']}_validation.npz"
    )
    save_bank(tuning_bank_path, tuning_bank)
    save_bank(validation_bank_path, validation_bank)

    lower = float(search["lower_bound"])
    upper = float(search["upper_bound"])
    max_candidates = int(search["max_candidates"])
    cache: dict[tuple[float, ...], dict] = {}
    history: list[dict] = []

    def evaluate_train(values, phase):
        key = normalize_vector(values, lower, upper)
        if key in cache:
            return cache[key]
        if len(cache) >= max_candidates:
            return None
        record = candidate_record(
            info, tuning_bank, base_schedule, key, int(job["tuning_seed"]), phase
        )
        record["candidate_index"] = len(history)
        cache[key] = record
        history.append(record)
        return record

    baseline = evaluate_train((1.0,) * dimension, "baseline")
    for cdc_scale, common_rdc_scale in itertools.product(
        search["coarse_grid"], repeat=2
    ):
        if len(cache) >= max_candidates:
            break
        evaluate_train(
            (float(cdc_scale),) + (float(common_rdc_scale),) * (dimension - 1),
            "coarse_shared_rdc",
        )

    current = min(history, key=lambda row: row["controllable_mean"])
    moves_by_step = {}
    for step in search["step_sizes"]:
        moves = 0
        while moves < int(search["max_moves_per_step"]) and len(cache) < max_candidates:
            neighbors = []
            current_vector = tuple(current["scales"])
            for index in range(dimension):
                for direction in (-1.0, 1.0):
                    trial = list(current_vector)
                    trial[index] += direction * float(step)
                    record = evaluate_train(trial, f"coordinate_step_{step:g}")
                    if record is not None:
                        neighbors.append(record)
            if not neighbors:
                break
            best_neighbor = min(neighbors, key=lambda row: row["controllable_mean"])
            if best_neighbor["controllable_mean"] < current["controllable_mean"] - 1e-12:
                current = best_neighbor
                moves += 1
            else:
                break
        moves_by_step[str(step)] = moves

    ranked_train = sorted(history, key=lambda row: row["controllable_mean"])
    validation_vectors = [tuple(row["scales"]) for row in ranked_train[: int(search["validation_top_k"])]]
    validation_vectors.extend([(1.0,) * dimension, tuple(current["scales"])])
    unique_validation = []
    seen = set()
    for values in validation_vectors:
        key = normalize_vector(values, lower, upper)
        if key not in seen:
            seen.add(key)
            unique_validation.append(key)

    validation_records = []
    for values in unique_validation:
        record = candidate_record(
            info,
            validation_bank,
            base_schedule,
            values,
            int(job["validation_seed"]),
            "validation",
        )
        train_record = cache[values]
        record["training_controllable_mean"] = train_record["controllable_mean"]
        record["distance_from_da"] = float(sum(abs(value - 1.0) for value in values))
        validation_records.append(record)
    selected = min(
        validation_records,
        key=lambda row: (
            row["controllable_mean"],
            row["distance_from_da"],
            tuple(row["scales"]),
        ),
    )

    history_path = (
        output_dir / "search_history" / f"case{job['case_index']}_{job['item_code']}.csv"
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "candidate_index",
            "phase",
            *names,
            "controllable_mean",
            "controllable_se",
            "full_mean",
            "fill_rate",
            "simulation_wall_seconds",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "candidate_index": row["candidate_index"],
                    "phase": row["phase"],
                    **dict(zip(names, row["scales"])),
                    "controllable_mean": row["controllable_mean"],
                    "controllable_se": row["controllable_se"],
                    "full_mean": row["full_mean"],
                    "fill_rate": row["fill_rate"],
                    "simulation_wall_seconds": row["simulation_wall_seconds"],
                }
            )

    payload = {
        **job,
        "dataset_metadata": metadata,
        "parameter_names": list(names),
        "baseline_training": baseline,
        "training_incumbent": current,
        "selected_validation": selected,
        "validation_candidates": validation_records,
        "search": {
            **search,
            "candidate_count": len(history),
            "validation_candidate_count": len(validation_records),
            "moves_by_step": moves_by_step,
            "candidate_simulation_seconds": float(
                sum(row["simulation_wall_seconds"] for row in history)
            ),
            "validation_simulation_seconds": float(
                sum(row["simulation_wall_seconds"] for row in validation_records)
            ),
            "case_wall_seconds": float(time.perf_counter() - started),
        },
        "banks": {
            "tuning": {
                "path": portable_path(tuning_bank_path),
                "episodes": int(search["tuning_episodes"]),
                "seed": int(job["tuning_seed"]),
                "sha256": sha256_bank(tuning_bank),
            },
            "validation": {
                "path": portable_path(validation_bank_path),
                "episodes": int(search["validation_episodes"]),
                "seed": int(job["validation_seed"]),
                "sha256": sha256_bank(validation_bank),
            },
        },
        "search_history_path": portable_path(history_path),
    }
    part_path = output_dir / "tuning_parts" / f"case{job['case_index']}_{job['item_code']}.json"
    atomic_json(part_path, payload)
    return payload


def load_formal_metadata(source_results: Path) -> tuple[dict, dict]:
    payload = json.loads(source_results.read_text(encoding="utf-8"))
    banks = {str(row["item_code"]): row for row in payload["demand_banks"]}
    ra_sac = {
        str(row["item_code"]): row
        for row in payload["formal_selections"]
        if row["method"] == "ra_sac"
    }
    return banks, ra_sac


def build_jobs(source_results: Path, args) -> list[dict]:
    banks, ra_sac = load_formal_metadata(source_results)
    jobs = []
    for case_index, item_code in enumerate(ITEM_CODES):
        bank = banks[item_code]
        jobs.append(
            {
                "case_index": case_index,
                "item_code": item_code,
                "tuning_seed": int(args.tuning_seed_base + 1009 * case_index),
                "validation_seed": int(args.validation_seed_base + 1009 * case_index),
                "test_seed": int(bank["seed"]),
                "test_bank_path": str(bank["path"]),
                "test_bank_sha256": str(bank["sha256"]),
                "ra_sac_episode_cost_path": str(ra_sac[item_code]["episode_cost_path"]),
                "ra_sac_step": int(ra_sac[item_code]["step"]),
            }
        )
    return jobs


def test_case(selection: dict, output_dir_text: str, test_episodes: int) -> dict:
    logging.getLogger("Benchmark").setLevel(logging.WARNING)
    output_dir = Path(output_dir_text)
    info, _ = build_info(selection["item_code"])
    base_schedule = decomposition_aggregation_rolling_distribution_aware(
        info, horizon=HORIZON
    )
    scales = tuple(selection["selected_validation"]["scales"])
    schedule = adjusted_schedule(base_schedule, info, scales)
    test_bank = load_bank(resolve_path(selection["test_bank_path"]), test_episodes)
    if test_episodes == 1000:
        actual_hash = sha256_bank(test_bank)
        if actual_hash != selection["test_bank_sha256"]:
            raise ValueError(
                f"Formal test bank mismatch for {selection['item_code']}: "
                f"{actual_hash} != {selection['test_bank_sha256']}"
            )

    tuned, tuned_arrays = evaluate_method(
        info,
        test_bank,
        schedule,
        allocation_rule="prioritized_greedy",
        seed=int(selection["test_seed"]),
    )
    case_index = int(selection["case_index"])
    item_code = str(selection["item_code"])
    tuned_path = (
        output_dir
        / "episode_costs"
        / f"case{case_index}_{item_code}_tuned_base_stock_reallocation.npz"
    )
    tuned_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tuned_path, **tuned_arrays)

    baseline_path = (
        BASELINE_OUTPUT
        / "episode_costs"
        / f"case{case_index}_{item_code}_rolling_da_reallocation.npz"
    )
    with np.load(baseline_path) as archive:
        baseline_arrays = {
            key: np.asarray(archive[key], dtype=float)[:test_episodes]
            for key in archive.files
        }
    with np.load(resolve_path(selection["ra_sac_episode_cost_path"])) as archive:
        ra_arrays = {
            key: np.asarray(archive[key], dtype=float)[:test_episodes]
            for key in archive.files
        }

    comparisons = {}
    for metric in ("controllable_weekly_cost", "full_weekly_cost"):
        comparisons[f"tuned_minus_baseline_{metric}"] = paired_stats(
            tuned_arrays[metric], baseline_arrays[metric]
        )
        comparisons[f"tuned_minus_ra_sac_{metric}"] = paired_stats(
            tuned_arrays[metric], ra_arrays[metric]
        )

    baseline = {
        "controllable_weekly_cost": summarize(
            baseline_arrays["controllable_weekly_cost"]
        ),
        "full_weekly_cost": summarize(baseline_arrays["full_weekly_cost"]),
        "demand_weighted_fill_rate": float(
            baseline_arrays["satisfied"].sum()
            / max(float(baseline_arrays["demand"].sum()), 1e-12)
        ),
    }
    ra_sac = {
        "step": int(selection["ra_sac_step"]),
        "controllable_weekly_cost": summarize(
            ra_arrays["controllable_weekly_cost"]
        ),
        "full_weekly_cost": summarize(ra_arrays["full_weekly_cost"]),
    }
    payload = {
        "case_index": case_index,
        "item_code": item_code,
        "episodes": int(test_episodes),
        "locked_scales": list(scales),
        "parameter_names": selection["parameter_names"],
        "tuned_base_stock_reallocation": tuned,
        "rolling_da_reallocation": baseline,
        "ra_sac": ra_sac,
        "comparisons": comparisons,
        "tuned_episode_cost_path": portable_path(tuned_path),
    }
    atomic_json(
        output_dir / "test_parts" / f"case{case_index}_{item_code}.json", payload
    )
    return payload


def run_parallel(function, inputs: list, max_workers: int, label: str) -> list:
    results = []
    with ProcessPoolExecutor(max_workers=min(max_workers, len(inputs))) as executor:
        futures = {executor.submit(function, *item): item for item in inputs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{label} {result['item_code']} {len(results)}/{len(inputs)}",
                flush=True,
            )
    return sorted(results, key=lambda row: int(row["case_index"]))


def write_final_outputs(output_dir: Path, selections: list[dict], results: list[dict], protocol: dict) -> None:
    rows = []
    for selection, result in zip(selections, results):
        tuned = result["tuned_base_stock_reallocation"]
        baseline = result["rolling_da_reallocation"]
        ra_sac = result["ra_sac"]
        delta = result["comparisons"]
        rows.append(
            {
                "item_code": result["item_code"],
                **dict(zip(result["parameter_names"], result["locked_scales"])),
                "ra_sac_controllable": ra_sac["controllable_weekly_cost"]["mean"],
                "baseline_controllable": baseline["controllable_weekly_cost"]["mean"],
                "tuned_controllable": tuned["controllable_weekly_cost"]["mean"],
                "tuned_vs_baseline_controllable": delta[
                    "tuned_minus_baseline_controllable_weekly_cost"
                ]["relative_mean_delta"],
                "tuned_vs_ra_sac_controllable": delta[
                    "tuned_minus_ra_sac_controllable_weekly_cost"
                ]["relative_mean_delta"],
                "ra_sac_full": ra_sac["full_weekly_cost"]["mean"],
                "baseline_full": baseline["full_weekly_cost"]["mean"],
                "tuned_full": tuned["full_weekly_cost"]["mean"],
                "tuned_vs_baseline_full": delta[
                    "tuned_minus_baseline_full_weekly_cost"
                ]["relative_mean_delta"],
                "tuned_vs_ra_sac_full": delta[
                    "tuned_minus_ra_sac_full_weekly_cost"
                ]["relative_mean_delta"],
                "baseline_fill_rate": baseline["demand_weighted_fill_rate"],
                "tuned_fill_rate": tuned["demand_weighted_fill_rate"],
                "search_candidates": selection["search"]["candidate_count"],
                "validation_candidates": selection["search"]["validation_candidate_count"],
                "search_case_wall_seconds": selection["search"]["case_wall_seconds"],
                "final_test_wall_seconds": tuned["timing"]["simulation_wall_seconds"],
            }
        )
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "protocol": protocol,
        "locked_selections": selections,
        "test_results": results,
    }
    atomic_json(output_dir / "results.json", payload)

    total_ra_ctl = sum(row["ra_sac_controllable"] for row in rows)
    total_base_ctl = sum(row["baseline_controllable"] for row in rows)
    total_tuned_ctl = sum(row["tuned_controllable"] for row in rows)
    total_ra_full = sum(row["ra_sac_full"] for row in rows)
    total_base_full = sum(row["baseline_full"] for row in rows)
    total_tuned_full = sum(row["tuned_full"] for row in rows)
    lines = [
        "# Tuned Base-Stock + ReAllocation on Five Nonstationary SKUs",
        "",
        "Selection used independent fixed tuning and validation demand banks. The formal common-1000 test banks were loaded only after all five parameter vectors were locked.",
        "",
        "## Locked node-level target multipliers",
        "",
        "| SKU | CDC local | RDC 1 | RDC 2 | RDC 3 | RDC 4 | Train candidates | Validation candidates | Search wall time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['item_code']} | {row['cdc_local_scale']:.2f} | "
            f"{row['rdc_1_scale']:.2f} | {row['rdc_2_scale']:.2f} | "
            f"{row['rdc_3_scale']:.2f} | {row['rdc_4_scale']:.2f} | "
            f"{row['search_candidates']} | {row['validation_candidates']} | "
            f"{row['search_case_wall_seconds']:.1f} s |"
        )
    lines.extend(
        [
            "",
            "## Table 8-aligned controllable weekly cost",
            "",
            "| SKU | RA-SAC | DA + ReAllocation | Tuned base-stock + ReAllocation | Tuned vs. DA+RA | Tuned vs. RA-SAC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['item_code']} | {row['ra_sac_controllable']:.3f} | "
            f"{row['baseline_controllable']:.3f} | {row['tuned_controllable']:.3f} | "
            f"{row['tuned_vs_baseline_controllable']:+.2%} | "
            f"{row['tuned_vs_ra_sac_controllable']:+.2%} |"
        )
    lines.append(
        f"| **Five-SKU sum** | **{total_ra_ctl:.3f}** | **{total_base_ctl:.3f}** | "
        f"**{total_tuned_ctl:.3f}** | **{total_tuned_ctl / total_base_ctl - 1:+.2%}** | "
        f"**{total_tuned_ctl / total_ra_ctl - 1:+.2%}** |"
    )
    lines.extend(
        [
            "",
            "## Full 13-week cost",
            "",
            "| SKU | RA-SAC | DA + ReAllocation | Tuned base-stock + ReAllocation | Tuned vs. DA+RA | Tuned vs. RA-SAC | Fill rate (base -> tuned) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['item_code']} | {row['ra_sac_full']:.3f} | "
            f"{row['baseline_full']:.3f} | {row['tuned_full']:.3f} | "
            f"{row['tuned_vs_baseline_full']:+.2%} | {row['tuned_vs_ra_sac_full']:+.2%} | "
            f"{row['baseline_fill_rate']:.2%} -> {row['tuned_fill_rate']:.2%} |"
        )
    lines.append(
        f"| **Five-SKU sum** | **{total_ra_full:.3f}** | **{total_base_full:.3f}** | "
        f"**{total_tuned_full:.3f}** | **{total_tuned_full / total_base_full - 1:+.2%}** | "
        f"**{total_tuned_full / total_ra_full - 1:+.2%}** | -- |"
    )
    lines.extend(
        [
            "",
            "## Computation budget",
            "",
            f"- Tuning episodes per candidate and SKU: {protocol['tuning_episodes']}.",
            f"- Validation episodes per shortlisted candidate and SKU: {protocol['validation_episodes']}.",
            f"- Formal test episodes per locked policy and SKU: {protocol['test_episodes']}.",
            f"- Total unique tuning candidates: {sum(row['search_candidates'] for row in rows)}.",
            f"- Total validation candidate evaluations: {sum(row['validation_candidates'] for row in rows)}.",
            f"- End-to-end campaign wall time: {protocol['campaign_wall_seconds']:.1f} s with {protocol['max_workers']} parallel SKU workers.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(value.strip()) for value in text.split(",") if value.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-results", type=Path, default=SOURCE_RESULTS)
    parser.add_argument("--tuning-episodes", type=int, default=128)
    parser.add_argument("--validation-episodes", type=int, default=256)
    parser.add_argument("--test-episodes", type=int, default=1000)
    parser.add_argument("--tuning-seed-base", type=int, default=20260803)
    parser.add_argument("--validation-seed-base", type=int, default=20260805)
    parser.add_argument("--lower-bound", type=float, default=0.40)
    parser.add_argument("--upper-bound", type=float, default=1.80)
    parser.add_argument("--coarse-grid", default=",".join(map(str, DEFAULT_COARSE_GRID)))
    parser.add_argument("--step-sizes", default=",".join(map(str, DEFAULT_STEPS)))
    parser.add_argument("--max-candidates", type=int, default=150)
    parser.add_argument("--max-moves-per-step", type=int, default=6)
    parser.add_argument("--validation-top-k", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=5)
    args = parser.parse_args()
    if min(args.tuning_episodes, args.validation_episodes, args.test_episodes) <= 0:
        raise ValueError("Episode counts must be positive")
    if not 0.0 < args.lower_bound <= 1.0 <= args.upper_bound:
        raise ValueError("Bounds must satisfy 0 < lower <= 1 <= upper")

    started = time.perf_counter()
    output_dir = resolve_path(args.output_dir)
    source_results = resolve_path(args.source_results)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(source_results, args)
    search = {
        "objective": "mean controllable weekly cost",
        "parameterization": "one CDC-local and one multiplier per retailer target",
        "tuning_episodes": int(args.tuning_episodes),
        "validation_episodes": int(args.validation_episodes),
        "lower_bound": float(args.lower_bound),
        "upper_bound": float(args.upper_bound),
        "coarse_grid": list(parse_float_tuple(args.coarse_grid)),
        "step_sizes": list(parse_float_tuple(args.step_sizes)),
        "max_candidates": int(args.max_candidates),
        "max_moves_per_step": int(args.max_moves_per_step),
        "validation_top_k": int(args.validation_top_k),
    }
    atomic_json(
        output_dir / "launch_manifest.json",
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "tuning",
            "test_data_accessed": False,
            "source_results": portable_path(source_results),
            "output_dir": portable_path(output_dir),
            "max_workers": min(int(args.max_workers), len(jobs)),
            "search": search,
            "jobs": jobs,
        },
    )

    selections = run_parallel(
        tune_case,
        [(job, str(output_dir), search) for job in jobs],
        int(args.max_workers),
        "TUNED",
    )
    locked_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_json(
        output_dir / "locked_selections.json",
        {
            "status": "locked_before_test",
            "locked_at": locked_at,
            "test_data_accessed": False,
            "selection_metric": "validation mean controllable weekly cost",
            "selections": selections,
        },
    )
    manifest = json.loads((output_dir / "launch_manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "testing_locked_parameters",
            "locked_at": locked_at,
            "test_data_accessed": True,
        }
    )
    atomic_json(output_dir / "launch_manifest.json", manifest)

    results = run_parallel(
        test_case,
        [(selection, str(output_dir), int(args.test_episodes)) for selection in selections],
        int(args.max_workers),
        "TESTED",
    )
    protocol = {
        "horizon": HORIZON,
        "items": list(ITEM_CODES),
        "objective": "controllable weekly cost",
        "tuning_episodes": int(args.tuning_episodes),
        "validation_episodes": int(args.validation_episodes),
        "test_episodes": int(args.test_episodes),
        "tuning_seed_base": int(args.tuning_seed_base),
        "validation_seed_base": int(args.validation_seed_base),
        "formal_test_source": portable_path(source_results),
        "allocation_rule": "exact unit-marginal equality ReAllocation",
        "max_workers": min(int(args.max_workers), len(jobs)),
        "selection_locked_before_test": True,
        "campaign_wall_seconds": float(time.perf_counter() - started),
    }
    write_final_outputs(output_dir, selections, results, protocol)
    manifest = json.loads((output_dir / "launch_manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "complete",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "campaign_wall_seconds": protocol["campaign_wall_seconds"],
        }
    )
    atomic_json(output_dir / "launch_manifest.json", manifest)
    print(f"WROTE {output_dir}", flush=True)


if __name__ == "__main__":
    main()
