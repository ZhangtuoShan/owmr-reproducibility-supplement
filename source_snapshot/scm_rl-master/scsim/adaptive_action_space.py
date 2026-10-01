"""Adaptive MultiDiscrete action-space sizing for Kaynov-source PPO.

The Kaynov RLlib environment uses order quantities:
    action = [warehouse_order, retailer_1_order, ..., retailer_N_order]
and constructs ``spaces.MultiDiscrete(max_allowed_action)``.  Therefore each
entry in ``max_allowed_action`` is a branch count, and the largest feasible
raw action in that dimension is ``branch_count - 1``.

This module estimates those branch counts from demand/capacity information.
It is intentionally separate from ``train_ppo.py`` so existing source-code
training remains unchanged: generate or patch a scenario config first, then run
the original PPO training command.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml


def _project_root() -> Path:
    # scm_rl-master/scsim/adaptive_action_space.py -> project root
    return Path(__file__).resolve().parents[2]


def _as_list(value_or_list, n: int) -> List[Any]:
    if isinstance(value_or_list, (int, float, str)):
        return [value_or_list for _ in range(n)]
    if hasattr(value_or_list, "__iter__"):
        values = list(value_or_list)
        if len(values) != n:
            raise ValueError(f"Expected {n} values, got {len(values)}: {values!r}")
        return values
    raise TypeError(f"Unsupported scalar/list value: {value_or_list!r}")


def _normal_ppf(prob: float) -> float:
    try:
        from statistics import NormalDist

        return NormalDist().inv_cdf(prob)
    except Exception:  # pragma: no cover - very old Python fallback
        from scipy import stats

        return float(stats.norm.ppf(prob))


def _poisson_ppf(lam: float, prob: float) -> int:
    if lam <= 0:
        return 0
    try:
        from scipy import stats

        return int(stats.poisson.ppf(prob, lam))
    except Exception:
        cdf = 0.0
        pmf = math.exp(-lam)
        k = 0
        while True:
            cdf += pmf
            if cdf >= prob:
                return k
            k += 1
            pmf *= lam / k
            if k > max(1000, int(lam + 20 * math.sqrt(lam + 1))):
                return k


def _floor_clipped_single_period_upper(
    distribution: str,
    mean_or_low: float,
    std_or_high: float,
    quantile: float,
    max_demand: int = 500,
) -> int:
    """Return an integer demand upper bound under Kaynov's floor+clip sampling.

    Source demand generation samples a raw value, then applies:
        floor(raw), clip(..., 0, 500)
    For normal/uniform, using ceil(raw quantile) matches the hand-tuned Kaynov
    K5/K6 action ranges better than using a discrete post-floor CDF inverse.
    """
    q = float(np.clip(quantile, 0.0, 1.0))
    dist = distribution.lower()
    if dist == "poisson":
        upper = _poisson_ppf(float(mean_or_low), q)
    elif dist == "truncated_poisson":
        upper = min(_poisson_ppf(float(mean_or_low), q), int(std_or_high))
    elif dist == "normal":
        std = max(float(std_or_high), 0.0)
        if std <= 0:
            upper = math.ceil(float(mean_or_low))
        else:
            upper = math.ceil(float(mean_or_low) + std * _normal_ppf(q))
    elif dist == "uniform":
        low = float(mean_or_low)
        high = float(std_or_high)
        if high < low:
            low, high = high, low
        upper = math.ceil(low + q * (high - low))
    else:
        raise ValueError(f"Unsupported demand distribution: {distribution!r}")
    return int(np.clip(upper, 0, max_demand))


def _sample_floor_clipped_demands(
    distributions: Iterable[str],
    means: Iterable[float],
    stds: Iterable[float],
    sample_size: int,
    seed: int,
    max_demand: int = 500,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = []
    for dist, mean, std in zip(distributions, means, stds):
        dist_l = str(dist).lower()
        if dist_l == "poisson":
            raw = rng.poisson(float(mean), size=sample_size)
        elif dist_l == "truncated_poisson":
            raw = np.minimum(rng.poisson(float(mean), size=sample_size), int(std))
        elif dist_l == "normal":
            raw = rng.normal(float(mean), max(float(std), 0.0), size=sample_size)
        elif dist_l == "uniform":
            low = float(mean)
            high = float(std)
            if high < low:
                low, high = high, low
            raw = rng.uniform(low, high, size=sample_size)
        else:
            raise ValueError(f"Unsupported demand distribution: {dist!r}")
        samples.append(np.clip(np.floor(raw), 0, max_demand))
    return np.vstack(samples).T


def estimate_aggregate_order_upper(
    env_config: Dict[str, Any],
    cdc_quantile: float = 0.95,
    cdc_multiplier: float = 1.10,
    cdc_round_to: int = 1,
    aggregate_sample_size: int = 200_000,
    aggregate_seed: int = 20260702,
) -> int:
    """Estimate a high aggregate single-period order quantity for the CDC.

    This is an order-quantity scale, not a base-stock/echelon-position scale.
    It is used both for finite production-capacity defaults and for sizing the
    warehouse branch in Kaynov-source ``MultiDiscrete`` action spaces.
    """
    n = int(env_config.get("number_of_stores", 3))
    distributions = [str(x).lower() for x in _as_list(env_config.get("demand_distribution", "poisson"), n)]
    means = [float(x) for x in _as_list(env_config.get("mean_demand_stores", 3), n)]
    stds = [float(x) for x in _as_list(env_config.get("std_demand_stores", 0), n)]
    aggregate = _sample_floor_clipped_demands(
        distributions,
        means,
        stds,
        sample_size=int(aggregate_sample_size),
        seed=int(aggregate_seed),
    ).sum(axis=1)
    cdc_upper = float(np.quantile(aggregate, float(cdc_quantile))) * float(cdc_multiplier)
    if cdc_round_to and cdc_round_to > 1:
        cdc_upper = math.ceil(cdc_upper / cdc_round_to) * cdc_round_to
    return max(1, int(math.ceil(cdc_upper)))


def estimate_warehouse_action_unit(
    cdc_order_upper: float,
    target_cdc_branches: int = 80,
    min_unit: int = 1,
) -> int:
    """Return an integer unit so raw CDC actions stay within target branches."""
    target_raw_max = max(1, int(target_cdc_branches) - 1)
    return max(int(min_unit), int(math.ceil(float(cdc_order_upper) / target_raw_max)))


def estimate_source_capacities(
    env_config: Dict[str, Any],
    cdc_quantile: float = 0.95,
    cdc_multiplier: float = 1.10,
    cdc_round_to: int = 1,
    aggregate_sample_size: int = 200_000,
    aggregate_seed: int = 20260702,
    production_capacity_multiplier: float = 1.0,
    warehouse_capacity_multiplier: float = 10.0,
) -> Dict[str, int]:
    """Estimate finite source-PPO capacities on the same order-quantity scale."""
    cdc_upper = estimate_aggregate_order_upper(
        env_config,
        cdc_quantile=cdc_quantile,
        cdc_multiplier=cdc_multiplier,
        cdc_round_to=cdc_round_to,
        aggregate_sample_size=aggregate_sample_size,
        aggregate_seed=aggregate_seed,
    )
    production_capacity = max(1, int(math.ceil(cdc_upper * float(production_capacity_multiplier))))
    warehouse_start_level = float(env_config.get("warehouse_start_level", 0) or 0)
    warehouse_capacity = max(
        production_capacity,
        int(math.ceil(max(warehouse_start_level, production_capacity) * float(warehouse_capacity_multiplier))),
    )
    return {
        "cdc_order_upper": int(cdc_upper),
        "production_capacity": int(production_capacity),
        "warehouse_capacity": int(warehouse_capacity),
    }


def estimate_max_allowed_action(
    env_config: Dict[str, Any],
    retailer_quantile: float = 0.92,
    high_variance_normal_quantile: float = 0.958,
    cdc_quantile: float = 0.95,
    cdc_multiplier: float = 1.10,
    cdc_round_to: int = 1,
    warehouse_action_unit: Optional[int] = None,
    min_small_case_branch: int = 2,
    min_large_case_retailer_branch: int = 2,
    large_case_threshold: int = 10,
    aggregate_sample_size: int = 200_000,
    aggregate_seed: int = 20260702,
) -> List[int]:
    """Estimate Kaynov-source ``max_allowed_action`` branch counts.

    Source PPO actions are order quantities, not order-up-to/base-stock levels.
    Therefore this estimator sizes each branch around single-period order needs:
    retailer branches use single-period demand quantiles, while the warehouse
    branch uses an aggregate single-period demand quantile unless a production
    capacity is explicitly present.

    This intentionally avoids the old fixed 30/26 floors, which made low-demand
    or many-retailer cases explore large over-ordering actions.  Existing
    Kaynov K5/K6-style cases with ``production_capacity=100`` still map to CDC
    branch count 101, matching the source-paper configuration.
    """
    n = int(env_config.get("number_of_stores", 3))
    distributions = [str(x).lower() for x in _as_list(env_config.get("demand_distribution", "poisson"), n)]
    means = [float(x) for x in _as_list(env_config.get("mean_demand_stores", 3), n)]
    stds = [float(x) for x in _as_list(env_config.get("std_demand_stores", 0), n)]

    if n >= large_case_threshold:
        min_retailer_branch = int(min_large_case_retailer_branch)
    else:
        min_retailer_branch = int(min_small_case_branch)

    retailer_branches: List[int] = []
    for dist, mean, std in zip(distributions, means, stds):
        q = float(retailer_quantile)
        if dist == "normal" and mean <= 1.0 and std >= 10.0:
            q = float(high_variance_normal_quantile)
        upper = _floor_clipped_single_period_upper(dist, mean, std, q)
        retailer_branches.append(max(min_retailer_branch, upper + 1))

    raw_production_capacity = float(env_config.get("production_capacity", 0) or 0)
    if raw_production_capacity > 0:
        cdc_upper = raw_production_capacity
    else:
        cdc_upper = estimate_aggregate_order_upper(
            env_config,
            cdc_quantile=cdc_quantile,
            cdc_multiplier=cdc_multiplier,
            cdc_round_to=cdc_round_to,
            aggregate_sample_size=aggregate_sample_size,
            aggregate_seed=aggregate_seed,
        )
    unit = int(warehouse_action_unit if warehouse_action_unit is not None else env_config.get("warehouse_action_unit", 1))
    unit = max(1, unit)
    cdc_branch = max(2, int(math.ceil(float(cdc_upper) / unit)) + 1)

    return [cdc_branch] + retailer_branches


def _project_case_to_source_config(
    data_tuple: Tuple[Any, Any, Any, Any, Any, Any, Any],
    is_retailer_backlog: bool,
    is_emergency_shipment: bool,
    prob_customer_waiting: float,
    allocation_type: str,
    production_capacity: float,
    warehouse_capacity: float,
    episode_length: int,
) -> Dict[str, Any]:
    rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, rdc_inventory, cdc_inventory = data_tuple
    keys = list(rdc_dict.keys())
    distributions = []
    means = []
    stds = []
    for k in keys:
        spec = demand_feature[k]
        if isinstance(spec, (int, float)):
            distributions.append("poisson")
            means.append(float(spec))
            stds.append(math.sqrt(float(spec)))
        else:
            dist = str(spec[0]).lower()
            distributions.append(dist)
            if dist == "normal":
                means.append(float(spec[1]))
                stds.append(math.sqrt(max(float(spec[2]), 0.0)))
            elif dist == "uniform":
                # Kaynov source uses mean_demand_stores as the lower bound and
                # std_demand_stores as the upper bound for uniform demand.
                means.append(float(spec[1]))
                stds.append(float(spec[2]))
            elif dist == "poisson":
                means.append(float(spec[1]))
                stds.append(math.sqrt(float(spec[1])))
            elif dist == "truncated_poisson":
                means.append(float(spec[1]))
                stds.append(float(spec[2]))
            else:
                raise ValueError(f"Unsupported project demand spec: {spec!r}")

    return {
        "number_of_stores": len(keys),
        "prob_customer_waiting": prob_customer_waiting,
        "allocation_type": allocation_type,
        "delay_to_warehouse": int(item_lead_time),
        "delay_to_stores": [int(rdc_dict[k]) for k in keys],
        "warehouse_holding_cost": float(holding_cost[0]),
        "store_holding_cost": [float(holding_cost[k]) for k in keys],
        "shortage_cost": [float(backorder_cost[k]) for k in keys],
        "is_retailer_backlog": bool(is_retailer_backlog),
        "store_capacity": 0 if is_retailer_backlog else 100000,
        "is_emergency_shipment": bool(is_emergency_shipment),
        "cost_emergency_shipment": 0,
        "demand_distribution": distributions,
        "mean_demand_stores": means,
        "std_demand_stores": stds,
        "episode_length": int(episode_length),
        "full_state": False,
        "stores_start_level": [int(np.asarray(rdc_inventory[k])[0]) for k in keys],
        "warehouse_start_level": int(np.asarray(cdc_inventory)[0]),
        "warehouse_capacity": float(warehouse_capacity),
        "production_capacity": float(production_capacity),
    }


def load_project_case_from_cli(args) -> Dict[str, Any]:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if args.kaynov_index is not None:
        from dataset import data_kaynov

        data_tuple = data_kaynov()[int(args.kaynov_index)]
    else:
        from dataset import generate_data_with_random_para

        data_tuple = generate_data_with_random_para(
            rdc_num=int(args.rdc_num),
            sample_num=max(int(args.sample_index) + 1, int(args.sample_num)),
            random_seed=int(args.random_seed),
        )[int(args.sample_index)]
    return _project_case_to_source_config(
        data_tuple=data_tuple,
        is_retailer_backlog=args.is_retailer_backlog,
        is_emergency_shipment=args.is_emergency_shipment,
        prob_customer_waiting=args.prob_customer_waiting,
        allocation_type=args.allocation_type,
        production_capacity=args.production_capacity,
        warehouse_capacity=args.warehouse_capacity,
        episode_length=args.episode_length,
    )


def patch_or_create_scenario(
    config: Dict[str, Any],
    output_scenario_dir: Path,
    source_scenario_dir: Path | None = None,
) -> None:
    output_scenario_dir.mkdir(parents=True, exist_ok=True)
    if source_scenario_dir is not None:
        for name in ("training_config.yaml", "evaluation_config.yaml", "search_spaces.yaml"):
            src = source_scenario_dir / name
            if src.exists():
                shutil.copy2(src, output_scenario_dir / name)
    for name in ("training_config.yaml", "evaluation_config.yaml"):
        path = output_scenario_dir / name
        if not path.exists():
            path.write_text("{}\n", encoding="utf-8")
    with (output_scenario_dir / "default_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate Kaynov-source PPO max_allowed_action.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=str, help="Existing Kaynov default_config.yaml to patch/inspect.")
    source.add_argument("--kaynov-index", type=int, help="Build config from this project's data_kaynov()[index].")
    source.add_argument("--random-n20-index", dest="sample_index", type=int, help="Build config from random n20 sample index.")
    parser.add_argument("--rdc-num", type=int, default=20)
    parser.add_argument("--sample-num", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=10)
    parser.add_argument("--retailer-quantile", type=float, default=0.92)
    parser.add_argument("--high-variance-normal-quantile", type=float, default=0.958)
    parser.add_argument("--cdc-quantile", type=float, default=0.95)
    parser.add_argument("--cdc-multiplier", type=float, default=1.10)
    parser.add_argument("--cdc-round-to", type=int, default=1)
    parser.add_argument("--warehouse-action-unit", type=int, default=None)
    parser.add_argument("--target-cdc-branches", type=int, default=None)
    parser.add_argument("--set-finite-capacities", action="store_true")
    parser.add_argument("--production-capacity-multiplier", type=float, default=1.0)
    parser.add_argument("--warehouse-capacity-multiplier", type=float, default=10.0)
    parser.add_argument("--min-small-case-branch", type=int, default=30)
    parser.add_argument("--min-large-case-retailer-branch", type=int, default=26)
    parser.add_argument("--large-case-threshold", type=int, default=10)
    parser.add_argument("--aggregate-sample-size", type=int, default=200_000)
    parser.add_argument("--aggregate-seed", type=int, default=20260702)
    parser.add_argument("--output-scenario-dir", type=str, default=None)
    parser.add_argument("--allocation-type", type=str, default="proportional_allocation")
    parser.add_argument("--is-retailer-backlog", action="store_true")
    parser.add_argument("--is-emergency-shipment", action="store_true")
    parser.add_argument("--prob-customer-waiting", type=float, default=0.8)
    parser.add_argument("--production-capacity", type=float, default=0.0)
    parser.add_argument("--warehouse-capacity", type=float, default=0.0)
    parser.add_argument("--episode-length", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = None
    if args.config:
        config_path = Path(args.config)
        source_dir = config_path.parent
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = load_project_case_from_cli(args)

    if args.set_finite_capacities:
        capacities = estimate_source_capacities(
            config,
            cdc_quantile=args.cdc_quantile,
            cdc_multiplier=args.cdc_multiplier,
            cdc_round_to=args.cdc_round_to,
            aggregate_sample_size=args.aggregate_sample_size,
            aggregate_seed=args.aggregate_seed,
            production_capacity_multiplier=args.production_capacity_multiplier,
            warehouse_capacity_multiplier=args.warehouse_capacity_multiplier,
        )
        config["production_capacity"] = capacities["production_capacity"]
        config["warehouse_capacity"] = capacities["warehouse_capacity"]

    warehouse_action_unit = args.warehouse_action_unit
    if warehouse_action_unit is None and args.target_cdc_branches is not None:
        raw_production_capacity = float(config.get("production_capacity", 0) or 0)
        if raw_production_capacity > 0:
            cdc_order_upper = raw_production_capacity
        else:
            cdc_order_upper = estimate_aggregate_order_upper(
                config,
                cdc_quantile=args.cdc_quantile,
                cdc_multiplier=args.cdc_multiplier,
                cdc_round_to=args.cdc_round_to,
                aggregate_sample_size=args.aggregate_sample_size,
                aggregate_seed=args.aggregate_seed,
            )
        warehouse_action_unit = estimate_warehouse_action_unit(
            cdc_order_upper,
            target_cdc_branches=args.target_cdc_branches,
        )
    if warehouse_action_unit is not None:
        config["warehouse_action_unit"] = int(max(1, warehouse_action_unit))

    max_allowed_action = estimate_max_allowed_action(
        config,
        retailer_quantile=args.retailer_quantile,
        high_variance_normal_quantile=args.high_variance_normal_quantile,
        cdc_quantile=args.cdc_quantile,
        cdc_multiplier=args.cdc_multiplier,
        cdc_round_to=args.cdc_round_to,
        warehouse_action_unit=config.get("warehouse_action_unit", 1),
        min_small_case_branch=args.min_small_case_branch,
        min_large_case_retailer_branch=args.min_large_case_retailer_branch,
        large_case_threshold=args.large_case_threshold,
        aggregate_sample_size=args.aggregate_sample_size,
        aggregate_seed=args.aggregate_seed,
    )
    config["max_allowed_action"] = max_allowed_action

    print("max_allowed_action:", max_allowed_action)
    print("max raw action:", [x - 1 for x in max_allowed_action])
    print("warehouse_action_unit:", config.get("warehouse_action_unit", 1))
    print("production_capacity:", config.get("production_capacity", 0))
    print("warehouse_capacity:", config.get("warehouse_capacity", 0))
    if args.output_scenario_dir:
        output_dir = Path(args.output_scenario_dir)
        patch_or_create_scenario(config, output_dir, source_scenario_dir=source_dir)
        print(f"wrote scenario config to: {output_dir}")


if __name__ == "__main__":
    main()
