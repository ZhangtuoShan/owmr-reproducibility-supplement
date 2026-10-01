import datetime
import logging
import math
import os
import random as rd
from copy import deepcopy
from logging.handlers import RotatingFileHandler

import numpy as np
import pandas as pd
import torch
from scipy import integrate, stats
from scipy.signal import fftconvolve

from dataset import *
from demand_utils import (
    demand_mean,
    demand_quantile,
    lead_time_demand_pmf,
    lead_time_demand_quantile,
    quantile_series_lead_time_pmf,
    quantile_series_sum_window_pmf,
    sample_demand,
    sample_quantile_series_demand,
)
from elegantrl.agents import (AgentA2C, AgentConvexSAC, AgentDDPG, AgentDQN,
                              AgentPPO, AgentSAC, AgentTD3)
from elegantrl.train.config import Config
from elegantrl.train.run import *
from evaluation_statistics import episode_average_statistics
from my_log import timer
from utils import logger
import heapq
import random

'''
0227
one-warehouse-multiple-retailers (OWMR) algorithms (one item)
'''


### Input
'''
demand_feature = {1: 20, 2, 30, 3: 25}
rdc_dict = {1: 2, 2: 3, 3: 3}
holding_cost = {0: 1, 1: 1.2, 2, 1.4, 3: 1.6}
backorder_cost = {1: 10, 2: 10, 3: 10, 4: 10}
item_lead_time = 2
'''

def kwargs_filter(function, kwargs: dict) -> dict:
    import inspect
    parameters = inspect.signature(function).parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
        return kwargs
    sign = {val.name for val in parameters}
    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}  # filtered kwargs


def generate_env(env_class=None, env_args: dict = None, gpu_id: int = -1):
    env_args['gpu_id'] = gpu_id  # set gpu_id for vectorized env before build it

    env = env_class(**kwargs_filter(env_class.__init__, env_args.copy()))

    return env


def my_mkdir(path):
    path = os.getcwd() + path
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    return path
LOG_DIR = my_mkdir('/log')

def record_log():
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    LOG_DIR

    log_path = os.path.join(LOG_DIR, "logger_{}_{}.log".format(
        'reinforcelearn-inv-replenish', datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y%m%d')))

    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] - %(module)s.%(funcName)s (%('
        'filename)s:%(lineno)d) - %(message)s')

    file_handler = RotatingFileHandler(
        filename=log_path, maxBytes=10 * 1024 * 1024, backupCount=30)
    file_handler.setFormatter(formatter)

    # 在代码运行主程序处，打印出日志
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = record_log()


class OWMR_input:
    def __init__(self, rdc_dict: dict, demand_feature, holding_cost, backorder_cost, item_lead_time, RDC_inventory, CDC_inventory):
        self.rdc_dict = rdc_dict.copy()
        self.demand_feature = demand_feature.copy()
        self.holding_cost = holding_cost.copy()
        self.backorder_cost = backorder_cost.copy()
        self.item_lead_time = item_lead_time

        self.echelon_holding_cost = {}
        self.Poisson_para = {}
        self.demand_mean = {}
        for j in self.rdc_dict.keys():
            self.echelon_holding_cost[j] = self.holding_cost[j] - self.holding_cost[0]
            self.demand_mean[j] = demand_mean(self.demand_feature[j])
            self.Poisson_para[j] = self.demand_mean[j] * (self.rdc_dict[j] + 1)
        self.Poisson_para[0] = (self.item_lead_time + 1) * np.sum(list(self.demand_mean.values()))
        self.RDC_inventory = RDC_inventory
        self.CDC_inventory = CDC_inventory



def low_round(data, i):
    return round(data - 0.1 ** (i + 1) * 5, i)


def _round_up_to_multiple(value, multiple):
    multiple = float(multiple)
    if multiple <= 0:
        return float(value)
    return float(math.ceil(float(value) / multiple) * multiple)


def _round_down_to_multiple(value, multiple):
    multiple = float(multiple)
    if multiple <= 0:
        return float(value)
    return float(math.floor(float(value) / multiple) * multiple)


def compute_joint_horizon_cdc_bound(
        info_input,
        quantile=0.01,
        round_to=1.0,
        round_up=False,
):
    """Compute a CDC echelon bound without duplicating the review period.

    For each RDC, the protected demand horizon is ``L0 + Lk + 1``. The PMFs
    are convolved before taking one system-level quantile. Historical
    additive bounds remain available explicitly for checkpoint reproduction.
    """
    total_pmf = np.array([1.0], dtype=float)
    support_offset = 0.0
    for rdc, lead_time in info_input.rdc_dict.items():
        periods = int(info_input.item_lead_time) + int(lead_time) + 1
        support, pmf = lead_time_demand_pmf(
            info_input.demand_feature[rdc],
            periods,
        )
        support_offset += float(support[0])
        total_pmf = np.convolve(total_pmf, np.asarray(pmf, dtype=float))

    total_pmf = np.maximum(total_pmf, 0.0)
    total = float(total_pmf.sum())
    if total <= 0.0:
        raise ValueError("Joint CDC demand PMF has zero mass.")
    total_pmf /= total
    q = float(np.clip(quantile, 0.0, 1.0))
    index = int(np.searchsorted(np.cumsum(total_pmf), q, side="left"))
    index = min(max(index, 0), len(total_pmf) - 1)
    value = support_offset + index
    rounding = _round_up_to_multiple if round_up else _round_down_to_multiple
    return float(rounding(value, round_to))


def compute_joint_horizon_cdc_lower_bound(
        info_input,
        quantile=0.01,
        round_to=1.0,
):
    return compute_joint_horizon_cdc_bound(
        info_input,
        quantile=quantile,
        round_to=round_to,
        round_up=False,
    )


def compute_joint_horizon_cdc_upper_bound(
        info_input,
        quantile=0.92,
        round_to=1.0,
):
    return compute_joint_horizon_cdc_bound(
        info_input,
        quantile=quantile,
        round_to=round_to,
        round_up=True,
    )


def expand_upper_tail_quantile(base_quantile, tail_retention=0.25, quantile_cap=0.99):
    """Turn an operating quantile into a conservative search-envelope quantile."""
    base_quantile = float(base_quantile)
    tail_retention = float(tail_retention)
    quantile_cap = float(quantile_cap)
    if not 0.0 <= base_quantile <= 1.0:
        raise ValueError("base_quantile must be in [0, 1]")
    if not 0.0 <= tail_retention <= 1.0:
        raise ValueError("tail_retention must be in [0, 1]")
    if not 0.0 < quantile_cap < 1.0:
        raise ValueError("quantile_cap must be in (0, 1)")
    if quantile_cap + 1e-12 < base_quantile:
        raise ValueError(
            f"quantile_cap={quantile_cap} is below base quantile={base_quantile}"
        )
    return min(
        quantile_cap,
        1.0 - tail_retention * (1.0 - base_quantile),
    )


def compute_adaptive_action_space_overrides(
        info_input,
        mode="cost_aware",
        quantile=0.92,
        cost_margin=0.03,
        min_quantile=0.85,
        max_quantile=0.95,
        cdc_multiplier=1.0,
        cdc_buffer_alpha=0.0,
        cdc_buffer_mode="required",
        cdc_horizon_mode="joint",
        cdc_upper_tail_retention=0.25,
        cdc_upper_quantile_cap=0.99,
        cdc_cost_margin=None,
        cdc_min_quantile=None,
        cdc_max_quantile=None,
        small_case_beta=0.0,
        small_case_threshold=3,
        rdc_round_to=1.0,
        cdc_round_to=1.0,
        min_rdc_bsl=1.0,
        rdc_lower_fraction=0.0,
        cdc_lower_fraction=0.25,
        rdc_lower_quantile=None,
        cdc_lower_quantile=None,
        return_bounds=False,
        return_quantiles=False,
):
    """Compute training-friendly continuous-action BSL bounds.

    The fixed quantile mode is intentionally below the old 0.99 construction.
    In the Kaynov K5 experiments, q=0.92 maps to RDC=50 and CDC=sum(RDC)=500,
    which gave a useful SAC-eq action-space trade-off after demand alignment.
    CDC actions in this project are echelon/base-stock targets rather than
    Kaynov-style order quantities. Therefore the default ``required`` mode
    always includes one full upstream lead-time demand buffer. In this mode,
    ``cdc_buffer_alpha`` controls only additional safety margin. The
    ``legacy_scaled`` mode preserves the historical formula where alpha scales
    the entire upstream buffer; it exists only to reproduce older checkpoints.

    The default cost-aware mode uses a conservative newsvendor-inspired
    service level:
        q_k = clip(b_k / (b_k + h_eff_k) - cost_margin,
                   min_quantile, max_quantile)
    where h_eff_k is the echelon holding cost h_k - h_0, falling back to h_k
    when the echelon term degenerates. The margin avoids expanding the action
    space all the way to a high critical fractile, which was empirically too
    broad for SAC on Kaynov K5. For a joint CDC horizon, this service level is
    treated as an operating quantile rather than a hard action upper bound.
    The CDC upper envelope retains ``cdc_upper_tail_retention`` of its residual
    upper-tail probability, capped by ``cdc_upper_quantile_cap``. Explicit
    ``additive`` mode preserves the historical formula for reproducibility.
    """
    mode = str(mode).lower()
    if mode not in {"fixed", "cost_aware", "cost-aware", "newsvendor"}:
        raise ValueError(f"Unsupported adaptive action-space mode: {mode!r}")
    cdc_buffer_mode = str(cdc_buffer_mode).lower().replace("-", "_")
    if cdc_buffer_mode not in {"required", "legacy_scaled"}:
        raise ValueError(f"Unsupported CDC buffer mode: {cdc_buffer_mode!r}")
    cdc_horizon_mode = str(cdc_horizon_mode).lower().replace("-", "_")
    if cdc_horizon_mode not in {"additive", "joint"}:
        raise ValueError(f"Unsupported CDC horizon mode: {cdc_horizon_mode!r}")
    if cdc_horizon_mode == "joint" and cdc_buffer_mode != "required":
        raise ValueError("Joint CDC horizons require cdc_buffer_mode='required'.")
    if float(cdc_buffer_alpha) < 0.0:
        raise ValueError("cdc_buffer_alpha must be nonnegative")
    cdc_buffer_weight = float(cdc_buffer_alpha)
    if cdc_buffer_mode == "required":
        cdc_buffer_weight += 1.0
    fallback_q = float(np.clip(quantile, 0.0, 1.0))
    min_q = float(np.clip(min_quantile, 0.0, 1.0))
    max_q = float(np.clip(max_quantile, min_q, 1.0))
    cdc_margin = float(cost_margin if cdc_cost_margin is None else cdc_cost_margin)
    cdc_min_q = min_q if cdc_min_quantile is None else float(np.clip(cdc_min_quantile, 0.0, 1.0))
    cdc_max_q = max_q if cdc_max_quantile is None else float(np.clip(cdc_max_quantile, cdc_min_q, 1.0))
    cdc_upper_tail_retention = float(cdc_upper_tail_retention)
    cdc_upper_quantile_cap = float(cdc_upper_quantile_cap)
    if not 0.0 <= cdc_upper_tail_retention <= 1.0:
        raise ValueError("cdc_upper_tail_retention must be in [0, 1]")
    if not 0.0 < cdc_upper_quantile_cap < 1.0:
        raise ValueError("cdc_upper_quantile_cap must be in (0, 1)")
    cost_eps = 1e-8

    rdc_overrides = {}
    rdc_lower_overrides = {}
    rdc_quantiles = {}
    cdc_quantiles = []
    cdc_buffer_upper = 0.0
    cdc_buffer_lower = 0.0
    default_rdc_upper = 0.0
    default_upstream_upper = 0.0
    default_rdc_lower = 0.0
    default_upstream_lower = 0.0
    use_rdc_lower = (rdc_lower_quantile is not None) or (float(rdc_lower_fraction) > 0.0)
    use_cdc_lower = (cdc_lower_quantile is not None) or (float(cdc_lower_fraction) > 0.0)
    rdc_lower_q = None if rdc_lower_quantile is None else float(np.clip(rdc_lower_quantile, 0.0, 1.0))
    cdc_lower_q = None if cdc_lower_quantile is None else float(np.clip(cdc_lower_quantile, 0.0, 1.0))
    for k, lead_time in info_input.rdc_dict.items():
        q_k = fallback_q
        q_cdc_k = fallback_q
        if mode in {"cost_aware", "cost-aware", "newsvendor"}:
            warehouse_holding = float(info_input.holding_cost.get(0, 0.0))
            rdc_holding = float(info_input.holding_cost.get(k, warehouse_holding))
            echelon_holding = rdc_holding - warehouse_holding
            h_eff = echelon_holding if echelon_holding > cost_eps else rdc_holding
            h_eff = max(h_eff, cost_eps)
            b_eff = max(float(info_input.backorder_cost.get(k, cost_eps)), 0.0)
            if b_eff > cost_eps:
                critical_fractile = b_eff / (b_eff + h_eff)
                q_k = float(np.clip(critical_fractile - float(cost_margin), min_q, max_q))
                cdc_critical_fractile = b_eff / (b_eff + max(warehouse_holding, cost_eps))
                q_cdc_k = float(np.clip(cdc_critical_fractile - cdc_margin, cdc_min_q, cdc_max_q))
        q_k = float(np.clip(q_k, 0.0, 1.0))
        q_cdc_k = float(np.clip(q_cdc_k, 0.0, 1.0))
        raw_upper = lead_time_demand_quantile(info_input.demand_feature[k], q_k, int(lead_time) + 1)
        rounded_upper = _round_up_to_multiple(raw_upper, rdc_round_to)
        rdc_overrides[k] = max(float(min_rdc_bsl), rounded_upper)
        rdc_quantiles[k] = q_k
        cdc_quantiles.append(q_cdc_k)
        original_rdc_lower = lead_time_demand_quantile(info_input.demand_feature[k], 0.01, int(lead_time) + 1)
        if use_rdc_lower:
            if rdc_lower_q is None:
                raw_lower = float(rdc_overrides[k]) * float(rdc_lower_fraction)
            else:
                raw_lower = lead_time_demand_quantile(info_input.demand_feature[k], rdc_lower_q, int(lead_time) + 1)
                if float(rdc_lower_fraction) > 0.0:
                    raw_lower = max(raw_lower, float(rdc_overrides[k]) * float(rdc_lower_fraction))
            rounded_lower = _round_down_to_multiple(raw_lower, rdc_round_to)
            rdc_lower_overrides[k] = min(max(float(original_rdc_lower), rounded_lower), float(rdc_overrides[k]))
        cdc_buffer_upper += lead_time_demand_quantile(
            info_input.demand_feature[k],
            q_cdc_k,
            int(info_input.item_lead_time) + 1,
        )
        lower_q_for_cdc = q_cdc_k if cdc_lower_q is None else cdc_lower_q
        cdc_buffer_lower += lead_time_demand_quantile(
            info_input.demand_feature[k],
            lower_q_for_cdc,
            int(info_input.item_lead_time) + 1,
        )
        default_rdc_upper += lead_time_demand_quantile(info_input.demand_feature[k], 0.99, int(lead_time) + 1)
        default_rdc_lower += original_rdc_lower
        default_upstream_upper += lead_time_demand_quantile(
            info_input.demand_feature[k],
            0.99,
            int(info_input.item_lead_time) + 1,
        )
        default_upstream_lower += lead_time_demand_quantile(
            info_input.demand_feature[k],
            0.01,
            int(info_input.item_lead_time) + 1,
        )

    default_cdc_upper = default_rdc_upper + default_upstream_upper
    default_cdc_lower = default_rdc_lower + default_upstream_lower
    if cdc_horizon_mode == "joint":
        joint_q = max(cdc_quantiles, default=fallback_q)
        cdc_envelope_q = expand_upper_tail_quantile(
            joint_q,
            tail_retention=cdc_upper_tail_retention,
            quantile_cap=cdc_upper_quantile_cap,
        )
        default_cdc_upper = compute_joint_horizon_cdc_upper_bound(
            info_input,
            quantile=0.99,
            round_to=cdc_round_to,
        )
        default_cdc_lower = compute_joint_horizon_cdc_lower_bound(
            info_input,
            quantile=0.01,
            round_to=cdc_round_to,
        )
        cdc_upper = compute_joint_horizon_cdc_upper_bound(
            info_input,
            quantile=cdc_envelope_q,
            round_to=cdc_round_to,
        )
        cdc_upper *= (1.0 + float(cdc_buffer_alpha)) * float(cdc_multiplier)
    else:
        cdc_upper = (sum(rdc_overrides.values()) + cdc_buffer_weight * cdc_buffer_upper) * float(cdc_multiplier)
    if small_case_beta > 0 and len(info_input.rdc_dict) <= int(small_case_threshold):
        cdc_upper = max(cdc_upper, float(small_case_beta) * default_cdc_upper)
    if default_cdc_upper > 0:
        cdc_upper = min(cdc_upper, default_cdc_upper)
    cdc_upper = _round_up_to_multiple(cdc_upper, cdc_round_to)

    cdc_lower = None
    if use_cdc_lower:
        if cdc_lower_q is None:
            raw_cdc_lower = float(cdc_upper) * float(cdc_lower_fraction)
        elif cdc_horizon_mode == "joint":
            raw_cdc_lower = compute_joint_horizon_cdc_lower_bound(
                info_input,
                quantile=cdc_lower_q,
                round_to=cdc_round_to,
            ) * float(cdc_multiplier)
            if float(cdc_lower_fraction) > 0.0:
                raw_cdc_lower = max(raw_cdc_lower, float(cdc_upper) * float(cdc_lower_fraction))
        else:
            rdc_lower_sum = 0.0
            for k, lead_time in info_input.rdc_dict.items():
                rdc_lower_sum += lead_time_demand_quantile(
                    info_input.demand_feature[k],
                    cdc_lower_q,
                    int(lead_time) + 1,
                )
            raw_cdc_lower = (rdc_lower_sum + cdc_buffer_weight * cdc_buffer_lower) * float(cdc_multiplier)
            if float(cdc_lower_fraction) > 0.0:
                raw_cdc_lower = max(raw_cdc_lower, float(cdc_upper) * float(cdc_lower_fraction))
        cdc_lower = min(max(float(default_cdc_lower), _round_down_to_multiple(raw_cdc_lower, cdc_round_to)), float(cdc_upper))

    if not use_rdc_lower:
        rdc_lower_overrides = None
    if return_quantiles:
        if return_bounds:
            return rdc_lower_overrides, cdc_lower, rdc_overrides, cdc_upper, rdc_quantiles
        return rdc_overrides, cdc_upper, rdc_quantiles
    if return_bounds:
        return rdc_lower_overrides, cdc_lower, rdc_overrides, cdc_upper
    return rdc_overrides, cdc_upper


def compute_tail_expanded_rdc_upper_overrides(
        info_input,
        tail_retention,
        quantile_cap=0.995,
        mode="cost_aware",
        quantile=0.92,
        cost_margin=0.03,
        min_quantile=0.85,
        max_quantile=0.95,
        rdc_round_to=1.0,
        min_rdc_bsl=1.0,
        return_quantiles=False,
):
    """Expand adaptive RDC upper bounds in probability space.

    ``tail_retention`` keeps a fraction of the base upper-tail probability:

        q_expanded = 1 - tail_retention * (1 - q_base)

    This gives comparable probability coverage across demand scales while
    ``quantile_cap`` prevents an accidental expansion into an extreme tail.
    CDC and RDC lower bounds are intentionally outside this helper so action
    space experiments can vary RDC upper coverage without confounding changes.
    """
    tail_retention = float(tail_retention)
    quantile_cap = float(quantile_cap)
    expand_upper_tail_quantile(0.0, tail_retention, quantile_cap)

    _, _, _, _, base_quantiles = compute_adaptive_action_space_overrides(
        info_input,
        mode=mode,
        quantile=quantile,
        cost_margin=cost_margin,
        min_quantile=min_quantile,
        max_quantile=max_quantile,
        rdc_lower_fraction=0.0,
        cdc_lower_fraction=0.0,
        cdc_horizon_mode="additive",
        return_bounds=True,
        return_quantiles=True,
    )
    expanded_bounds = {}
    expanded_quantiles = {}
    for k, lead_time in info_input.rdc_dict.items():
        base_q = float(base_quantiles[k])
        try:
            expanded_q = expand_upper_tail_quantile(
                base_q,
                tail_retention=tail_retention,
                quantile_cap=quantile_cap,
            )
        except ValueError as error:
            raise ValueError(f"Invalid RDC {k} tail expansion: {error}") from error
        raw_upper = lead_time_demand_quantile(
            info_input.demand_feature[k],
            expanded_q,
            int(lead_time) + 1,
        )
        expanded_bounds[k] = max(
            float(min_rdc_bsl),
            _round_up_to_multiple(raw_upper, rdc_round_to),
        )
        expanded_quantiles[k] = expanded_q

    if return_quantiles:
        return expanded_bounds, expanded_quantiles, base_quantiles
    return expanded_bounds


TABLE15_SIZE_AWARE_TAIL_RETENTION = {
    2: 0.15,
    6: 0.10,
    20: 1.0,
}


def compute_table15_size_aware_action_space_overrides(
        info_input,
        cdc_horizon_mode="joint",
        cdc_upper_tail_retention=0.25,
        cdc_upper_quantile_cap=0.99,
        return_metadata=False,
):
    """Return the validated size-aware action bounds for the 15 table cases.

    RDC upper-tail expansion is selected by RDC count from the representative
    2-1, 6-1, and 20-1 experiments. The joint CDC upper bound uses its own
    tail-envelope controls, independently of the RDC expansion. A retention of
    one means that the corresponding base quantile is left unexpanded.
    """
    rdc_count = len(info_input.rdc_dict)
    if rdc_count not in TABLE15_SIZE_AWARE_TAIL_RETENTION:
        supported = ", ".join(str(value) for value in sorted(TABLE15_SIZE_AWARE_TAIL_RETENTION))
        raise ValueError(
            f"Table15 size-aware action space supports RDC counts {{{supported}}}, "
            f"got {rdc_count}."
        )

    rdc_min, cdc_min, base_rdc_max, cdc_max, base_quantiles = (
        compute_adaptive_action_space_overrides(
            info_input,
            mode="cost_aware",
            cost_margin=0.03,
            min_quantile=0.85,
            max_quantile=0.95,
            cdc_buffer_alpha=0.0,
            cdc_buffer_mode="required",
            cdc_horizon_mode=cdc_horizon_mode,
            cdc_upper_tail_retention=cdc_upper_tail_retention,
            cdc_upper_quantile_cap=cdc_upper_quantile_cap,
            rdc_lower_fraction=0.0,
            cdc_lower_fraction=0.25,
            return_bounds=True,
            return_quantiles=True,
        )
    )
    tail_retention = TABLE15_SIZE_AWARE_TAIL_RETENTION[rdc_count]
    if tail_retention < 1.0:
        rdc_max, expanded_quantiles, _ = compute_tail_expanded_rdc_upper_overrides(
            info_input,
            tail_retention=tail_retention,
            quantile_cap=0.995,
            mode="cost_aware",
            cost_margin=0.03,
            min_quantile=0.85,
            max_quantile=0.95,
            return_quantiles=True,
        )
    else:
        rdc_max = base_rdc_max
        expanded_quantiles = base_quantiles

    if not return_metadata:
        return rdc_min, cdc_min, rdc_max, cdc_max

    if cdc_horizon_mode == "additive":
        profile = "table15_size_aware_v1"
    elif float(cdc_upper_tail_retention) == 1.0:
        profile = "table15_size_aware_v2_joint_cdc"
    else:
        profile = "table15_size_aware_v3_joint_cdc_tail"
    metadata = {
        "profile": profile,
        "cdc_horizon_mode": cdc_horizon_mode,
        "cdc_upper_tail_retention": float(cdc_upper_tail_retention),
        "cdc_upper_quantile_cap": float(cdc_upper_quantile_cap),
        "rdc_count": rdc_count,
        "tail_retention": tail_retention,
        "quantile_cap": 0.995,
        "base_quantiles": base_quantiles,
        "effective_quantiles": expanded_quantiles,
        "base_rdc_max": base_rdc_max,
    }
    return rdc_min, cdc_min, rdc_max, cdc_max, metadata

### DRL
class Rdc:
    def __init__(self, name: str, lt: int) -> None:
        self.name: str = name
        self.lead_time: int = lt


class OWMRInventoryEnv:
    def __init__(self, all_demand_list: dict, info_input: OWMR_input,
                holding_weight=1, penalty_weight=0,
                 cost_plus=100, demand_type='poisson', seed=None, if_generate_demand=True, method='SAC',
                 rho=1.0, backorder_proportion=None, allocation_rule='prioritized_greedy',
                 demand_fulfillment='partial_backorder', fulfillment_mode=None,
                 emergency_shipment_prob=None, kaynov_partial_backlogging=False,
                 discrete_action_lengths=None, rdc_bsl_min_override=None,
                 cdc_bsl_min_override=None, rdc_bsl_max_override=None,
                 cdc_bsl_max_override=None, allocation_seed=None,
                 include_stationary_action_mask_in_state=False,
                 total_periods=100) -> None:
        self.seed = seed
        if allocation_seed is None and seed is not None:
            allocation_seed = (int(seed) + 1_000_003) % (2 ** 32)
        self.allocation_seed = allocation_seed
        self.allocation_rng = random.Random(allocation_seed)
        if backorder_proportion is not None:
            rho = backorder_proportion
        self.rho = float(rho)
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError(f"rho/backorder_proportion must be in [0, 1], got {self.rho}.")
        self.backorder_proportion = self.rho
        self.demand_rng = np.random.default_rng(seed)
        self.backorder_rng = np.random.default_rng(seed)
        if fulfillment_mode is not None:
            demand_fulfillment = fulfillment_mode
        if kaynov_partial_backlogging:
            demand_fulfillment = 'kaynov_partial_backlogging'
        self.demand_fulfillment = str(demand_fulfillment).lower()
        self.is_kaynov_partial_backlogging = self.demand_fulfillment in {
            'kaynov_partial_backlogging',
            'kaynov_emergency',
            'kaynov_emergency_shipment',
            'kaynov_partial_backlogging_emergency',
            'kaynov_partial_backlogging_emergency_shipment',
            'partial_backlogging_emergency',
            'partial_lost_sales_emergency',
        }
        if emergency_shipment_prob is None:
            emergency_shipment_prob = self.rho
        self.emergency_shipment_prob = float(emergency_shipment_prob)
        if not 0.0 <= self.emergency_shipment_prob <= 1.0:
            raise ValueError(
                f"emergency_shipment_prob must be in [0, 1], got {self.emergency_shipment_prob}."
            )
        self.emergency_rng = np.random.default_rng(seed)
        self.rdc_size = len(info_input.rdc_dict)
        self.info_input = info_input
        self.all_demand_list = all_demand_list
        self.rdc_bsl_min_override = rdc_bsl_min_override
        self.cdc_bsl_min_override = cdc_bsl_min_override
        self.rdc_bsl_max_override = rdc_bsl_max_override
        self.cdc_bsl_max_override = cdc_bsl_max_override
        self.include_stationary_action_mask_in_state = bool(
            include_stationary_action_mask_in_state
        )

        example_rdc_demand = list(all_demand_list.values())[0]
        # example_demand = list(example_rdc_demand.values())[0]
        self.num_of_instances = np.shape(example_rdc_demand)[0]
        self.if_generate_demand = if_generate_demand

        self.rdcs = {}
        for k, v in info_input.rdc_dict.items():
            self.rdcs[k] = Rdc(k, v)
        self.demand_feature = info_input.demand_feature
        self.demand_type = demand_type
        self.initial_inventory_cdc = deepcopy(info_input.CDC_inventory)
        self.initial_inventory_rdc = deepcopy(info_input.RDC_inventory)
        self.item_lead_time = info_input.item_lead_time

        self.holding_cost = info_input.holding_cost
        self.backorder_cost = info_input.backorder_cost

        # dynamic initial state
        # 节点库存
        self.period = 0
        self._initialize_state()  # 假设所有lead time > 1，如果是1的话会有问题，当天生产/发运当天入库

        self._initialize_max_min_stock_level()
        self.holding_cost_revise = False  # 同 级库存水位，需要对不同价值、层级的编码的unit holding cost作正则化
        self.penalty_revise = True
        self.penalty_weight = penalty_weight  # 同 级库存水位，需要对不同价值、层级的编码的超订惩罚因子作正则化
        self.holding_weight = holding_weight
        self.cost_plus = cost_plus  # 给予成本结果一些正反馈

        # self._initialize_weights()

        self.env_name = 'OWMRInventoryEnv'

        self.compress_lead_time = False
        self.compress_size = 5

        # Keep the established stationary-state protocol by default. Explicit
        # experiments may append a constant mask when they intentionally
        # collapse a BSL dimension and want SAC to exclude it from entropy.
        self.stationary_action_mask = None
        self.action_mask_state_start = None
        self.action_mask_dim = 0
        self.inactive_action_value = 0.0

        # if self.compress_lead_time:
        self.state_dim = len(self.get_state())

        # self.state_dim = len(self.selling_item) * sum([self.rdcs[k].lead_time for k in self.rdcs.keys()]) + sum([self.items_dict[k].lead_time for k in self.items_dict.keys()])
        self.method = method
        if self.method != 'paper':
            self.action_dim = self.rdc_size + 1
        else:
            if discrete_action_lengths is not None:
                self.length = [int(x) for x in discrete_action_lengths]
                if len(self.length) != self.rdc_size + 1:
                    raise ValueError(
                        "discrete_action_lengths must contain one branch per RDC plus one CDC branch; "
                        f"got {len(self.length)} entries for {self.rdc_size} RDCs."
                    )
                if any(x <= 0 for x in self.length):
                    raise ValueError(f"discrete_action_lengths entries must be positive, got {self.length}.")
                self.action_dim = sum(self.length)
            else:
                self.action_dim = sum(self.RDC_max_bsl_one_period.values()) + self.rdc_size + self.CDC_max_bsl_one_period + 1  # dicretePPO action (0, 1, ..., r1, 0, 1, ..., r2, ..., rn, 0, 1, ..., c)
                self.length = [x + 1 for x in self.RDC_max_bsl_one_period.values()] + [self.CDC_max_bsl_one_period + 1]
        if self.method != 'paper' and self.__class__ is OWMRInventoryEnv:
            action_mask = np.ones(self.action_dim, dtype=float)
            for n, k in enumerate(self.rdcs.keys()):
                if self.RDC_max_bsl[k] - self.RDC_min_bsl[k] <= 1e-12:
                    action_mask[n] = 0.0
            if self.CDC_max_bsl - self.CDC_min_bsl <= 1e-12:
                action_mask[-1] = 0.0
            if np.any(action_mask == 0.0):
                self.stationary_action_mask = action_mask
            if (
                    self.stationary_action_mask is not None
                    and self.include_stationary_action_mask_in_state):
                self.action_mask_state_start = self.state_dim
                self.action_mask_dim = self.action_dim
                self.state_dim += self.action_dim
        self.total_periods = int(total_periods)
        if self.total_periods <= 0:
            raise ValueError(f"total_periods must be positive, got {total_periods}.")
        self.if_discrete = False

        self.episode = 0
        self.state = self.get_state()

        self.allocation_rule = self._resolve_allocation_rule(allocation_rule)

    @staticmethod
    def _resolve_allocation_rule(allocation_rule):
        if callable(allocation_rule):
            return allocation_rule

        rule_map = {
            'proportion': allocation_proportion,
            'prioritized_greedy': allocation_prioritized_greedy,
            'prioritized_greedy_equality': allocation_prioritized_greedy,
            'prioritized_greedy_coarse_legacy': allocation_prioritized_greedy_coarse_legacy,
            'prioritized_greedy_inequality': allocation_prioritized_greedy_inequality,
            'rb_relaxed_marginal': allocation_rb_relaxed_marginal,
            'random_sequential': allocation_random_sequential,
            'larger_order': allocation_larger_order,
        }
        try:
            return rule_map[allocation_rule]
        except KeyError as exc:
            supported = ', '.join(sorted(rule_map))
            raise ValueError(f"Unsupported allocation_rule={allocation_rule!r}. Supported rules: {supported}") from exc


    def reduce_inventory_state(self, inventory_array: np.array, target: int):
        assert target > 1
        current_length = len(inventory_array)
        if current_length <= target:
            return inventory_array
        piece_length = (current_length - 1) // (target - 1) + 1
        reduced_inventory_array = np.zeros(target)
        reduced_inventory_array[0] = inventory_array[0]
        for i in range(target - 2):
            reduced_inventory_array[i + 1] = sum(inventory_array[i * piece_length + 1: (i + 1) * piece_length + 1])
        reduced_inventory_array[-1] = sum(inventory_array[(target - 2) * piece_length + 1:])
        return reduced_inventory_array

    def get_state(self):
        if self.compress_lead_time:
            # 压缩状态规模
            state_rdc = np.hstack(
                [self.RDC_inventory[k] for k in self.RDC_inventory.keys()])
            state_cdc = np.hstack([self.reduce_inventory_state(self.CDC_inventory, self.compress_size)])
        else:
            state_rdc = np.hstack(
                [self.RDC_inventory[k] for k in self.RDC_inventory.keys()])
            state_cdc = self.CDC_inventory.copy()

        # normed_state_rdc = get_normalized_list(state_rdc)
        # normed_state_cdc = get_normalized_list(state_cdc)
        # state = np.hstack((normed_state_rdc, normed_state_cdc))
        state = np.hstack((state_rdc, state_cdc))
        if (
                getattr(self, "stationary_action_mask", None) is not None
                and getattr(self, "include_stationary_action_mask_in_state", False)):
            state = np.hstack((state, self.stationary_action_mask))
        return state

    def get_echelon_state(self):
        self.state_echelon_rdc = {}
        for k in self.RDC_inventory.keys():
            state_echelon_rdc_k = np.zeros(np.shape(self.RDC_inventory[k]))
            state_echelon_rdc_k[0] = self.RDC_inventory[k][0]
            for d in range(len(self.RDC_inventory[k]) - 1):
                state_echelon_rdc_k[d + 1] = state_echelon_rdc_k[d] + self.RDC_inventory[k][d + 1]
            self.state_echelon_rdc[k] = state_echelon_rdc_k
        state_rdc = np.hstack([self.state_echelon_rdc[k] for k in self.state_echelon_rdc.keys()])

        self.state_echelon_cdc = np.zeros(np.shape(self.CDC_inventory))
        self.state_echelon_cdc[0] = sum(
            [self.state_echelon_rdc[k][-1] for k in self.state_echelon_rdc.keys()]) + self.CDC_inventory[0]
        for d in range(len(self.state_echelon_cdc) - 1):
            self.state_echelon_cdc[d + 1] = self.state_echelon_cdc[d] + self.CDC_inventory[d + 1]

        state_cdc = self.state_echelon_cdc.copy()

        state = np.hstack((state_rdc, state_cdc))
        # self.state = state
        return state

    def reset(self):
        self.period = 0
        self.episode += 1

        self._initialize_state()
        self.intransit_transition()
        self.get_echelon_state()
        return self.get_state()


    def intransit_transition(self):
        # 入库
        for k in self.rdcs.keys():
            if len(self.RDC_inventory[k]) > 2:
                self.RDC_inventory[k] = np.hstack(
                    (self.RDC_inventory[k][0] + self.RDC_inventory[k][1], self.RDC_inventory[k][2:], [0]))
            else:
                self.RDC_inventory[k] = np.hstack(
                    (self.RDC_inventory[k][0] + self.RDC_inventory[k][1], [0]))
        if len(self.CDC_inventory) > 2:
            self.CDC_inventory = np.hstack(
                (self.CDC_inventory[0] + self.CDC_inventory[1], self.CDC_inventory[2:], [0]))
        else:
            self.CDC_inventory = np.hstack((self.CDC_inventory[0] + self.CDC_inventory[1], [0]))

    def sample_backordered_unmet_demand(self, unmet_demand):
        unmet_demand = max(float(unmet_demand), 0.0)
        if unmet_demand <= 1e-12:
            return 0.0, 0.0
        backordered = self.rho * unmet_demand
        lost = max(unmet_demand - backordered, 0.0)
        return backordered, lost

    def _sample_period_demand(self, k):
        if self.period < 1:
            return 0.0
        if self.if_generate_demand:
            return float(sample_demand(self.demand_feature[k], random_state=self.demand_rng))
        return float(self.all_demand_list[k][self.episode % self.num_of_instances, self.period % self.total_periods])

    def step_satisfy_demand(self):
        if self.is_kaynov_partial_backlogging:
            return self.step_satisfy_demand_kaynov_partial_backlogging()

        demand_info = {}
        for k in self.rdcs.keys():
            real_demand = self._sample_period_demand(k)
            previous_inventory = float(self.RDC_inventory[k][0])
            demand_to_satisfy = float(real_demand)
            satisfied_demand = min(max(previous_inventory, 0.0), demand_to_satisfy)
            unmet_demand = demand_to_satisfy - satisfied_demand
            backordered_demand, lost_demand = self.sample_backordered_unmet_demand(unmet_demand)
            inventory_on_hand = previous_inventory - satisfied_demand - backordered_demand
            self.RDC_inventory[k][0] = inventory_on_hand
            demand_info[k] = {
                "demand": demand_to_satisfy,
                "satisfied": satisfied_demand,
                "unmet": unmet_demand,
                "backordered": backordered_demand,
                "lost": lost_demand,
            }
        self.last_demand_info = demand_info
        return demand_info

    def step_satisfy_demand_kaynov_partial_backlogging(self):
        """Kaynov partial lost-sales/emergency-shipment demand transition.

        In this setting retailer inventory follows lost-sales dynamics: it never
        becomes negative and no historical backlog is stored. Unsatisfied demand
        may request an immediate emergency shipment from the remaining CDC
        on-hand inventory. Emergency shipments do not enter retailer inventory;
        they only reduce lost demand and consume CDC inventory.
        """
        demand_info = {}
        emergency_available = max(float(self.CDC_inventory[0]), 0.0)

        for k in self.rdcs.keys():
            demand_to_satisfy = self._sample_period_demand(k)
            previous_inventory = max(float(self.RDC_inventory[k][0]), 0.0)
            regular_satisfied = min(previous_inventory, demand_to_satisfy)
            shortage = max(demand_to_satisfy - previous_inventory, 0.0)

            emergency_requested = False
            emergency_shipment = 0.0
            if shortage > 1e-12:
                emergency_requested = self.emergency_rng.uniform() < self.emergency_shipment_prob
                if emergency_requested and emergency_available > 1e-12:
                    emergency_shipment = min(shortage, emergency_available)
                    emergency_available -= emergency_shipment

            lost_demand = max(shortage - emergency_shipment, 0.0)
            inventory_on_hand = max(previous_inventory - demand_to_satisfy, 0.0)
            self.RDC_inventory[k][0] = inventory_on_hand

            demand_info[k] = {
                "demand": demand_to_satisfy,
                "satisfied": regular_satisfied + emergency_shipment,
                "regular_satisfied": regular_satisfied,
                "unmet": shortage,
                "backordered": 0.0,
                "lost": lost_demand,
                "emergency_requested": emergency_requested,
                "emergency_shipment": emergency_shipment,
            }

        self.CDC_inventory[0] = max(emergency_available, 0.0)
        self.last_demand_info = demand_info
        self.last_emergency_shipments = {
            k: demand_info[k]["emergency_shipment"] for k in self.rdcs.keys()
        }
        return demand_info


    def step(self, action):
        cost = {'RHC': 0, 'BC': 0, 'LC': 0, 'WHC': 0, 'penalty': 0}
        inv_info, order = {}, {}
        reward = 0
        # analyze action
        self.analyze_action(action)
        if not self.is_kaynov_partial_backlogging:
            cost['WHC'] = self.holding_cost[0] * self.CDC_inventory[0]

        demand_info = self.step_satisfy_demand()
        if self.is_kaynov_partial_backlogging:
            cost['WHC'] = self.holding_cost[0] * self.CDC_inventory[0]
        for k in self.rdcs.keys():
            inventory_on_hand = self.RDC_inventory[k][0]
            if self.is_kaynov_partial_backlogging:
                cost['RHC'] += self.holding_cost[k] * max(inventory_on_hand, 0.0)
            elif inventory_on_hand < 0:
                cost['BC'] += -1 * self.backorder_cost[k] * inventory_on_hand
                # self.backorder_demand += min(-1 * inventory_on_hand, real_demand)
            else:
                cost['RHC'] += self.holding_cost[k] * inventory_on_hand
            cost['LC'] += self.backorder_cost[k] * demand_info[k]["lost"]

            inv_info[k] = inventory_on_hand
            order[k] = self.RDC_inventory[k][-1]
        inv_info[0] = self.CDC_inventory[0]
        order[0] = self.CDC_inventory[-1]
        # self.cost['holding_cost'] *= self.holding_weight
        total_cost = sum(cost.values())
        reward -= total_cost
        reward += self.cost_plus


        self.intransit_transition()
        # update echelon inventory
        self.get_echelon_state()

        self.period += 1
        done = self.period == self.total_periods
        self.state = self.get_state()
        
        return self.state, reward, done, (cost, inv_info, order)

    def step_feasible_action(self, feasible_action):
        """Execute already-reallocated order quantities.

        The recommended RA-SAC variant computes the relaxed action and its
        feasible reallocation inside the agent. This method lets the environment
        execute those feasible quantities directly, without applying another
        allocation rule.
        """
        cost = {'RHC': 0, 'BC': 0, 'LC': 0, 'WHC': 0, 'penalty': 0}
        inv_info, order = {}, {}
        reward = 0

        action_rdc = np.asarray(feasible_action[: self.rdc_size], dtype=float)
        action_cdc = float(feasible_action[-1])
        action_rdc = np.maximum(action_rdc, 0.0)
        action_cdc = max(action_cdc, 0.0)

        shipped_to_rdc = min(float(np.sum(action_rdc)), max(float(self.CDC_inventory[0]), 0.0))
        if np.sum(action_rdc) > shipped_to_rdc + 1e-8:
            scale = shipped_to_rdc / (np.sum(action_rdc) + 1e-8)
            action_rdc = action_rdc * scale

        self.target_inventory_position = {i: 0. for i in range(self.rdc_size + 1)}
        for n, k in enumerate(self.rdcs.keys()):
            self.target_inventory_position[k] = self.state_echelon_rdc[k][-1] + action_rdc[n]
            self.RDC_inventory[k][-1] = round(float(action_rdc[n]), 3)
        self.target_inventory_position[0] = self.state_echelon_cdc[-1] + action_cdc
        self.CDC_inventory[0] = max(float(self.CDC_inventory[0]) - float(np.sum(action_rdc)), 0.0)
        self.CDC_inventory[-1] = round(action_cdc, 3)

        if not self.is_kaynov_partial_backlogging:
            cost['WHC'] = self.holding_cost[0] * self.CDC_inventory[0]

        demand_info = self.step_satisfy_demand()
        if self.is_kaynov_partial_backlogging:
            cost['WHC'] = self.holding_cost[0] * self.CDC_inventory[0]
        for k in self.rdcs.keys():
            inventory_on_hand = self.RDC_inventory[k][0]
            if self.is_kaynov_partial_backlogging:
                cost['RHC'] += self.holding_cost[k] * max(inventory_on_hand, 0.0)
            elif inventory_on_hand < 0:
                cost['BC'] += -1 * self.backorder_cost[k] * inventory_on_hand
            else:
                cost['RHC'] += self.holding_cost[k] * inventory_on_hand
            cost['LC'] += self.backorder_cost[k] * demand_info[k]["lost"]

            inv_info[k] = inventory_on_hand
            order[k] = self.RDC_inventory[k][-1]
        inv_info[0] = self.CDC_inventory[0]
        order[0] = self.CDC_inventory[-1]

        total_cost = sum(cost.values())
        reward -= total_cost
        reward += self.cost_plus

        self.intransit_transition()
        self.get_echelon_state()

        self.period += 1
        done = self.period == self.total_periods
        self.state = self.get_state()

        return self.state, reward, done, (cost, inv_info, order)


    def _initialize_state(self):
        # base-stock max inventory position

        # RDC节点库存
        self.RDC_inventory = {}
        for k in self.rdcs.keys():
            item_inventory = {}
                # item_inventory[x] = np.zeros(self.rdcs[k].lead_time) # 3天，Array([4,2,2]), 4代表on-hand，2代表2天各自的in-transit
            item_inventory = self.initial_inventory_rdc[k].copy()
            item_inventory = [round(t, 3) for t in item_inventory]
            self.RDC_inventory[k] = item_inventory

        # CDC节点库存
        # self.CDC_inventory = {}
        self.CDC_inventory = self.initial_inventory_cdc.copy()
        self.CDC_inventory = [round(t, 3) for t in self.CDC_inventory]

        # 评价指标 0406
        # self.items_throughput = {}
        # self.items_on_hand = {}
        # self.items_on_hand_rdc = {}
        # self.items_on_hand_cdc = {}
        # for x in self.items_dict.keys():
        #     self.items_throughput[x] = 0
        #     self.items_on_hand[x] = 0
        #     self.items_on_hand_cdc[x] = 0

        # for k in self.rdcs.keys():
        #     items_on_hand_rdc_k = {}
        #     for x in self.selling_item:
        #         items_on_hand_rdc_k[x] = 0
        #     self.items_on_hand_rdc[k] = items_on_hand_rdc_k

        # self.total_demand = {}
        # self.backorder_demand = {}
        # for x in self.selling_item:
        #     self.total_demand[x] = 0
        #     self.backorder_demand[x] = 0
        # self.total_cost = 0
        # self.holding_cost = 0

    # 计算各库存结点最大需求下的的理论最大/最小级库存水位
    def _initialize_max_min_stock_level(self):
        self.RDC_max_bsl = {}
        self.RDC_min_bsl = {}
        self.RDC_max_bsl_one_period = {}
        for k in self.rdcs.keys():
            rdc_k_max_bsl = {}
            rdc_k_min_bsl = {}
            if self.demand_type == 'quantile':  # 分位数, 周需求, 每周变化
                max_demand, min_demand = 0., 0.
                if self.demand_feature[k]:
                    max_demand = sum([self.demand_feature[k]['0.9'][w] for w in range(13)]) / 13 / 7
                    min_demand = sum([self.demand_feature[k]['0.1'][w] for w in range(13)]) / 13 / 7
            else:
                max_demand_one_period = demand_quantile(self.demand_feature[k], 0.99)
                max_demand = lead_time_demand_quantile(self.demand_feature[k], 0.99, self.rdcs[k].lead_time + 1)
                min_demand = lead_time_demand_quantile(self.demand_feature[k], 0.01, self.rdcs[k].lead_time + 1)
            self.RDC_max_bsl_one_period[k] = max_demand if self.demand_type == 'quantile' else max_demand_one_period
            rdc_k_max_bsl = (self.rdcs[k].lead_time + 1) * max_demand if self.demand_type == 'quantile' else max_demand
            rdc_k_min_bsl = (self.rdcs[k].lead_time + 1) * min_demand if self.demand_type == 'quantile' else min_demand
            self.RDC_max_bsl[k] = rdc_k_max_bsl
            self.RDC_min_bsl[k] = rdc_k_min_bsl

        # self.CDC_max_bsl = {}
        # self.CDC_min_bsl = {}
        self.CDC_max_bsl = sum([self.RDC_max_bsl[k] for k in self.rdcs.keys()])
        self.CDC_min_bsl = sum([self.RDC_min_bsl[k] for k in self.rdcs.keys()])
        self.CDC_max_bsl_one_period = sum(self.RDC_max_bsl_one_period.values())
        if self.demand_type == 'quantile':
            max_demand, min_demand = 0., 0.
            for k in self.rdcs.keys():
                if self.demand_feature[k]:
                    max_demand += sum([self.demand_feature[k]['0.9'][w] for w in range(13)]) / 13 / 7
                    min_demand += sum([self.demand_feature[k]['0.1'][w] for w in range(13)]) / 13 / 7
        else:
            max_demand = sum([
                lead_time_demand_quantile(self.demand_feature[k], 0.99, self.item_lead_time + 1)
                for k in self.rdcs.keys()
            ])
            min_demand = sum([
                lead_time_demand_quantile(self.demand_feature[k], 0.01, self.item_lead_time + 1)
                for k in self.rdcs.keys()
            ])
            self.CDC_max_bsl += max_demand
            self.CDC_min_bsl += min_demand

        self.base_stock_level_proportion = 1.0
        for k in self.rdcs.keys():
            self.RDC_max_bsl[k] *= self.base_stock_level_proportion
        self.CDC_max_bsl *= self.base_stock_level_proportion

        if self.rdc_bsl_max_override is not None:
            if isinstance(self.rdc_bsl_max_override, dict):
                for k in self.rdcs.keys():
                    if k in self.rdc_bsl_max_override:
                        self.RDC_max_bsl[k] = max(
                            self.RDC_min_bsl[k],
                            float(self.rdc_bsl_max_override[k]),
                        )
            else:
                rdc_override = float(self.rdc_bsl_max_override)
                for k in self.rdcs.keys():
                    self.RDC_max_bsl[k] = max(self.RDC_min_bsl[k], rdc_override)

        if (
                self.cdc_bsl_min_override is not None
                and self.cdc_bsl_max_override is not None):
            explicit_lower = max(0.0, float(self.cdc_bsl_min_override))
            explicit_upper = max(explicit_lower, float(self.cdc_bsl_max_override))
            self.CDC_min_bsl = explicit_lower
            self.CDC_max_bsl = explicit_upper
        elif self.cdc_bsl_max_override is not None:
            self.CDC_max_bsl = max(self.CDC_min_bsl, float(self.cdc_bsl_max_override))

        if self.rdc_bsl_min_override is not None:
            if isinstance(self.rdc_bsl_min_override, dict):
                for k in self.rdcs.keys():
                    if k in self.rdc_bsl_min_override:
                        self.RDC_min_bsl[k] = min(
                            self.RDC_max_bsl[k],
                            max(0.0, float(self.rdc_bsl_min_override[k])),
                        )
            else:
                rdc_min_override = float(self.rdc_bsl_min_override)
                for k in self.rdcs.keys():
                    self.RDC_min_bsl[k] = min(self.RDC_max_bsl[k], max(0.0, rdc_min_override))

        if self.cdc_bsl_min_override is not None and self.cdc_bsl_max_override is None:
            self.CDC_min_bsl = min(self.CDC_max_bsl, max(0.0, float(self.cdc_bsl_min_override)))


    def _initialize_weights(self):
        self.items_weight = {}
        for x in self.CDC_max_bsl.keys():
            self.items_weight[x] = 0.
            if self.CDC_max_bsl[x] > 0:
                self.items_weight[x] = 1 / self.CDC_max_bsl[x]
        weight_sum = sum(self.items_weight.values())
        for x in self.items_weight.keys():
            self.items_weight[x] /= (weight_sum + 1)  # 防止除0

        if self.holding_cost_revise:
            for x in self.items_dict.keys():
                self.items_dict[x].holding_cost *= self.items_weight[x] * 10


    def nn_output_to_action(self, action):
        self.target_inventory_position = {i: 0. for i in range(self.rdc_size + 1)}
        
        # [-1,1]映射至真实决策量
        action = (action + 1) / 2
        action_rdc = action[: self.rdc_size].copy()
        action_cdc = action[-1]

        for n, k in enumerate(self.rdcs.keys()):
            self.target_inventory_position[k] = (1 - action_rdc[n]) * self.RDC_min_bsl[k] + action_rdc[n] * \
                                            self.RDC_max_bsl[k]
        self.target_inventory_position[0] = (1 - action_cdc) * self.CDC_min_bsl + action_cdc * self.CDC_max_bsl
        return self.target_inventory_position


    def analyze_action(self, action):
        if self.method == 'heu':
            action_rdc = action[: self.rdc_size].copy()
            for n, k in enumerate(self.rdcs.keys()):
                self.RDC_inventory[k][-1] = action_rdc[n]
        elif self.method == 'paper':   # action is order quantity
            action_rdc = action[: -1]
            for n, k in enumerate(self.rdcs.keys()):
                self.RDC_inventory[k][-1] = action_rdc[n]
        else:
            self.nn_output_to_action(action)
            # action = (action + 1) / 2
            # action_rdc = action[: self.rdc_size].copy()
            
            for k in self.rdcs.keys():
                # target_inventory_position = (1 - action_rdc[n]) * self.RDC_min_bsl[k] + action_rdc[n] * \
                #                             self.RDC_max_bsl[k]
                # logger.info(f'算法建议的RDC{k}的订货水位为{target_inventory_position}(action_rdc为{action_rdc[n]})')
                # self.state_info['%s_%s_base_stock_level' % (x, k)] = (round(item_action[i][n], 2), round(target_inventory_position, 2))
                current_inventory_position = self.state_echelon_rdc[k][-1]
                self.RDC_inventory[k][-1] = round(max(0, self.target_inventory_position[k] - current_inventory_position), 0)
                # self.target_inventory_position[k] = target_inventory_position

        # allocation
        # x_discount = 1.0
        cdc_inventory_on_hand = round(self.CDC_inventory[0], 0)
        x_ordering = [self.RDC_inventory[k][-1] for k in self.rdcs.keys()]
        rdc_inventory_position = {k: self.state_echelon_rdc[k][-1] for k in self.rdcs.keys()}
        if self.allocation_rule in {
            allocation_prioritized_greedy,
            allocation_prioritized_greedy_coarse_legacy,
            allocation_prioritized_greedy_inequality,
            allocation_rb_relaxed_marginal,
        }:
            demand_para = {
                k: (self.demand_feature[k], self.rdcs[k].lead_time + 1)
                for k in self.rdcs.keys()
            }
            on_hand, new_allocate = self.allocation_rule(
                cdc_inventory_on_hand,
                x_ordering,
                demand_para,
                rdc_inventory_position,
                self.holding_cost,
                self.backorder_cost,
            )
        elif self.allocation_rule in {
            allocation_prioritized_greedy_fast,
            allocation_prioritized_greedy_approx,
        }:
            demand_para = {
                k: demand_mean(self.demand_feature[k]) * (self.rdcs[k].lead_time + 1)
                for k in self.rdcs.keys()
            }
            on_hand, new_allocate = self.allocation_rule(
                cdc_inventory_on_hand,
                x_ordering,
                demand_para,
                rdc_inventory_position,
                self.holding_cost,
                self.backorder_cost,
            )
        elif self.allocation_rule is allocation_random_sequential:
            on_hand, new_allocate = self.allocation_rule(
                cdc_inventory_on_hand,
                x_ordering,
                rng=self.allocation_rng,
            )
        else:
            on_hand, new_allocate = self.allocation_rule(cdc_inventory_on_hand, x_ordering)
        requested_total = float(np.sum(x_ordering))
        allocated_total = float(np.sum(new_allocate))
        self.last_allocation_info = {
            "requested": np.asarray(x_ordering, dtype=float).copy(),
            "allocated": np.asarray(new_allocate, dtype=float).copy(),
            "requested_total": requested_total,
            "allocated_total": allocated_total,
            "cdc_available_before": float(cdc_inventory_on_hand),
            "cdc_available_after": float(on_hand),
            "inventory_binding": bool(requested_total > cdc_inventory_on_hand + 1e-9),
            "unfilled_request": max(0.0, requested_total - allocated_total),
            "retained_with_unfilled_request": (
                max(0.0, float(on_hand))
                if requested_total > allocated_total + 1e-9 else 0.0
            ),
        }
        self.CDC_inventory[0] = on_hand
        for n, x in enumerate(new_allocate):
            self.RDC_inventory[n+1][-1] = x

        # warehouse
        action_cdc = action[-1]
        if self.method == 'heu' or self.method == 'paper':
            policy_order_quantity = action_cdc
        else:
            # target_inventory_position = (1 - action_cdc) * self.CDC_min_bsl + action_cdc * self.CDC_max_bsl
            # logger.info(f'算法建议的CDC的订货水位为{target_inventory_position}(action_cdc为{action_cdc})')
            # self.state_info['%s_cdc_base_stock_level' % v.name] = round(target_inventory_position, 2)
            current_inventory_position = self.state_echelon_cdc[-1]
            policy_order_quantity = round(max(0, self.target_inventory_position[0] - current_inventory_position), 0)
            # self.target_inventory_position[0] = target_inventory_position
        order_quantity = policy_order_quantity
        # product
        self.CDC_inventory[-1] = round(order_quantity, 3)

    def generate_action(self):
        return rd.rand(self.action_dim) * 2 - 1


class FiniteHorizonNonstationaryOWMREnv(OWMRInventoryEnv):
    """Finite-horizon OWMR environment with nonstationary quantile forecasts.

    Demand specs are dicts with type="quantile_series" and weekly 0.1..0.9
    quantile arrays. The original stationary OWMRInventoryEnv is left
    unchanged; this subclass only overrides demand sampling, action masking,
    period-dependent allocation cost, and finite-horizon training cost.
    """

    def __init__(
            self,
            all_demand_list: dict = None,
            info_input: OWMR_input = None,
            horizon: int = 13,
            include_period_in_state: bool = True,
            include_action_mask_in_state: bool = True,
            train_cost_mode: str = "controllable",
            cdc_effective_downstream_lead_time: str = "min",
            action_bound_profile: str = "joint_quantile_v2",
            action_lower_quantile: float = None,
            action_upper_quantile: float = None,
            **kwargs,
    ) -> None:
        if info_input is None:
            raise ValueError("info_input is required for FiniteHorizonNonstationaryOWMREnv.")
        self.finite_horizon = int(horizon)
        if self.finite_horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}.")
        self.include_period_in_state = bool(include_period_in_state)
        self.include_action_mask_in_state = bool(include_action_mask_in_state)
        self.train_cost_mode = str(train_cost_mode).lower()
        if self.train_cost_mode not in {"controllable", "full"}:
            raise ValueError(
                f"train_cost_mode must be 'controllable' or 'full', got {train_cost_mode!r}."
            )
        self.cdc_effective_downstream_lead_time = str(cdc_effective_downstream_lead_time).lower()
        if self.cdc_effective_downstream_lead_time not in {"min", "max"}:
            raise ValueError(
                "cdc_effective_downstream_lead_time must be 'min' or 'max', "
                f"got {cdc_effective_downstream_lead_time!r}."
            )
        self.action_bound_profile = str(action_bound_profile).lower()
        if self.action_bound_profile not in {"legacy_v1", "joint_quantile_v2"}:
            raise ValueError(
                "action_bound_profile must be 'legacy_v1' or 'joint_quantile_v2', "
                f"got {action_bound_profile!r}."
            )
        default_lower = 0.1 if self.action_bound_profile == "legacy_v1" else 0.5
        self.action_lower_quantile = float(np.clip(
            default_lower if action_lower_quantile is None else action_lower_quantile,
            0.0,
            1.0,
        ))
        if action_upper_quantile is None:
            self.action_upper_quantile = (
                None if self.action_bound_profile == "legacy_v1" else 0.99
            )
        else:
            self.action_upper_quantile = float(np.clip(
                action_upper_quantile,
                self.action_lower_quantile,
                1.0,
            ))
        self.nonstationary_rng = np.random.default_rng(kwargs.get("seed", None))
        for k in info_input.rdc_dict.keys():
            spec = info_input.demand_feature[k]
            quantiles = spec.get("quantiles", spec)
            lengths = [len(quantiles.get(f"{q / 10:.1f}", [])) for q in range(1, 10)]
            if any(length < self.finite_horizon for length in lengths):
                raise ValueError(
                    f"RDC {k} quantile forecast must contain at least {self.finite_horizon} periods; "
                    f"got lengths={lengths}."
                )
        if all_demand_list is None:
            all_demand_list = {
                k: np.zeros((1, self.finite_horizon), dtype=float)
                for k in info_input.rdc_dict.keys()
            }
        else:
            for k in info_input.rdc_dict.keys():
                demand_array = np.asarray(all_demand_list[k])
                if demand_array.ndim != 2 or demand_array.shape[1] < self.finite_horizon:
                    raise ValueError(
                        f"all_demand_list[{k}] must have shape (episodes, >= {self.finite_horizon}); "
                        f"got {demand_array.shape}."
                    )
        base_kwargs = kwargs_filter(OWMRInventoryEnv.__init__, kwargs)
        super().__init__(
            all_demand_list=all_demand_list,
            info_input=info_input,
            demand_type="nonstationary_quantile",
            **base_kwargs,
        )
        self.total_periods = self.finite_horizon
        self.max_step = self.finite_horizon
        base_state_dim = len(super().get_state())
        self.action_mask_state_start = base_state_dim + int(self.include_period_in_state)
        self.action_mask_dim = self.rdc_size + 1
        self.state_dim = len(self.get_state())
        self.state = self.get_state()

    def get_state(self):
        state = super().get_state()
        parts = [state]
        if getattr(self, "include_period_in_state", False):
            horizon = max(float(getattr(self, "finite_horizon", 1)), 1.0)
            parts.append(np.asarray([float(getattr(self, "period", 0)) / horizon], dtype=float))
        if getattr(self, "include_action_mask_in_state", False):
            parts.append(self.get_action_mask())
        return np.hstack(parts)

    def get_action_mask(self, period=None):
        """Return one validity flag per RDC action plus the CDC action."""
        if period is None:
            period = int(getattr(self, "period", 0))
        mask = [float(self._is_rdc_order_effective(k, period=period)) for k in self.rdcs.keys()]
        mask.append(float(self._is_cdc_order_effective(period=period)))
        return np.asarray(mask, dtype=float)

    def _effective_downstream_lead_time(self):
        lead_times = [int(r.lead_time) for r in self.rdcs.values()]
        if not lead_times:
            return 0
        if self.cdc_effective_downstream_lead_time == "max":
            return max(lead_times)
        return min(lead_times)

    def _is_rdc_order_effective(self, k, period=None):
        period = self.period if period is None else int(period)
        total_periods = int(getattr(self, "total_periods", self.finite_horizon))
        return period < total_periods - int(self.rdcs[k].lead_time)

    def _is_cdc_order_effective(self, period=None):
        period = self.period if period is None else int(period)
        total_lead = int(self.item_lead_time) + self._effective_downstream_lead_time()
        total_periods = int(getattr(self, "total_periods", self.finite_horizon))
        return period < total_periods - total_lead

    def _is_rdc_cost_train_period(self, k):
        if self.train_cost_mode == "full":
            return True
        return self.period >= int(self.rdcs[k].lead_time)

    def _is_cdc_holding_train_period(self):
        if self.train_cost_mode == "full":
            return True
        return self.period < self.total_periods - self._effective_downstream_lead_time()

    @staticmethod
    def _pmf_quantile(support, pmf, q):
        q = float(np.clip(q, 0.0, 1.0))
        idx = int(np.searchsorted(np.cumsum(pmf), q, side="left"))
        idx = min(max(idx, 0), len(support) - 1)
        return float(support[idx])

    def _period_lead_time_pmf_model(self, k, start_period, periods):
        support, pmf = quantile_series_lead_time_pmf(
            self.demand_feature[k],
            int(start_period),
            int(periods),
        )
        return "pmf", tuple(support.tolist()), tuple(pmf.tolist())

    def _period_lead_time_quantile(self, k, start_period, periods, q):
        support, pmf = quantile_series_lead_time_pmf(
            self.demand_feature[k],
            int(start_period),
            int(periods),
        )
        return self._pmf_quantile(support, pmf, q)

    def _aggregate_period_quantile(self, start_period, periods, q):
        windows = [
            (self.demand_feature[k], int(start_period), int(periods))
            for k in self.rdcs.keys()
        ]
        support, pmf = quantile_series_sum_window_pmf(windows)
        return self._pmf_quantile(support, pmf, q)

    def _cost_aware_action_quantiles(self, k):
        lower = self.action_lower_quantile
        if self.action_upper_quantile is not None:
            return lower, self.action_upper_quantile
        warehouse_holding = float(self.holding_cost.get(0, 0.0))
        rdc_holding = float(self.holding_cost.get(k, warehouse_holding))
        echelon_holding = rdc_holding - warehouse_holding
        effective_holding = echelon_holding if echelon_holding > 1e-8 else max(rdc_holding, 1e-8)
        shortage = max(float(self.backorder_cost.get(k, 0.0)), 0.0)
        critical_fractile = shortage / (shortage + effective_holding)
        return lower, float(np.clip(critical_fractile, max(lower, 0.5), 0.9))

    def _critical_fractile(self, k):
        warehouse_holding = float(self.holding_cost.get(0, 0.0))
        rdc_holding = float(self.holding_cost.get(k, warehouse_holding))
        echelon_holding = rdc_holding - warehouse_holding
        effective_holding = (
            echelon_holding if echelon_holding > 1e-8 else max(rdc_holding, 1e-8)
        )
        shortage = max(float(self.backorder_cost.get(k, 0.0)), 0.0)
        return shortage / (shortage + effective_holding)

    def _joint_cdc_action_bounds(self, period, lower_quantile, upper_quantile):
        windows = [
            (
                self.demand_feature[k],
                int(period),
                int(self.item_lead_time) + int(self.rdcs[k].lead_time) + 1,
            )
            for k in self.rdcs.keys()
        ]
        support, pmf = quantile_series_sum_window_pmf(windows)
        lower = self._pmf_quantile(support, pmf, lower_quantile)
        upper = self._pmf_quantile(support, pmf, upper_quantile)
        return max(0.0, lower), max(lower, upper)

    def _initialize_max_min_stock_level(self):
        horizon = int(getattr(self, "finite_horizon", 13))
        self.RDC_bsl_bounds_by_period = {k: {} for k in self.rdcs.keys()}
        self.CDC_bsl_bounds_by_period = {}
        self.RDC_max_bsl_one_period = {}
        self.RDC_critical_fractile = {
            k: self._critical_fractile(k) for k in self.rdcs.keys()
        }

        for k in self.rdcs.keys():
            lower_q, upper_q = self._cost_aware_action_quantiles(k)
            lead_periods = int(self.rdcs[k].lead_time) + 1
            self.RDC_max_bsl_one_period[k] = max(
                self._period_lead_time_quantile(k, t, 1, upper_q)
                for t in range(horizon)
            )
            for t in range(horizon):
                if t >= horizon - int(self.rdcs[k].lead_time):
                    bounds = (0.0, 0.0)
                else:
                    lower = self._period_lead_time_quantile(k, t, lead_periods, lower_q)
                    upper = self._period_lead_time_quantile(k, t, lead_periods, upper_q)
                    bounds = (max(0.0, lower), max(lower, upper))
                self.RDC_bsl_bounds_by_period[k][t] = bounds

        cdc_lower_q = self.action_lower_quantile
        if self.action_upper_quantile is not None:
            cdc_upper_q = self.action_upper_quantile
        else:
            cdc_upper_q = max(self._cost_aware_action_quantiles(k)[1] for k in self.rdcs.keys())
        upstream_periods = int(self.item_lead_time) + 1
        for t in range(horizon):
            if t >= horizon - (int(self.item_lead_time) + self._effective_downstream_lead_time()):
                self.CDC_bsl_bounds_by_period[t] = (0.0, 0.0)
                continue
            if self.action_bound_profile == "joint_quantile_v2":
                self.CDC_bsl_bounds_by_period[t] = self._joint_cdc_action_bounds(
                    t,
                    cdc_lower_q,
                    cdc_upper_q,
                )
            else:
                rdc_lower = sum(
                    self.RDC_bsl_bounds_by_period[k][t][0] for k in self.rdcs.keys()
                )
                rdc_upper = sum(
                    self.RDC_bsl_bounds_by_period[k][t][1] for k in self.rdcs.keys()
                )
                upstream_lower = self._aggregate_period_quantile(
                    t, upstream_periods, cdc_lower_q
                )
                upstream_upper = self._aggregate_period_quantile(
                    t, upstream_periods, cdc_upper_q
                )
                self.CDC_bsl_bounds_by_period[t] = (
                    max(0.0, rdc_lower + upstream_lower),
                    max(rdc_lower + upstream_lower, rdc_upper + upstream_upper),
                )

        def override_for(value, k):
            if value is None:
                return None
            if isinstance(value, dict):
                return value.get(k)
            return value

        for k in self.rdcs.keys():
            min_override = override_for(self.rdc_bsl_min_override, k)
            max_override = override_for(self.rdc_bsl_max_override, k)
            for t, (lower, upper) in self.RDC_bsl_bounds_by_period[k].items():
                if lower == upper == 0.0:
                    continue
                if max_override is not None:
                    upper = max(lower, float(max_override))
                if min_override is not None:
                    lower = min(upper, max(0.0, float(min_override)))
                self.RDC_bsl_bounds_by_period[k][t] = (lower, upper)

        for t, (lower, upper) in self.CDC_bsl_bounds_by_period.items():
            if lower == upper == 0.0:
                continue
            if self.cdc_bsl_max_override is not None:
                upper = max(lower, float(self.cdc_bsl_max_override))
            if self.cdc_bsl_min_override is not None:
                lower = min(upper, max(0.0, float(self.cdc_bsl_min_override)))
            self.CDC_bsl_bounds_by_period[t] = (lower, upper)

        self.RDC_min_bsl = {}
        self.RDC_max_bsl = {}
        for k in self.rdcs.keys():
            active = [bounds for bounds in self.RDC_bsl_bounds_by_period[k].values() if bounds != (0.0, 0.0)]
            self.RDC_min_bsl[k] = min(bounds[0] for bounds in active)
            self.RDC_max_bsl[k] = max(bounds[1] for bounds in active)
        active_cdc = [bounds for bounds in self.CDC_bsl_bounds_by_period.values() if bounds != (0.0, 0.0)]
        self.CDC_min_bsl = min(bounds[0] for bounds in active_cdc)
        self.CDC_max_bsl = max(bounds[1] for bounds in active_cdc)
        self.CDC_max_bsl_one_period = max(
            self._aggregate_period_quantile(t, 1, cdc_upper_q) for t in range(horizon)
        )

    def nn_output_to_action(self, action):
        """Map normalized actions through the current week's BSL bounds."""
        self.target_inventory_position = {i: 0.0 for i in [0, *self.rdcs.keys()]}
        normalized = (np.asarray(action, dtype=float) + 1.0) / 2.0
        t = min(max(int(self.period), 0), self.total_periods - 1)
        for n, k in enumerate(self.rdcs.keys()):
            lower, upper = self.RDC_bsl_bounds_by_period[k][t]
            self.target_inventory_position[k] = lower + normalized[n] * (upper - lower)
        lower, upper = self.CDC_bsl_bounds_by_period[t]
        self.target_inventory_position[0] = lower + normalized[-1] * (upper - lower)
        return self.target_inventory_position

    def _sample_period_demand(self, k):
        if self.period >= self.total_periods:
            return 0.0
        if self.if_generate_demand:
            return float(sample_quantile_series_demand(
                self.demand_feature[k],
                self.period,
                random_state=self.nonstationary_rng,
            ))
        idx = (self.episode - 1) % self.num_of_instances
        return float(self.all_demand_list[k][idx, self.period])

    def analyze_action(self, action):
        if self.method == 'heu':
            action_rdc = action[: self.rdc_size].copy()
            for n, k in enumerate(self.rdcs.keys()):
                self.RDC_inventory[k][-1] = action_rdc[n] if self._is_rdc_order_effective(k) else 0.0
        elif self.method == 'paper':
            action_rdc = action[: -1]
            for n, k in enumerate(self.rdcs.keys()):
                self.RDC_inventory[k][-1] = action_rdc[n] if self._is_rdc_order_effective(k) else 0.0
        else:
            self.nn_output_to_action(action)
            for k in self.rdcs.keys():
                current_inventory_position = self.state_echelon_rdc[k][-1]
                order = round(max(0, self.target_inventory_position[k] - current_inventory_position), 0)
                self.RDC_inventory[k][-1] = order if self._is_rdc_order_effective(k) else 0.0

        cdc_inventory_on_hand = round(self.CDC_inventory[0], 0)
        x_ordering = [self.RDC_inventory[k][-1] for k in self.rdcs.keys()]
        rdc_inventory_position = {k: self.state_echelon_rdc[k][-1] for k in self.rdcs.keys()}
        if self.allocation_rule in {
            allocation_prioritized_greedy,
            allocation_prioritized_greedy_coarse_legacy,
            allocation_prioritized_greedy_inequality,
            allocation_rb_relaxed_marginal,
        }:
            demand_para = {
                k: self._period_lead_time_pmf_model(k, self.period, self.rdcs[k].lead_time + 1)
                for k in self.rdcs.keys()
            }
            on_hand, new_allocate = self.allocation_rule(
                cdc_inventory_on_hand,
                x_ordering,
                demand_para,
                rdc_inventory_position,
                self.holding_cost,
                self.backorder_cost,
            )
        elif self.allocation_rule in {
            allocation_prioritized_greedy_fast,
            allocation_prioritized_greedy_approx,
        }:
            demand_para = {}
            for k in self.rdcs.keys():
                support, pmf = quantile_series_lead_time_pmf(
                    self.demand_feature[k],
                    self.period,
                    self.rdcs[k].lead_time + 1,
                )
                demand_para[k] = float(np.dot(support, pmf))
            on_hand, new_allocate = self.allocation_rule(
                cdc_inventory_on_hand,
                x_ordering,
                demand_para,
                rdc_inventory_position,
                self.holding_cost,
                self.backorder_cost,
            )
        elif self.allocation_rule is allocation_random_sequential:
            on_hand, new_allocate = self.allocation_rule(
                cdc_inventory_on_hand,
                x_ordering,
                rng=self.allocation_rng,
            )
        else:
            on_hand, new_allocate = self.allocation_rule(cdc_inventory_on_hand, x_ordering)
        requested_total = float(np.sum(x_ordering))
        allocated_total = float(np.sum(new_allocate))
        self.last_allocation_info = {
            "requested": np.asarray(x_ordering, dtype=float).copy(),
            "allocated": np.asarray(new_allocate, dtype=float).copy(),
            "requested_total": requested_total,
            "allocated_total": allocated_total,
            "cdc_available_before": float(cdc_inventory_on_hand),
            "cdc_available_after": float(on_hand),
            "inventory_binding": bool(requested_total > cdc_inventory_on_hand + 1e-9),
            "unfilled_request": max(0.0, requested_total - allocated_total),
            "retained_with_unfilled_request": (
                max(0.0, float(on_hand))
                if requested_total > allocated_total + 1e-9 else 0.0
            ),
        }
        self.CDC_inventory[0] = on_hand
        for k, x in zip(self.rdcs.keys(), new_allocate):
            self.RDC_inventory[k][-1] = x

        action_cdc = action[-1]
        if self.method == 'heu' or self.method == 'paper':
            policy_order_quantity = action_cdc
        else:
            current_inventory_position = self.state_echelon_cdc[-1]
            policy_order_quantity = round(max(0, self.target_inventory_position[0] - current_inventory_position), 0)
        if not self._is_cdc_order_effective():
            policy_order_quantity = 0.0
        self.CDC_inventory[-1] = round(policy_order_quantity, 3)

    def _step_after_orders(self):
        full_cost = {'RHC': 0, 'BC': 0, 'LC': 0, 'WHC': 0, 'penalty': 0}
        train_cost = {'RHC': 0, 'BC': 0, 'LC': 0, 'WHC': 0, 'penalty': 0}
        inv_info, order = {}, {}

        whc = self.holding_cost[0] * self.CDC_inventory[0]
        if not self.is_kaynov_partial_backlogging:
            full_cost['WHC'] = whc
            if self._is_cdc_holding_train_period():
                train_cost['WHC'] = whc

        demand_info = self.step_satisfy_demand()
        if self.is_kaynov_partial_backlogging:
            whc = self.holding_cost[0] * self.CDC_inventory[0]
            full_cost['WHC'] = whc
            if self._is_cdc_holding_train_period():
                train_cost['WHC'] = whc

        for k in self.rdcs.keys():
            inventory_on_hand = self.RDC_inventory[k][0]
            rdc_cost = {'RHC': 0.0, 'BC': 0.0, 'LC': 0.0}
            if self.is_kaynov_partial_backlogging:
                rdc_cost['RHC'] = self.holding_cost[k] * max(inventory_on_hand, 0.0)
            elif inventory_on_hand < 0:
                rdc_cost['BC'] = -1 * self.backorder_cost[k] * inventory_on_hand
            else:
                rdc_cost['RHC'] = self.holding_cost[k] * inventory_on_hand
            rdc_cost['LC'] = self.backorder_cost[k] * demand_info[k]["lost"]
            for key, value in rdc_cost.items():
                full_cost[key] += value
                if self._is_rdc_cost_train_period(k):
                    train_cost[key] += value

            inv_info[k] = inventory_on_hand
            order[k] = self.RDC_inventory[k][-1]
        inv_info[0] = self.CDC_inventory[0]
        order[0] = self.CDC_inventory[-1]

        self.last_cost_breakdown = {
            "full_cost": full_cost.copy(),
            "train_cost": train_cost.copy(),
            "fixed_or_masked_cost": {
                key: full_cost[key] - train_cost[key] for key in full_cost.keys()
            },
            "demand_info": demand_info,
        }
        reward = -sum(train_cost.values()) + self.cost_plus

        self.intransit_transition()
        self.get_echelon_state()
        self.period += 1
        done = self.period == self.total_periods
        self.state = self.get_state()
        return self.state, reward, done, (train_cost, inv_info, order)

    def step(self, action):
        self.analyze_action(action)
        return self._step_after_orders()

    def step_feasible_action(self, feasible_action):
        """Execute feasible quantities under the same finite-horizon contract."""
        feasible_action = np.asarray(feasible_action, dtype=float)
        if feasible_action.shape != (self.rdc_size + 1,):
            raise ValueError(
                f"feasible_action must have shape {(self.rdc_size + 1,)}, got {feasible_action.shape}."
            )
        mask = self.get_action_mask()
        action_rdc = np.maximum(feasible_action[:self.rdc_size], 0.0) * mask[:self.rdc_size]
        action_cdc = max(float(feasible_action[-1]), 0.0) * mask[-1]

        available = max(float(self.CDC_inventory[0]), 0.0)
        requested = float(np.sum(action_rdc))
        if requested > available + 1e-8:
            action_rdc *= available / requested

        self.target_inventory_position = {i: 0.0 for i in [0, *self.rdcs.keys()]}
        for n, k in enumerate(self.rdcs.keys()):
            quantity = round(float(action_rdc[n]), 3)
            self.target_inventory_position[k] = self.state_echelon_rdc[k][-1] + quantity
            self.RDC_inventory[k][-1] = quantity
        self.CDC_inventory[0] = max(available - float(np.sum(action_rdc)), 0.0)
        self.target_inventory_position[0] = self.state_echelon_cdc[-1] + action_cdc
        self.CDC_inventory[-1] = round(action_cdc, 3)
        return self._step_after_orders()


def deterministic_largest_remainder_allocation(continuous_allocation, capacity):
    """Convert a nonnegative allocation into feasible deterministic integers."""
    quantities = np.maximum(np.asarray(continuous_allocation, dtype=float), 0.0)
    integer_capacity = max(int(math.floor(max(float(capacity), 0.0) + 1e-9)), 0)
    if quantities.size == 0 or integer_capacity == 0 or quantities.sum() <= 0:
        return [0.0] * int(quantities.size)

    target_total = min(
        integer_capacity,
        max(int(math.floor(float(quantities.sum()) + 0.5)), 0),
    )
    floors = np.floor(quantities).astype(np.int64)
    if int(floors.sum()) > target_total:
        quantities = quantities * (target_total / max(float(quantities.sum()), 1e-12))
        floors = np.floor(quantities).astype(np.int64)

    residual = target_total - int(floors.sum())
    if residual > 0:
        fractions = quantities - floors
        order = sorted(range(len(fractions)), key=lambda idx: (-fractions[idx], idx))
        for idx in order[:residual]:
            floors[idx] += 1
    return floors.astype(float).tolist()


def allocation_proportion(on_hand, to_allocate):
    requested = np.maximum(np.asarray(to_allocate, dtype=float), 0.0)
    if requested.sum() > max(float(on_hand), 0.0) + 1e-9:
        if requested.sum() > 0:
            proportional = requested * max(float(on_hand), 0.0) / requested.sum()
        else:
            proportional = requested
        new_allocate = deterministic_largest_remainder_allocation(proportional, on_hand)
    else:
        new_allocate = requested.tolist()

    used = float(np.sum(new_allocate))
    return max(float(on_hand) - used, 0.0), new_allocate


def allocation_larger_order(on_hand, to_allocate):
    new_allocate = [0. for x in to_allocate]
    x_usage = sum(to_allocate)
    if x_usage > on_hand:
        original_order = deepcopy(to_allocate)
        max_unmet = new_max_unmet = max(original_order)
        original_order.remove(max_unmet)
        while True:
            max_unmet = new_max_unmet
            new_max_unmet = 0
            if original_order:
                new_max_unmet = max(original_order)
                original_order.remove(new_max_unmet)
            if new_max_unmet == max_unmet:
                continue
            new_allocate = [max(0, x - new_max_unmet) for x in to_allocate]
            exceeds = sum(new_allocate) - on_hand
            if exceeds >= 0:
                break
        if exceeds > 0:
            exceed_count = sum([1 for x in to_allocate if x > new_max_unmet])
            cut_down, remainder = exceeds // exceed_count, exceeds % exceed_count
            count = 0
            for n, x in enumerate(new_allocate):
                if to_allocate[n] > new_max_unmet:
                    count += 1
                    if count <= remainder:
                        new_allocate[n] -= (cut_down + 1)
                    else:
                        new_allocate[n] -= cut_down
    else:
        new_allocate = [x for x in to_allocate]
    x_usage = sum(new_allocate)
    on_hand -= round(x_usage, 0)
    if on_hand < 0:
        on_hand = 0.
    return on_hand, new_allocate

def allocation_prioritized_greedy_approx(on_hand, to_allocate, demand_para, rdc_inv,
                                         holding_cost, backorder_cost):
    num_rdc = len(to_allocate)
    new_allocate = np.zeros(num_rdc)

    total_request = sum(to_allocate)

    # 若 on_hand 足够，无需分配
    if total_request <= on_hand:
        return on_hand - total_request, to_allocate

    # 准备数据数组化
    lam = np.array([demand_para[i] for i in range(1, num_rdc+1)])
    inv = np.array([rdc_inv[i] for i in range(1, num_rdc+1)])
    std = np.sqrt(lam)

    # z-score 代表相对库存水平（越低 → 越紧缺）
    z = (inv - lam) / std

    # 构建最小堆（z 越小越优先）
    heap = [(z[i], i) for i in range(num_rdc)]
    heapq.heapify(heap)

    # 分配库存
    for _ in range(int(on_hand)):
        zi, idx = heapq.heappop(heap)

        new_allocate[idx] += 1
        inv[idx] += 1
        
        # 更新 z-score
        zi += 1.0 / std[idx]

        # 插回堆
        heapq.heappush(heap, (zi, idx))

    remaining_on_hand = 0.0
    return remaining_on_hand, new_allocate.tolist()



# ============================================================
#  向量化 poisson_cost_closed —— 完全支持 numpy 数组
# ============================================================
def poisson_cost_closed(hc, bc, para, inv):
    """
    Vectorized Poisson cost expectation.
    Works for:
        - scalar hc, bc, para, inv
        - numpy arrays (broadcasting supported)

    Returns:
        scalar or numpy array (depending on input types)
    """
    hc   = np.asarray(hc, dtype=float)
    bc   = np.asarray(bc, dtype=float)
    para = np.asarray(para, dtype=float)
    inv  = np.asarray(inv)

    # Poisson demand only defined for integer support
    inv_int = np.floor(inv).astype(int)

    # Vectorized CDF calls
    F_inv = stats.poisson.cdf(inv_int, para)
    F_inv_minus = stats.poisson.cdf(inv_int - 1, para)
    F_inv_minus = np.where(inv_int > 0, F_inv_minus, 0.0)

    # Over and shortage expectations
    overage  = hc * (inv_int * F_inv - para * F_inv_minus)
    shortage = bc * ((para - inv_int) * (1.0 - F_inv))

    res = overage + shortage

    # Convert to Python float when all inputs are scalar
    if res.size == 1:
        return float(res)
    return res


_DEMAND_COST_PMF_CACHE = {}


def _lead_time_pmf_cached(demand_model):
    if (
            isinstance(demand_model, tuple)
            and len(demand_model) == 3
            and str(demand_model[0]).lower() == "pmf"
    ):
        support = np.asarray(demand_model[1], dtype=float)
        pmf = np.asarray(demand_model[2], dtype=float)
        return support, pmf / pmf.sum()
    if isinstance(demand_model, dict) and str(demand_model.get("type", "")).lower() == "pmf":
        support = np.asarray(demand_model["support"], dtype=float)
        pmf = np.asarray(demand_model["pmf"], dtype=float)
        return support, pmf / pmf.sum()

    if (
            isinstance(demand_model, tuple)
            and len(demand_model) == 2
            and not isinstance(demand_model[0], str)
    ):
        spec, periods = demand_model
    else:
        spec, periods = demand_model, 1

    periods = int(periods)
    cache_key = (repr(spec), periods)
    if cache_key not in _DEMAND_COST_PMF_CACHE:
        _DEMAND_COST_PMF_CACHE[cache_key] = lead_time_demand_pmf(spec, periods)
    return _DEMAND_COST_PMF_CACHE[cache_key]


def demand_cost_closed(hc, bc, demand_model, inv):
    """
    Distribution-aware expected inventory cost:
        hc * E[(inv - D)^+] + bc * E[(D - inv)^+].

    `demand_model` can be a Poisson mean for backward compatibility, or
    `(single_period_demand_spec, periods)` for exact lead-time demand PMF.
    """
    if isinstance(demand_model, (int, float, np.number)):
        return poisson_cost_closed(hc, bc, demand_model, inv)

    support, pmf = _lead_time_pmf_cached(demand_model)
    inv_arr = np.asarray(inv, dtype=float)
    inv_expanded = np.expand_dims(inv_arr, axis=-1)

    overage = np.maximum(inv_expanded - support, 0.0)
    shortage = np.maximum(support - inv_expanded, 0.0)
    expected_overage = np.sum(overage * pmf, axis=-1)
    expected_shortage = np.sum(shortage * pmf, axis=-1)
    res = np.asarray(hc, dtype=float) * expected_overage + np.asarray(bc, dtype=float) * expected_shortage

    if np.asarray(res).size == 1:
        return float(res)
    return res


def _unit_inventory_cost_reductions(hc, bc, demand_model, inventory_levels):
    """Return ``C(x) - C(x + 1)`` for integer-valued demand.

    For integer ``x`` and demand ``D``, the exact reduction is
    ``bc - (hc + bc) * P(D <= x)``. Inventory states in the OWMR simulator
    are integral, so this avoids constructing overage/shortage matrices. A
    direct cost-difference fallback preserves semantics for unusual
    non-integral states or PMF supports.
    """
    levels = np.asarray(inventory_levels, dtype=float)

    if isinstance(demand_model, (int, float, np.number)):
        # Preserve the historical closed-form Poisson objective exactly. Its
        # shortage convention is not algebraically identical to the generic
        # PMF objective below, so changing to the textbook marginal formula
        # here would alter previously trained policies.
        if levels.size == 0:
            return np.empty(0, dtype=float)
        cost_levels = np.concatenate((levels, levels[-1:] + 1.0))
        costs = np.atleast_1d(
            poisson_cost_closed(hc, bc, demand_model, cost_levels)
        )
        return costs[:-1] - costs[1:]

    integral_levels = np.allclose(levels, np.rint(levels), atol=1e-9, rtol=0.0)
    if integral_levels:
        support, pmf = _lead_time_pmf_cached(demand_model)
        support = np.asarray(support, dtype=float)
        pmf = np.asarray(pmf, dtype=float)
        if (
                support.ndim == 1
                and support.size == pmf.size
                and np.all(np.diff(support) >= 0.0)
                and np.allclose(support, np.rint(support), atol=1e-9, rtol=0.0)
        ):
            cumulative = np.cumsum(pmf)
            indices = np.searchsorted(support, np.rint(levels), side="right") - 1
            cdf = np.zeros_like(levels, dtype=float)
            valid = indices >= 0
            cdf[valid] = cumulative[np.minimum(indices[valid], cumulative.size - 1)]
            return np.atleast_1d(
                float(bc) - (float(hc) + float(bc)) * cdf
            )

    old_cost = np.asarray(
        demand_cost_closed(hc, bc, demand_model, levels),
        dtype=float,
    )
    new_cost = np.asarray(
        demand_cost_closed(hc, bc, demand_model, levels + 1.0),
        dtype=float,
    )
    return np.atleast_1d(old_cost - new_cost)


def _select_ranked_marginal_units(
        marginal_sequences,
        capacity,
        tolerance=None,
        force_global_stable=False,
):
    """Select unit marginals with the legacy stable ordering.

    Convex retailer costs make every sequence non-increasing. A k-way heap
    therefore produces the same ranking as concatenating every unit and using
    a stable global sort, while avoiding that sort and its large temporary
    arrays. The flattened index preserves the old RDC/unit tie-break order.
    """
    allocated = np.zeros(len(marginal_sequences), dtype=float)
    arrays = [np.asarray(sequence, dtype=float).reshape(-1) for sequence in marginal_sequences]

    # The historical scalar-Poisson objective is not always discretely convex.
    # Preserve its exact ranking with a stable global fallback; sufficiently
    # large arrays use a linear-time top-k threshold with stable tie repair.
    if force_global_stable or any(
            np.any(np.diff(values) > 1e-12) for values in arrays
    ):
        if not arrays or sum(values.size for values in arrays) == 0:
            return allocated
        benefits = np.concatenate(arrays)
        owners = np.concatenate([
            np.full(values.size, idx, dtype=np.int64)
            for idx, values in enumerate(arrays)
        ])
        eligible = (
            np.flatnonzero(benefits > tolerance)
            if tolerance is not None
            else np.arange(benefits.size)
        )
        count = min(int(capacity), int(eligible.size))
        if count <= 0:
            return allocated
        if count == eligible.size:
            selected = eligible
        elif eligible.size < 16_384:
            ranked = eligible[
                np.argsort(-benefits[eligible], kind="stable")
            ]
            selected = ranked[:count]
        else:
            eligible_values = benefits[eligible]
            threshold = np.partition(
                eligible_values,
                eligible_values.size - count,
            )[eligible_values.size - count]
            higher = eligible[eligible_values > threshold]
            tie_count = count - higher.size
            ties = eligible[eligible_values == threshold][:tie_count]
            selected = np.concatenate((higher, ties))
        return np.bincount(owners[selected], minlength=len(arrays)).astype(float)

    heap = []
    flat_offset = 0
    for rdc_index, values in enumerate(arrays):
        if values.size:
            heapq.heappush(
                heap,
                (-float(values[0]), flat_offset, rdc_index, 0),
            )
        flat_offset += int(values.size)

    for _ in range(int(capacity)):
        if not heap:
            break
        negative_benefit, flat_index, rdc_index, unit_index = heapq.heappop(heap)
        benefit = -negative_benefit
        if tolerance is not None and benefit <= tolerance:
            break
        allocated[rdc_index] += 1.0

        next_unit = unit_index + 1
        sequence = arrays[rdc_index]
        if next_unit < len(sequence):
            heapq.heappush(
                heap,
                (
                    -float(sequence[next_unit]),
                    flat_index + 1,
                    rdc_index,
                    next_unit,
                ),
            )
    return allocated


def _inventory_cost_reduction_sequences(
        requests,
        demand_para,
        rdc_inventory_position,
        holding_cost,
        backorder_cost,
):
    """Build one unit-marginal sequence per RDC."""
    requests = np.asarray(requests, dtype=int)
    sequences = []
    for idx, requested_units in enumerate(requests):
        if requested_units <= 0:
            sequences.append(np.empty(0, dtype=float))
            continue
        rdc = idx + 1
        levels = float(rdc_inventory_position[rdc]) + np.arange(
            requested_units,
            dtype=float,
        )
        sequences.append(
            _unit_inventory_cost_reductions(
                holding_cost[rdc],
                backorder_cost[rdc],
                demand_para[rdc],
                levels,
            )
        )
    return sequences


def allocation_rb_relaxed_marginal(
        on_hand,
        to_allocate,
        demand_para,
        rdc_inventory_position,
        holding_cost,
        backorder_cost,
):
    """Apply RB's relaxed retailer marginals at a fixed shipment budget.

    The native RB policy also chooses an aggregate downstream release quantity
    from its fixed retailer base-stock levels.  This allocator deliberately
    holds that quantity at ``min(on_hand, sum(requests))`` so controlled A/B
    evaluations can isolate RB's RDC split from its aggregate release rule.
    """
    requests = np.maximum(
        np.rint(np.asarray(to_allocate, dtype=float)),
        0.0,
    ).astype(int)
    available = max(float(on_hand), 0.0)
    capacity = min(
        max(int(np.floor(available + 1e-9)), 0),
        int(requests.sum()),
    )
    if capacity <= 0:
        return available, np.zeros_like(requests, dtype=float).tolist()

    h0 = float(holding_cost.get(0, 0.0))
    marginal_values = []
    for index, requested_units in enumerate(requests):
        if requested_units <= 0:
            marginal_values.append(np.empty(0, dtype=float))
            continue
        rdc = index + 1
        levels = float(rdc_inventory_position[rdc]) + np.arange(
            requested_units,
            dtype=float,
        )
        # C_j(x)=(h_j-h_0)E[(x-D)^+]+(b_j+h_0)E[(D-x)^+].
        marginal_values.append(
            _unit_inventory_cost_reductions(
                float(holding_cost[rdc]) - h0,
                float(backorder_cost[rdc]) + h0,
                demand_para[rdc],
                levels,
            )
        )

    numeric_poisson = all(
        count <= 0 or isinstance(demand_para[index + 1], (int, float, np.number))
        for index, count in enumerate(requests)
    )
    allocated = _select_ranked_marginal_units(
        marginal_values,
        capacity,
        force_global_stable=numeric_poisson,
    )
    allocated_total = float(allocated.sum())
    if np.any(allocated - requests > 1e-9):
        raise RuntimeError("RB marginal allocation exceeded an RDC request.")
    if allocated_total > available + 1e-9:
        raise RuntimeError("RB marginal allocation exceeded CDC on-hand inventory.")
    return max(available - allocated_total, 0.0), allocated.tolist()


# ============================================================
#  allocation_prioritized_greedy_fast —— 高性能版本
# ============================================================
# @timer
def allocation_prioritized_greedy_fast(
        on_hand, to_allocate, demand_para, rdc_inventory_position,
        holding_cost, backorder_cost):

    num_rdc = len(to_allocate)
    new_allocate = np.zeros(num_rdc)

    # If on-hand stock is enough → allocate fully
    if sum(to_allocate) <= on_hand:
        return on_hand - sum(to_allocate), to_allocate

    # ------------------------------------------------------------
    # Vectorize parameters
    # Demand parameters (lambda)
    para = np.array([demand_para[r] for r in range(1, num_rdc + 1)])
    # Current inventory positions
    inv = np.array([rdc_inventory_position[r] for r in range(1, num_rdc + 1)])
    # Costs
    hc = np.array([holding_cost[r] for r in range(1, num_rdc + 1)])
    bc = np.array([backorder_cost[r] for r in range(1, num_rdc + 1)])

    # ------------------------------------------------------------
    # Initial ΔC calculation (vectorized)
    old = poisson_cost_closed(hc, bc, para, inv)
    new = poisson_cost_closed(hc, bc, para, inv + 1)
    delta = old - new

    # ------------------------------------------------------------
    # Build max heap (Python uses min heap → store negative delta)
    heap = [(-delta[i], i) for i in range(num_rdc)]
    heapq.heapify(heap)

    # ------------------------------------------------------------
    # Greedy allocation loop
    # for _ in range(int(on_hand)):
    while on_hand > 0:
        neg_dc, idx = heapq.heappop(heap)

        # allocate 1 unit to this RDC
        new_allocate[idx] += 1
        inv[idx] += 1

        # incremental update
        old[idx] = new[idx]  # previous new becomes old
        new[idx] = poisson_cost_closed(hc[idx], bc[idx], para[idx], inv[idx] + 1)
        delta[idx] = old[idx] - new[idx]

        # push back to heap
        heapq.heappush(heap, (-delta[idx], idx))

    return 0.0, new_allocate.tolist()


def allocation_random_sequential(on_hand, to_allocate, rng=None):
    new_allocate = [0. for x in to_allocate]
    num_rdc = len(to_allocate)
    x_usage = sum(to_allocate)
    if x_usage > on_hand:
        numbers = list(range(num_rdc))
        (rng if rng is not None else random).shuffle(numbers)
        for rdc in numbers:
            pre_allocate = to_allocate[rdc]
            if pre_allocate > on_hand:
                new_allocate[rdc] = on_hand
                on_hand = 0.
                break
            else:
                new_allocate[rdc] = pre_allocate
                on_hand -= pre_allocate
        return on_hand, new_allocate
    else:
        new_allocate = [x for x in to_allocate]
        x_usage = sum(new_allocate)
        on_hand -= round(x_usage, 0)
        if on_hand < 0:
            on_hand = 0.
        # logger.info(new_allocate, rdc_inventory_position)
        return on_hand, new_allocate
    


def allocation_prioritized_greedy_inequality(
        on_hand,
        to_allocate,
        demand_para,
        rdc_inventory_position,
        holding_cost,
        backorder_cost,
        marginal_tolerance=1e-9,
):
    """Exact feasible unit-marginal inequality allocation.

    A requested unit is allocated only when it reduces total one-period
    expected system cost. Its benefit includes both the downstream expected
    inventory-cost reduction and the warehouse holding cost avoided by
    shipping that unit. Unlike equality allocation, non-beneficial units may
    remain at the CDC even when sufficient inventory is available.
    """
    requests = np.maximum(
        np.rint(np.asarray(to_allocate, dtype=float)),
        0.0,
    ).astype(int)
    available = max(float(on_hand), 0.0)
    capacity = max(int(np.floor(available + 1e-9)), 0)
    if capacity <= 0 or int(requests.sum()) <= 0:
        return available, np.zeros_like(requests, dtype=float).tolist()

    h0 = float(holding_cost.get(0, 0.0))
    marginal_values = _inventory_cost_reduction_sequences(
        requests,
        demand_para,
        rdc_inventory_position,
        holding_cost,
        backorder_cost,
    )
    marginal_values = [values + h0 for values in marginal_values]
    numeric_poisson = all(
        count <= 0 or isinstance(demand_para[idx + 1], (int, float, np.number))
        for idx, count in enumerate(requests)
    )
    allocated = _select_ranked_marginal_units(
        marginal_values,
        capacity,
        tolerance=float(marginal_tolerance),
        force_global_stable=numeric_poisson,
    )

    if np.any(allocated - requests > 1e-9):
        raise RuntimeError("Exact inequality allocation exceeded an RDC request.")
    allocated_total = float(allocated.sum())
    if allocated_total > available + 1e-9:
        raise RuntimeError("Exact inequality allocation exceeded CDC on-hand inventory.")
    return max(available - allocated_total, 0.0), allocated.tolist()



def allocation_prioritized_greedy_coarse_legacy(
        on_hand,
        to_allocate,
        demand_para,
        rdc_inventory_position,
        holding_cost,
        backorder_cost,
):
    """Legacy coarse-batch equality allocator retained for controlled A/B tests."""
    # logger.info(on_hand)
    new_allocate = [0. for x in to_allocate]
    x_usage = sum(to_allocate)
    primal_on_hand = on_hand
    num_rdc = len(to_allocate)
    if x_usage > on_hand:
        allocation_interval = allocation_quant = max(int(on_hand / 20), 1)
        delta_cost = np.zeros(num_rdc)
        old, new = {}, {}
        for r in range(1, num_rdc + 1):
            if new_allocate[r-1] > to_allocate[r-1] - 1e-6:
                delta_cost[r-1] = -1000
            else:
                para = demand_para[r]
                inv_level = rdc_inventory_position[r]
                hc, bc = holding_cost[r], backorder_cost[r]
                old[r] = demand_cost_closed(hc, bc, para, inv_level)
                new[r] = demand_cost_closed(hc, bc, para, inv_level + allocation_quant)
                delta_cost[r-1] = old[r] - new[r]
        while on_hand > 0:
            # if max(delta_cost) < 1e-6:
            #     break
            rdc_best = np.argmax(delta_cost)
            available_allocation = min(allocation_quant, to_allocate[rdc_best] - new_allocate[rdc_best])
            new_allocate[rdc_best] += available_allocation
            rdc_inventory_position[rdc_best + 1] += available_allocation
            on_hand -= available_allocation

            if new_allocate[rdc_best] > to_allocate[rdc_best] - 1e-6:
                delta_cost[rdc_best] = -1000
            else:
                para = demand_para[rdc_best + 1]
                inv_level = rdc_inventory_position[rdc_best + 1]
                hc, bc = holding_cost[rdc_best + 1], backorder_cost[rdc_best + 1]
                allocation_quant = allocation_interval if on_hand > allocation_interval else on_hand

                old[rdc_best + 1] = new[rdc_best + 1]
                new[rdc_best + 1] = demand_cost_closed(hc, bc, para, inv_level + allocation_quant)
                delta_cost[rdc_best] = old[rdc_best + 1] - new[rdc_best + 1]
        return on_hand, new_allocate
        
    else:
        new_allocate = [x for x in to_allocate]
        x_usage = sum(new_allocate)
        on_hand -= round(x_usage, 0)
        if on_hand < 0:
            on_hand = 0.
        # logger.info(new_allocate, rdc_inventory_position)
        return on_hand, new_allocate


def allocation_prioritized_greedy(
        on_hand,
        to_allocate,
        demand_para,
        rdc_inventory_position,
        holding_cost,
        backorder_cost,
):
    """Exact feasible unit-marginal equality allocation.

    Each requested unit is assigned the reduction in expected downstream
    inventory cost produced by adding that unit at its retailer. Selecting the
    largest marginal reductions is equivalent to unit-by-unit greedy
    allocation because each retailer cost is convex. Computing each
    retailer's full marginal sequence in one vectorized call keeps the method
    practical for high-dimensional cases.
    """
    requests = np.maximum(np.rint(np.asarray(to_allocate, dtype=float)), 0.0).astype(int)
    available = max(float(on_hand), 0.0)
    capacity = max(int(np.floor(available + 1e-9)), 0)
    total_requested = int(requests.sum())

    if total_requested <= capacity:
        allocated = requests.astype(float)
        return max(available - float(allocated.sum()), 0.0), allocated.tolist()
    if capacity <= 0 or total_requested <= 0:
        return available, np.zeros_like(requests, dtype=float).tolist()

    marginal_values = _inventory_cost_reduction_sequences(
        requests,
        demand_para,
        rdc_inventory_position,
        holding_cost,
        backorder_cost,
    )
    numeric_poisson = all(
        count <= 0 or isinstance(demand_para[idx + 1], (int, float, np.number))
        for idx, count in enumerate(requests)
    )
    allocated = _select_ranked_marginal_units(
        marginal_values,
        capacity,
        force_global_stable=numeric_poisson,
    )

    if np.any(allocated - requests > 1e-9):
        raise RuntimeError("Exact prioritized allocation exceeded an RDC request.")
    allocated_total = float(allocated.sum())
    if allocated_total > available + 1e-9:
        raise RuntimeError("Exact prioritized allocation exceeded CDC on-hand inventory.")
    return max(available - allocated_total, 0.0), allocated.tolist()


def poisson_prepare(para):
    sd = np.sqrt(para)
    begin = max(0, int(para - 6*sd))
    end   = int(para + 6*sd)
    d = np.arange(begin, end+1)
    pmf = stats.poisson.pmf(d, para)
    return d, pmf

def poisson_cost_fast(hc, bc, inv_level, d, pmf):
    over = np.maximum(inv_level - d, 0)
    short = np.maximum(d - inv_level, 0)
    return np.sum((over*hc + short*bc) * pmf)


# @timer
def poisson_cost_expectation_vec(hc, bc, para, inv_level):
    # 使用整型范围
    # begin = int(stats.poisson.ppf(0.01, para))
    # end   = int(stats.poisson.ppf(0.99, para))
    sd = np.sqrt(para)
    begin = max(0, int(para - 6*sd))
    end   = int(para + 6*sd)
    d = np.arange(begin, end + 1)
    pmf = stats.poisson.pmf(d, para)

    # 直接向量化成本计算
    overage  = np.maximum(inv_level - d, 0)
    shortage = np.maximum(d - inv_level, 0)
    
    cost = (overage * hc + shortage * bc) * pmf
    return cost.sum()


### Rong Ying Recursive Optimization Heuristic
def main_heuristic_cost_simualtion(info_input: OWMR_input, input_args, method='RO'):
    if method == 'RO':
        retailer_base_stock_level, best_warehouse_local_level, best_warehouse_base_stock_level = recursive_heuristic(info_input)
    if method in {'RO_tail_fixed', 'RO_tail'}:
        retailer_base_stock_level, best_warehouse_local_level, best_warehouse_base_stock_level = recursive_heuristic_tail_fixed(info_input)
    if method == 'DA':
        retailer_base_stock_level, best_warehouse_base_stock_level = decomposition_aggregation(info_input)
    if method in {'DA_distribution_aware', 'DA_dist', 'DA_DA'}:
        retailer_base_stock_level, best_warehouse_base_stock_level = decomposition_aggregation_distribution_aware(info_input)
    if method == 'RB':
        best_cost, retailer_base_stock_level, best_warehouse_base_stock_level = relaxation_based_heuristic(info_input)
    if method in {'RB_distribution_aware', 'RB_dist', 'RB_distribution_aware_capped', 'RB_dist_capped'}:
        best_cost, retailer_base_stock_level, best_warehouse_base_stock_level = relaxation_based_heuristic_distribution_aware_capped(info_input)
    if method in {'RB_distribution_aware_uncapped', 'RB_dist_uncapped'}:
        best_cost, retailer_base_stock_level, best_warehouse_base_stock_level = relaxation_based_heuristic_distribution_aware(info_input)

    logger.info('算法给出的策略为:')
    # logger.info(f'warehouse的发货水位为{recursive_level}')
    logger.info(f'warehouse的订货水位为{best_warehouse_base_stock_level}')
    logger.info(f'rdc的订货水位为{retailer_base_stock_level}')

    env = generate_env(OWMRInventoryEnv, input_args)
    env.method = 'heu'
    env.total_periods = 1000
    env.reset()
    reward_process = []
    eval_times = 100
    # heu_reward, std, median = 0., 0., 0.
    calc_step = int(env.total_periods / 2)
    # statistics
    rewards_step = np.zeros((eval_times, calc_step))
    order_step = {k: np.zeros((eval_times, calc_step)) for k in env.rdcs.keys()}
    on_hand_inv_step = {k: np.zeros((eval_times, calc_step)) for k in env.rdcs.keys()}
    echelon_inv_step = {k: np.zeros((eval_times, calc_step)) for k in env.rdcs.keys()}
    backorder_step = {k: np.zeros((eval_times, calc_step)) for k in env.rdcs.keys()}

    order_cdc_step = np.zeros((eval_times, calc_step))
    on_hand_inv_cdc_step = np.zeros((eval_times, calc_step))
    echelon_inv_cdc_step = np.zeros((eval_times, calc_step))
    

    for j in range(eval_times):
        i = 0
        while True:
            # env.step_satisfy_demand()
            # env.get_echelon_state()

            action = heuristic_order_qty(env, info_input, retailer_base_stock_level, best_warehouse_base_stock_level, method=method)
            _, reward, _, info = env.step(action)
            cost, inv_info, order = info
            reward_process.append(-1 * reward)
            
            if i >= calc_step:
                env.get_echelon_state()
                position = i - calc_step
                rewards_step[j, position] = reward
                for rdc in env.rdcs.keys():
                    order_step[rdc][j, position] = order[rdc]
                    on_hand_inv_step[rdc][j, position] = inv_info[rdc]
                    echelon_inv_step[rdc][j, position] = env.state_echelon_rdc[rdc][-1]
                    backorder_step[rdc][j, position] = -1 * inv_info[rdc] if inv_info[rdc] < 0 else 0.
                order_cdc_step[j, position] = order[0]
                on_hand_inv_cdc_step[j, position] =  inv_info[0]
                echelon_inv_cdc_step[j, position] = env.state_echelon_cdc[-1]
            i += 1
            if i == env.total_periods:
                env.reset()
                break
        # if len(reward_process) > 300 and len(reward_process) % 100 == 0:
        #     current_reward = np.mean(np.array(reward_process[-100:]))
        #     past_reward = np.mean(np.array(reward_process[-200: -100]))
        #     # past_past_reward = np.sum(reward_process[-300: -200]) / 100
        #     logger.info(f'最后100期的平均成本为{current_reward}, 最后100-200期的平均成本为{past_reward}')
        #     if abs(current_reward - past_reward) / past_reward < 0.01:
        #         heu_reward = current_reward
        #         std = np.std(np.array(reward_process[-100:]))
        #         median = np.median(np.array(reward_process[-100:]))
        #         break
    return retailer_base_stock_level, best_warehouse_base_stock_level, rewards_step, order_step, on_hand_inv_step, echelon_inv_step, backorder_step, order_cdc_step, on_hand_inv_cdc_step, echelon_inv_cdc_step


def heuristic_order_qty(env, info_input, retailer_base_stock_level, best_warehouse_base_stock_level, method='RO'):
    rb_methods = {
        'RB',
        'RB_distribution_aware',
        'RB_dist',
        'RB_distribution_aware_capped',
        'RB_dist_capped',
        'RB_distribution_aware_uncapped',
        'RB_dist_uncapped',
    }
    if method not in rb_methods:
        # retailer订货及warehouse发货
        itpj = {k: env.state_echelon_rdc[k][-1] for k in env.rdcs.keys()}
        action_rdc = np.zeros(len(info_input.rdc_dict))
        # 订货至最优水平
        for rdc in env.rdcs.keys():
            action_rdc[rdc - 1] = max(0, retailer_base_stock_level[rdc] - itpj[rdc])
    else:
        # retailer订货及warehouse发货
        itpr = np.sum([env.state_echelon_rdc[k][-1] for k in env.state_echelon_rdc.keys()])
        out_qty = max(0, retailer_base_stock_level[0] - itpr)
        # logger.info(f'当前下游水位为{itpr}, 理论出库量为{out_qty}')
        # logger.info(f'按照算法, 当前仓库库存为{env.state_echelon_cdc[0] - itpr}')
        # logger.info(f'按照模型, 当前仓库库存为{env.CDC_inventory[0]}')
        if env.CDC_inventory[0] < out_qty:
            # logger.info('当前库存无法满足发货需求')
            out_qty = env.CDC_inventory[0]
        if out_qty > 0:  # allocate
            itpj = {k: env.state_echelon_rdc[k][-1] for k in env.rdcs.keys()}
            if method in {
                'RB_distribution_aware',
                'RB_dist',
                'RB_distribution_aware_capped',
                'RB_dist_capped',
                'RB_distribution_aware_uncapped',
                'RB_dist_uncapped',
            }:
                action_rdc = rb_distribution_aware_allocation(
                    info_input, itpj, retailer_base_stock_level, out_qty
                )
                action_cdc = max(0, best_warehouse_base_stock_level - env.state_echelon_cdc[-1])
                action = np.hstack([action_rdc, action_cdc])
                return action
            action_rdc = np.zeros(len(info_input.rdc_dict))
            # 订货至最优水平
            for rdc in env.rdcs.keys():
                action_rdc[rdc - 1] = max(0, retailer_base_stock_level[rdc] - itpj[rdc])
            order_qty = np.sum(action_rdc)
            # logger.info(f'理论出库量为{out_qty}, 理论订货量为{action_rdc} (总订货量为{order_qty}) ,')
            if order_qty > out_qty:  # 未能满足订货要求
                delta = int(order_qty - out_qty)
                # combination of allocation of unmet stock
                allocated_quantities = distribute_unmet_stock(delta, retailer_base_stock_level)
                best_allocation = [0. for _ in info_input.rdc_dict.keys()]
                best_cost = 1e5
                for allocated in allocated_quantities:
                    valid = True
                    for rdc in env.rdcs.keys():
                        if allocated[rdc - 1] > action_rdc[rdc - 1]:
                            valid = False
                            break
                    if not valid: continue
                    retailer_cost_function = cost_function_retailers_distribution_aware \
                        if method in {'RB_distribution_aware', 'RB_dist'} else cost_function_retailers
                    current_cost = np.sum([
                        retailer_cost_function(
                            info_input,
                            rdc,
                            min(retailer_base_stock_level[rdc], itpj[rdc] + action_rdc[rdc - 1]) - allocated[rdc - 1]
                        )
                        for rdc in info_input.rdc_dict.keys()
                    ])
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_allocation = allocated
                for rdc in env.rdcs.keys():
                    action_rdc[rdc - 1] -= best_allocation[rdc - 1]
        else:
            action_rdc = np.zeros(len(info_input.rdc_dict))
    # warehouse订货
    action_cdc = max(0, best_warehouse_base_stock_level - env.state_echelon_cdc[-1])
    action = np.hstack([action_rdc, action_cdc])
    return action



def calc_heu_results(input_args, instance_name='dataset_15', method='RO', output_path='output_OWMR_0822_size20', detail=True):
    my_mkdir(f'/{output_path}')
    info_input = input_args['info_input']
    base_stock_level, best_warehouse_base_stock_level, rewards_step, order_step, on_hand_inv_step, echelon_inv_step, backorder_step, order_cdc_step, on_hand_inv_cdc_step, echelon_inv_cdc_step = main_heuristic_cost_simualtion(info_input, input_args, method)

    rdcs = order_step.keys()
    rdcs_num = len(rdcs)
    eval_times = np.shape(order_cdc_step)[0]
    calc_steps = np.shape(order_cdc_step)[1]
    if detail:
        df_agent_result = pd.DataFrame(columns=['Episode', 'Step', 'Location', 'On hand inventory', 'Echelon inventory', 'Order quantity', 'Order-up-to level', 'Demand'])
        datas = []
        for i in range(eval_times):
            for j in range(calc_steps):
                for x in rdcs:
                    datas.append([i, j, 'Retailer %d'%x, on_hand_inv_step[x][i, j], echelon_inv_step[x][i, j], order_step[x][i, j], base_stock_level[x], input_args['all_demand_list'][x][(i+1) % 100, 500 + j]])
                datas.append([i, j, 'Warehouse', on_hand_inv_cdc_step[i, j], echelon_inv_cdc_step[i, j], order_cdc_step[i, j], best_warehouse_base_stock_level, '/'])

        df_agent_result = pd.DataFrame([pd.Series(row, index=df_agent_result.columns) for row in datas])

    backorder_cost = info_input.backorder_cost
    holding_cost = info_input.holding_cost

    df_result_total = pd.DataFrame(columns=['Item', 'Value'])

    avg_cost, se_cost, median_cost = episode_average_statistics(-rewards_step)

    positive_on_hand = {rdc: np.maximum(on_hand_inv_step[rdc], 0) for rdc in rdcs}

    avg_on_hand_inv_rdc = {rdc: np.mean(on_hand_inv_step[rdc]) for rdc in rdcs}
    avg_positive_on_hand_inv_rdc = {rdc: np.mean(positive_on_hand[rdc]) for rdc in rdcs}
    avg_echelon_inv_rdc = {rdc: np.mean(echelon_inv_step[rdc]) for rdc in rdcs}
    avg_backorder_rdc = {rdc: np.mean(backorder_step[rdc]) for rdc in rdcs}
    avg_order_rdc = {rdc: np.mean(order_step[rdc]) for rdc in rdcs}

    avg_echelon_inv_cdc = np.mean(echelon_inv_cdc_step)
    avg_order_cdc = np.mean(order_cdc_step)

    rdc_decomposition_cost = {rdc: backorder_cost[rdc] * avg_backorder_rdc[rdc] + holding_cost[rdc] * avg_positive_on_hand_inv_rdc[rdc] for rdc in rdcs}
    cdc_cost = avg_on_hand_inv_cdc = round(np.mean(on_hand_inv_cdc_step), 3)
    rdc_decomposition_bh_cost = {'holding': np.sum([holding_cost[rdc] * avg_positive_on_hand_inv_rdc[rdc] for rdc in rdcs]), 'backorder': np.sum([backorder_cost[rdc] * avg_backorder_rdc[rdc] for rdc in rdcs]), 'warehouse': avg_on_hand_inv_cdc}
    max_cost, min_cost = max(rdc_decomposition_cost.values()), min(rdc_decomposition_cost.values())
    ratio = round(max_cost / min_cost, 3)

    avg_on_hand_inv_rdc[0] = avg_on_hand_inv_cdc
    avg_positive_on_hand_inv_rdc[0] = avg_on_hand_inv_cdc
    avg_order_rdc[0] = avg_order_cdc
    avg_echelon_inv_rdc[0] = avg_echelon_inv_cdc
    rdc_decomposition_cost[0] = avg_on_hand_inv_cdc
    base_stock_level[0] = best_warehouse_base_stock_level
    ratio_component = {rdc: rdc_decomposition_cost[rdc] / avg_cost for rdc in rdc_decomposition_cost.keys()}

    avg_cost = round(avg_cost, 3)
    se_cost = round(se_cost, 3)
    avg_on_hand_inv_rdc = {k: round(v, 3) for k,v in avg_on_hand_inv_rdc.items()}
    avg_positive_on_hand_inv_rdc = {k: round(v, 3) for k,v in avg_positive_on_hand_inv_rdc.items()}
    avg_order_rdc = {k: round(v, 3) for k,v in avg_order_rdc.items()}
    avg_echelon_inv_rdc = {k: round(v, 3) for k,v in avg_echelon_inv_rdc.items()}
    rdc_decomposition_cost = {k: round(v, 3) for k,v in rdc_decomposition_cost.items()}
    rdc_decomposition_bh_cost = {k: round(v, 3) for k,v in rdc_decomposition_bh_cost.items()}
    ratio_component = {k: 100 * round(v, 4) for k,v in ratio_component.items()}
    avg_backorder_rdc = {k: round(v, 3) for k,v in avg_backorder_rdc.items()}
    
    datas = []
    datas.append(['No. of retailers', rdcs_num])
    datas.append(['Demand feature', info_input.demand_feature])
    datas.append(['Retailer lead time', info_input.rdc_dict])
    datas.append(['Warehouse lead time', info_input.item_lead_time])
    datas.append(['Unit backorder cost', backorder_cost])
    datas.append(['Unit holding cost', holding_cost])
    datas.append(['Total cost', avg_cost])
    datas.append(['Cost SE', se_cost])
    datas.append(['Cost median', median_cost])
    datas.append(['Cost decomposition distribution', rdc_decomposition_cost])
    datas.append(['Cost decomposition distribution proportion', ratio_component])
    datas.append(['Cost decomposition holding backorder', rdc_decomposition_bh_cost])
    datas.append(['Max-min ratio', ratio])
    datas.append(['Inventory position', avg_on_hand_inv_rdc])
    datas.append(['On hand', avg_positive_on_hand_inv_rdc])
    datas.append(['Backorder', avg_backorder_rdc])
    datas.append(['Echelon', avg_echelon_inv_rdc])
    datas.append(['Order-up-to level', base_stock_level])
    datas.append(['Order', avg_order_rdc])
    
    df_result_total = pd.DataFrame([pd.Series(row, index=df_result_total.columns) for row in datas])
    df_result_total = df_result_total.round(3)
    with pd.ExcelWriter(f'./{output_path}/heu_results_{method}_{instance_name}.xlsx') as writer:
        if detail:
            df_agent_result.to_excel(writer, sheet_name='Detailed results', index=False)
        df_result_total.to_excel(writer, sheet_name='Summary', index=False)


### Rong Ying Decomposition-Aggregation Heuristic
def decomposition_aggregation(info_input: OWMR_input):
    # Step 1, 与RO一致，求RDC的订货水位
    rdcs = list(info_input.rdc_dict.keys())
    ## Figure out each base stock level $s_j^*$ for retailers
    order_up_to_level = {}
    for j in rdcs:
        # Newsvendor quantile
        cdf_quantile = (info_input.backorder_cost[j] + info_input.holding_cost[0]) / (
            info_input.backorder_cost[j] + info_input.holding_cost[j]
        )
        para = info_input.Poisson_para[j]
        order_up_to_level[j] = stats.poisson.ppf(min(0.99, cdf_quantile), para)
        # order_up_to_level[0] += order_up_to_level[j]
    # 计算serial system中warehouse的水位
    order_up_to_level_warehouse = {}
    for j in rdcs:
        para_ware = demand_mean(info_input.demand_feature[j]) * (info_input.item_lead_time + 1 + info_input.rdc_dict[j])
        # def poisson_cdf(x):
        #     return stats.poisson.cdf(k=x, mu=para_ware)
        # def plf(x):
        #     return piecewise_linear_function(poisson_cdf, x)
        order_up_to_level_warehouse[j] = 0.5 * (
            find_cdf_inverse_point(para_ware, 
            info_input.backorder_cost[j] / (info_input.backorder_cost[j] + info_input.holding_cost[j])) 
            + find_cdf_inverse_point(para_ware,
            info_input.backorder_cost[j] / (info_input.backorder_cost[j] + info_input.holding_cost[0])))

    # Step 2, approximate the expected backorders of the warehouse in serial system
    backorder_serial = 0
    for j in rdcs:
        local_base_stock_level = order_up_to_level_warehouse[j] - order_up_to_level[j]
        para_current = (info_input.item_lead_time + 1) * demand_mean(info_input.demand_feature[j])
        backorder_serial += loss_fucntion_poisson(para_current, local_base_stock_level)
    
    # Step 3, Aggregate the serial systems back into the OWMR system utilizing a “backorder matching” procedure
    # Bisection search
    warehouse_local_base_stock_level = backorder_matching(info_input.Poisson_para[0], backorder_serial)
    warehouse_base_stock_level = warehouse_local_base_stock_level + np.sum(list(order_up_to_level.values()))
    return order_up_to_level, warehouse_base_stock_level


def pmf_expected_shortage(support, pmf, target):
    support = np.asarray(support, dtype=float)
    pmf = np.asarray(pmf, dtype=float)
    return float(np.sum(np.maximum(support - target, 0.0) * pmf))


def pmf_piecewise_inverse_point(support, pmf, quantile):
    support = np.asarray(support, dtype=float)
    pmf = np.asarray(pmf, dtype=float)
    q = float(np.clip(quantile, 0.0, 1.0))
    cdf = np.cumsum(pmf)
    idx = int(np.searchsorted(cdf, q, side="left"))
    idx = min(max(idx, 0), len(support) - 1)
    prev_cdf = 0.0 if idx == 0 else float(cdf[idx - 1])
    current_mass = max(float(cdf[idx] - prev_cdf), 1e-12)
    if idx == 0 and support[idx] <= 0 and cdf[idx] >= q:
        return float(q / (2.0 * max(float(cdf[idx]), 1e-12)))
    return float(max(0.0, support[idx] - 0.5 + (q - prev_cdf) / current_mass))


def pmf_backorder_matching(support, pmf, target):
    support = np.asarray(support, dtype=float)
    upper = int(max(0, np.ceil(support[-1])))
    target = float(max(0.0, target))
    for s in range(upper + 1):
        if pmf_expected_shortage(support, pmf, s) <= target:
            return float(s)
    return float(upper + 1)


def aggregate_lead_time_demand_pmf(info_input: OWMR_input, periods):
    total_pmf = np.array([1.0], dtype=float)
    support_offset = 0
    for rdc in info_input.rdc_dict.keys():
        support, pmf = lead_time_demand_pmf(info_input.demand_feature[rdc], periods)
        support_offset += int(round(float(support[0])))
        total_pmf = np.convolve(total_pmf, pmf)
    total_support = np.arange(len(total_pmf), dtype=float) + support_offset
    total_pmf = total_pmf / total_pmf.sum()
    return total_support, total_pmf


def decomposition_aggregation_distribution_aware(info_input: OWMR_input):
    rdcs = list(info_input.rdc_dict.keys())
    order_up_to_level = {}
    order_up_to_level_warehouse = {}
    backorder_serial = 0.0

    for j in rdcs:
        cdf_quantile = (info_input.backorder_cost[j] + info_input.holding_cost[0]) / (
            info_input.backorder_cost[j] + info_input.holding_cost[j]
        )
        retailer_periods = info_input.rdc_dict[j] + 1
        order_up_to_level[j] = lead_time_demand_quantile(
            info_input.demand_feature[j], min(0.99, cdf_quantile), retailer_periods
        )

        serial_periods = info_input.item_lead_time + 1 + info_input.rdc_dict[j]
        serial_support, serial_pmf = lead_time_demand_pmf(info_input.demand_feature[j], serial_periods)
        order_up_to_level_warehouse[j] = 0.5 * (
            pmf_piecewise_inverse_point(
                serial_support,
                serial_pmf,
                info_input.backorder_cost[j] / (info_input.backorder_cost[j] + info_input.holding_cost[j]),
            )
            + pmf_piecewise_inverse_point(
                serial_support,
                serial_pmf,
                info_input.backorder_cost[j] / (info_input.backorder_cost[j] + info_input.holding_cost[0]),
            )
        )

        local_base_stock_level = order_up_to_level_warehouse[j] - order_up_to_level[j]
        warehouse_periods = info_input.item_lead_time + 1
        warehouse_support, warehouse_pmf = lead_time_demand_pmf(info_input.demand_feature[j], warehouse_periods)
        backorder_serial += pmf_expected_shortage(warehouse_support, warehouse_pmf, local_base_stock_level)

    aggregate_support, aggregate_pmf = aggregate_lead_time_demand_pmf(info_input, info_input.item_lead_time + 1)
    warehouse_local_base_stock_level = pmf_backorder_matching(aggregate_support, aggregate_pmf, backorder_serial)
    warehouse_base_stock_level = warehouse_local_base_stock_level + np.sum(list(order_up_to_level.values()))
    return order_up_to_level, warehouse_base_stock_level


DA_distribution_aware = decomposition_aggregation_distribution_aware


def _quantile_series_horizon_for_heuristic(spec):
    """Return the common forecast horizon of a quantile-series demand spec."""
    if not isinstance(spec, dict):
        raise ValueError(
            "Rolling-horizon DA requires quantile-series demand dictionaries; "
            f"got {spec!r}."
        )
    quantiles = spec.get("quantiles", spec)
    lengths = []
    for q in range(1, 10):
        key = f"{q / 10:.1f}"
        values = quantiles.get(key, quantiles.get(q / 10))
        if values is None:
            raise ValueError(f"Quantile-series demand is missing {key}.")
        lengths.append(len(values))
    if not lengths or min(lengths) <= 0 or len(set(lengths)) != 1:
        raise ValueError(f"Quantile-series horizons must be equal and positive, got {lengths}.")
    return lengths[0]


def decomposition_aggregation_rolling_distribution_aware(
        info_input: OWMR_input,
        horizon=None,
        retailer_quantile_cap=0.99,
        cdc_effective_downstream_lead_time="min",
        cdc_review_periods=None,
):
    """Build finite-horizon, period-dependent DA base-stock levels.

    This is the nonstationary counterpart of
    :func:`decomposition_aggregation_distribution_aware`.  At decision period
    ``t`` every PMF starts at ``t``.  The CDC target protects the downstream
    targets that will still be actionable when a production order placed at
    ``t`` arrives after ``L0`` periods.  Inactive decisions are represented by
    ``NaN`` levels plus explicit boolean masks.

    ``cdc_review_periods`` defaults to ``L0 + 1``, matching the periodic-review
    convention of the existing DA implementation.  It remains an explicit
    argument so the literature timing convention can be tested independently
    without changing the policy interface.
    """
    rdcs = list(info_input.rdc_dict.keys())
    if not rdcs:
        raise ValueError("Rolling-horizon DA requires at least one RDC.")

    horizons = {
        _quantile_series_horizon_for_heuristic(info_input.demand_feature[k])
        for k in rdcs
    }
    if len(horizons) != 1:
        raise ValueError(f"All RDC forecasts must share one horizon, got {sorted(horizons)}.")
    forecast_horizon = horizons.pop()
    horizon = forecast_horizon if horizon is None else int(horizon)
    if horizon <= 0 or horizon > forecast_horizon:
        raise ValueError(
            f"horizon must be in [1, {forecast_horizon}], got {horizon}."
        )

    retailer_quantile_cap = float(retailer_quantile_cap)
    if not 0.0 < retailer_quantile_cap <= 1.0:
        raise ValueError("retailer_quantile_cap must be in (0, 1].")
    cdc_mode = str(cdc_effective_downstream_lead_time).lower()
    if cdc_mode not in {"min", "max"}:
        raise ValueError("cdc_effective_downstream_lead_time must be 'min' or 'max'.")

    l0 = int(info_input.item_lead_time)
    if l0 < 0:
        raise ValueError(f"CDC lead time must be nonnegative, got {l0}.")
    if cdc_review_periods is None:
        cdc_review_periods = l0 + 1
    cdc_review_periods = int(cdc_review_periods)
    if cdc_review_periods <= 0:
        raise ValueError("cdc_review_periods must be positive.")

    rdc_levels = {k: np.full(horizon, np.nan, dtype=float) for k in rdcs}
    rdc_active = {k: np.zeros(horizon, dtype=bool) for k in rdcs}
    retailer_quantiles = {}
    for k in rdcs:
        quantile = (
            float(info_input.backorder_cost[k]) + float(info_input.holding_cost[0])
        ) / (
            float(info_input.backorder_cost[k]) + float(info_input.holding_cost[k])
        )
        quantile = float(np.clip(quantile, 0.0, retailer_quantile_cap))
        retailer_quantiles[k] = quantile
        lead_periods = int(info_input.rdc_dict[k]) + 1
        for t in range(horizon):
            active = t < horizon - int(info_input.rdc_dict[k])
            rdc_active[k][t] = active
            if not active:
                continue
            support, pmf = quantile_series_lead_time_pmf(
                info_input.demand_feature[k], t, lead_periods
            )
            rdc_levels[k][t] = pmf_piecewise_inverse_point(support, pmf, quantile)

    downstream_leads = [int(info_input.rdc_dict[k]) for k in rdcs]
    effective_downstream = (
        min(downstream_leads) if cdc_mode == "min" else max(downstream_leads)
    )
    cdc_active = np.zeros(horizon, dtype=bool)
    cdc_local_levels = np.full(horizon, np.nan, dtype=float)
    cdc_echelon_levels = np.full(horizon, np.nan, dtype=float)
    reachable_rdcs = {t: [] for t in range(horizon)}
    serial_levels = {k: np.full(horizon, np.nan, dtype=float) for k in rdcs}
    expected_serial_backorders = {k: np.full(horizon, np.nan, dtype=float) for k in rdcs}

    for t in range(horizon):
        cdc_active[t] = t < horizon - (l0 + effective_downstream)
        if not cdc_active[t]:
            continue

        arrival_period = t + l0
        reachable = [
            k for k in rdcs
            if arrival_period < horizon - int(info_input.rdc_dict[k])
        ]
        reachable_rdcs[t] = reachable
        if not reachable:
            cdc_active[t] = False
            continue

        backorder_serial = 0.0
        downstream_target = 0.0
        for k in reachable:
            retailer_target = float(rdc_levels[k][arrival_period])
            downstream_target += retailer_target
            serial_periods = l0 + int(info_input.rdc_dict[k]) + 1
            serial_support, serial_pmf = quantile_series_lead_time_pmf(
                info_input.demand_feature[k], t, serial_periods
            )
            b = float(info_input.backorder_cost[k])
            hj = float(info_input.holding_cost[k])
            h0 = float(info_input.holding_cost[0])
            serial_level = 0.5 * (
                pmf_piecewise_inverse_point(serial_support, serial_pmf, b / (b + hj))
                + pmf_piecewise_inverse_point(serial_support, serial_pmf, b / (b + h0))
            )
            serial_levels[k][t] = serial_level
            local_level = serial_level - retailer_target
            warehouse_support, warehouse_pmf = quantile_series_lead_time_pmf(
                info_input.demand_feature[k], t, cdc_review_periods
            )
            expected_backorder = pmf_expected_shortage(
                warehouse_support, warehouse_pmf, local_level
            )
            expected_serial_backorders[k][t] = expected_backorder
            backorder_serial += expected_backorder

        aggregate_support, aggregate_pmf = quantile_series_sum_window_pmf([
            (info_input.demand_feature[k], t, cdc_review_periods)
            for k in reachable
        ])
        cdc_local = pmf_backorder_matching(
            aggregate_support, aggregate_pmf, backorder_serial
        )
        cdc_local_levels[t] = cdc_local
        cdc_echelon_levels[t] = cdc_local + downstream_target

    return {
        "method": "DA_rolling_distribution_aware",
        "horizon": horizon,
        "rdc_base_stock": rdc_levels,
        "cdc_local_base_stock": cdc_local_levels,
        "cdc_echelon_base_stock": cdc_echelon_levels,
        "rdc_active_mask": rdc_active,
        "cdc_active_mask": cdc_active,
        "reachable_rdcs": reachable_rdcs,
        "retailer_quantiles": retailer_quantiles,
        "serial_base_stock": serial_levels,
        "expected_serial_backorders": expected_serial_backorders,
        "cdc_review_periods": cdc_review_periods,
        "cdc_effective_downstream_lead_time": cdc_mode,
    }


DA_rolling_distribution_aware = decomposition_aggregation_rolling_distribution_aware


def rolling_da_order_qty(env, schedule):
    """Convert one period of a rolling DA schedule into feasible requests.

    RDC and CDC quantities are requests.  The environment's configured
    allocation rule remains responsible for resolving scarce CDC inventory.
    """
    t = int(env.period)
    horizon = int(schedule["horizon"])
    if not 0 <= t < horizon:
        raise IndexError(f"Environment period {t} is outside schedule horizon {horizon}.")

    action_rdc = np.zeros(len(env.rdcs), dtype=float)
    for n, k in enumerate(env.rdcs.keys()):
        if not bool(schedule["rdc_active_mask"][k][t]):
            continue
        target = float(schedule["rdc_base_stock"][k][t])
        current_position = float(env.state_echelon_rdc[k][-1])
        action_rdc[n] = max(0.0, target - current_position)

    action_cdc = 0.0
    if bool(schedule["cdc_active_mask"][t]):
        target = float(schedule["cdc_echelon_base_stock"][t])
        current_position = float(env.state_echelon_cdc[-1])
        action_cdc = max(0.0, target - current_position)
    return np.hstack((action_rdc, action_cdc))


def backorder_matching(para, target):
    begin, end = stats.poisson.ppf(0.01, para), stats.poisson.ppf(0.99, para)
    current = -1
    while begin != current:
        current = int(0.5 * (begin + end))
        current_value = loss_fucntion_poisson(para, current)
        if current_value == target:
            return current
        if current_value < target:
            end = current
        else:
            begin = current
    return begin + 1



def loss_fucntion_poisson(para, target):
    end = stats.poisson.ppf(0.99, para)
    begin = int(target) + 1
    return np.sum([stats.poisson.pmf(x, para) * (x - target) for x in np.arange(begin, end + 1)])
    

def find_cdf_inverse_point(para_f, quantile):
    if stats.poisson.cdf(0, mu=para_f) >= quantile:
        return quantile / 2 / stats.poisson.cdf(0, mu=para_f)
    critical_point = stats.poisson.ppf(quantile, para_f)
    begin_value, end_value = stats.poisson.cdf(critical_point - 1, para_f), stats.poisson.cdf(critical_point, para_f)
    return critical_point - 0.5 + (quantile - begin_value) / (end_value - begin_value)


def piecewise_linear_function(f, x):
    if x <= 0.5:
        return 2 * f(0) * x
    else:
        floor, ceil = math.floor(x - 0.5), math.ceil(x - 0.5)
        return f(floor) + (f(ceil) - f(floor)) * (x - 0.5 - floor)



def _warehouse_incremental_poisson_mean(info_input: OWMR_input):
    """Poisson demand seen during the CDC's incremental protection window."""
    return float(info_input.item_lead_time) * float(sum(info_input.demand_mean.values()))


def recursive_heuristic(info_input: OWMR_input):
    # Step 1, 与Zipkin一致，求RDC的订货水位
    rdcs = list(info_input.rdc_dict.keys())
    ## Figure out each base stock level $s_j^*$ for retailers
    order_up_to_level = {0: 0.}
    for j in rdcs:
        # Newsvendor quantile
        cdf_quantile = (info_input.backorder_cost[j] + info_input.holding_cost[0]) / (
            info_input.backorder_cost[j] + info_input.holding_cost[j]
        )
        para = info_input.Poisson_para[j]
        order_up_to_level[j] = stats.poisson.ppf(min(0.99, cdf_quantile), para)
        order_up_to_level[0] += order_up_to_level[j]

    # Step 2, 与Zipkin相似，只是将allocation设置为FCFS，将warehouse对RDC的缺货刻画为二项分布
    warehouse_para = _warehouse_incremental_poisson_mean(info_input)
    current_traverse = order_up_to_level[0] + warehouse_para
    best_cost = 1e5
    traverse_interval = max(1, int(current_traverse / 1000))
    while True:
        current_cost = cost_warehouse_ro(info_input, current_traverse, order_up_to_level)
        print(current_cost, current_traverse)
        if current_cost > best_cost:
            break
        else:
            best_cost = current_cost
            current_traverse -= traverse_interval
    best_warehouse_base_stock_level = current_traverse + 1
    best_warehouse_local_level = best_warehouse_base_stock_level - order_up_to_level[0]
    
    # Step 3，更新retailer level
    recursive_level = {}
    for k in rdcs:
        recursive_level[k] = cost_retailer_recursive(info_input, k, best_warehouse_local_level)

    best_warehouse_base_stock_level = best_warehouse_local_level + np.sum(list(recursive_level.values()))
    return recursive_level, best_warehouse_local_level, best_warehouse_base_stock_level


def cost_retailer_recursive(info_input: OWMR_input, rdc, best_warehouse_local_level):
    para = _warehouse_incremental_poisson_mean(info_input)
    para_i = info_input.Poisson_para[rdc]
    hi, bi = info_input.holding_cost[rdc], info_input.backorder_cost[rdc]
    demand_sum = sum([demand_mean(info_input.demand_feature[k]) for k in info_input.rdc_dict.keys()])
    begin, end = stats.poisson.ppf(0.01, para), stats.poisson.ppf(0.99, para)
    begin_i, end_i = stats.poisson.ppf(0.01, para_i), stats.poisson.ppf(0.99, para_i)
    best_cost = 1e5

    para_i = info_input.Poisson_para[rdc]
    begin_rdc, end_rdc = begin_i, end_i
    previous_si = int((begin_rdc + end_rdc) / 2)
    # cost_begin = calc_total_cost(begin_rdc, info_input, rdc, best_warehouse_local_level, begin, end, begin_i, end_i, demand_sum, para, para_i, hi, bi)
    # cost_end = calc_total_cost(end_rdc, info_input, rdc, best_warehouse_local_level, begin, end, begin_i, end_i, demand_sum, para, para_i, hi, bi)
    previous_cost = calc_total_cost(previous_si, info_input, rdc, best_warehouse_local_level, begin, end, begin_i, end_i, demand_sum, para, para_i, hi, bi)
    while True:
        if end_rdc == begin_rdc:
            break
        current_si_left = int((begin_rdc + previous_si) / 2)
        current_si_right = int((end_rdc + previous_si) / 2)
        current_cost_left = calc_total_cost(current_si_left, info_input, rdc, best_warehouse_local_level, begin, end, begin_i, end_i, demand_sum, para, para_i, hi, bi)
        current_cost_right = calc_total_cost(current_si_right, info_input, rdc, best_warehouse_local_level, begin, end, begin_i, end_i, demand_sum, para, para_i, hi, bi)
        if current_cost_left > previous_cost:
            begin_rdc = current_si_left
        else:
            end_rdc = previous_si
            previous_si = current_si_left
        if current_cost_right > previous_cost:
            end_rdc = current_si_right
        else:
            begin_rdc = previous_si
            previous_si = current_si_right
        previous_cost = min(current_cost_left, current_cost_right, previous_cost)
            # previous_cost = current_cost
            # previous_si = current_si
    
    return begin_rdc
                

def calc_total_cost(current_si, info_input: OWMR_input, rdc, best_warehouse_local_level, begin, end, begin_i, end_i, demand_sum, para, para_i, hi, bi):
    total_cost = 0
    for total_demand in np.arange(begin, end + 1):
        p1 = stats.poisson.pmf(total_demand, para)
        warehouse_backorder = max(0, total_demand - best_warehouse_local_level)
        if warehouse_backorder > 0:
            theta = demand_mean(info_input.demand_feature[rdc]) / demand_sum
            begin_ii, end_ii = stats.binom.ppf(0.01, warehouse_backorder, theta), stats.binom.ppf(0.99, warehouse_backorder, theta)
            for backorder in np.arange(begin_ii, end_ii + 1):
                p2 = stats.binom.pmf(backorder, warehouse_backorder, theta)
                for retailer_demand in np.arange(begin_i, end_i + 1):
                    p3 = stats.poisson.pmf(retailer_demand, para_i)
                    retailer_level = backorder + retailer_demand - current_si
                    b_level = max(0, retailer_level)
                    i_level = -1 * min(0, retailer_level)
                    total_cost += (hi * i_level + bi * b_level) * p1 * p2 * p3
        else:
            p2 = 1.
            backorder = 0.
            for retailer_demand in np.arange(begin_i, end_i + 1):
                p3 = stats.poisson.pmf(retailer_demand, para_i)
                retailer_level = backorder + retailer_demand - current_si
                b_level = max(0, retailer_level)
                i_level = -1 * min(0, retailer_level)
                total_cost += (hi * i_level + bi * b_level) * p1 * p2 * p3
    return total_cost


def cost_warehouse_binomial_backorder(info_input: OWMR_input, x, order_up_to_level):
    if x >= order_up_to_level[0]:
        return info_input.holding_cost[0] * x + sum([cost_function_retailers(info_input, k, order_up_to_level[k]) for k in info_input.rdc_dict.keys()] )
    else:
        backlog = int(order_up_to_level[0] - x)
        demand_sum = sum([demand_mean(info_input.demand_feature[k]) for k in info_input.rdc_dict.keys()])
        cost_retailers = 0
        for k in info_input.rdc_dict.keys():
            theta = demand_mean(info_input.demand_feature[k]) / demand_sum
            begin, end = stats.binom.ppf(0.01, backlog, theta), stats.binom.ppf(0.99, backlog, theta)
            
            for n in np.arange(begin, end + 1):
                level = order_up_to_level[k] - n
                cost_retailers += stats.binom.pmf(n, backlog, theta) * cost_function_retailers(info_input, k, level)

        return info_input.holding_cost[0] * x + cost_retailers
    

def cost_warehouse_ro(info_input: OWMR_input, x, order_up_to_level):
    para = _warehouse_incremental_poisson_mean(info_input)
    begin, end = stats.poisson.ppf(0.01, para), stats.poisson.ppf(0.99, para)
    return np.sum([stats.poisson.pmf(n, para) * cost_warehouse_binomial_backorder(info_input, x - n, order_up_to_level) for n in np.arange(begin, end + 1)])


RO_TAIL_FIXED_PROB = 1e-8


def _poisson_range_tail_fixed(para, tail_prob=RO_TAIL_FIXED_PROB):
    begin = max(0, int(math.floor(stats.poisson.ppf(tail_prob, para))))
    end = int(math.ceil(stats.poisson.ppf(1.0 - tail_prob, para)))
    return begin, max(begin, end)


def _binom_range_tail_fixed(n, theta, tail_prob=RO_TAIL_FIXED_PROB):
    n = int(round(max(0, n)))
    theta = float(theta)
    if n <= 0 or theta <= 0:
        return 0, 0
    if theta >= 1:
        return n, n
    begin = int(math.floor(stats.binom.ppf(tail_prob, n, theta)))
    end = int(math.ceil(stats.binom.ppf(1.0 - tail_prob, n, theta)))
    begin = max(0, min(n, begin))
    end = max(begin, min(n, end))
    return begin, end


def recursive_heuristic_tail_fixed(info_input: OWMR_input, tail_prob=RO_TAIL_FIXED_PROB):
    rdcs = list(info_input.rdc_dict.keys())
    order_up_to_level = {0: 0.0}
    for j in rdcs:
        cdf_quantile = (info_input.backorder_cost[j] + info_input.holding_cost[0]) / (
            info_input.backorder_cost[j] + info_input.holding_cost[j]
        )
        para = info_input.Poisson_para[j]
        order_up_to_level[j] = stats.poisson.ppf(min(0.99, cdf_quantile), para)
        order_up_to_level[0] += order_up_to_level[j]

    warehouse_para = _warehouse_incremental_poisson_mean(info_input)
    current_traverse = order_up_to_level[0] + warehouse_para
    best_cost = float("inf")
    traverse_interval = max(1, int(current_traverse / 1000))
    while True:
        current_cost = cost_warehouse_ro_tail_fixed(
            info_input,
            current_traverse,
            order_up_to_level,
            tail_prob=tail_prob,
        )
        if current_cost > best_cost:
            break
        best_cost = current_cost
        current_traverse -= traverse_interval
    best_warehouse_base_stock_level = current_traverse + 1
    best_warehouse_local_level = best_warehouse_base_stock_level - order_up_to_level[0]

    recursive_level = {}
    for k in rdcs:
        recursive_level[k] = cost_retailer_recursive_tail_fixed(
            info_input,
            k,
            best_warehouse_local_level,
            tail_prob=tail_prob,
        )

    best_warehouse_base_stock_level = best_warehouse_local_level + np.sum(list(recursive_level.values()))
    return recursive_level, best_warehouse_local_level, best_warehouse_base_stock_level


def cost_retailer_recursive_tail_fixed(
        info_input: OWMR_input,
        rdc,
        best_warehouse_local_level,
        tail_prob=RO_TAIL_FIXED_PROB,
):
    para = _warehouse_incremental_poisson_mean(info_input)
    para_i = info_input.Poisson_para[rdc]
    hi, bi = info_input.holding_cost[rdc], info_input.backorder_cost[rdc]
    demand_sum = sum([demand_mean(info_input.demand_feature[k]) for k in info_input.rdc_dict.keys()])
    begin, end = _poisson_range_tail_fixed(para, tail_prob=tail_prob)
    begin_i, end_i = _poisson_range_tail_fixed(para_i, tail_prob=tail_prob)

    begin_rdc, end_rdc = begin_i, end_i
    previous_si = int((begin_rdc + end_rdc) / 2)
    previous_cost = calc_total_cost_tail_fixed(
        previous_si,
        info_input,
        rdc,
        best_warehouse_local_level,
        begin,
        end,
        begin_i,
        end_i,
        demand_sum,
        para_i,
        hi,
        bi,
        tail_prob=tail_prob,
    )
    while True:
        if end_rdc == begin_rdc:
            break
        current_si_left = int((begin_rdc + previous_si) / 2)
        current_si_right = int((end_rdc + previous_si) / 2)
        current_cost_left = calc_total_cost_tail_fixed(
            current_si_left,
            info_input,
            rdc,
            best_warehouse_local_level,
            begin,
            end,
            begin_i,
            end_i,
            demand_sum,
            para_i,
            hi,
            bi,
            tail_prob=tail_prob,
        )
        current_cost_right = calc_total_cost_tail_fixed(
            current_si_right,
            info_input,
            rdc,
            best_warehouse_local_level,
            begin,
            end,
            begin_i,
            end_i,
            demand_sum,
            para_i,
            hi,
            bi,
            tail_prob=tail_prob,
        )
        if current_cost_left > previous_cost:
            begin_rdc = current_si_left
        else:
            end_rdc = previous_si
            previous_si = current_si_left
        if current_cost_right > previous_cost:
            end_rdc = current_si_right
        else:
            begin_rdc = previous_si
            previous_si = current_si_right
        previous_cost = min(current_cost_left, current_cost_right, previous_cost)

    return begin_rdc


def calc_total_cost_tail_fixed(
        current_si,
        info_input: OWMR_input,
        rdc,
        best_warehouse_local_level,
        begin,
        end,
        begin_i,
        end_i,
        demand_sum,
        para_i,
        hi,
        bi,
        tail_prob=RO_TAIL_FIXED_PROB,
):
    total_cost = 0.0
    theta = demand_mean(info_input.demand_feature[rdc]) / demand_sum
    retailer_demands = np.arange(int(begin_i), int(end_i) + 1)
    retailer_pmfs = stats.poisson.pmf(retailer_demands, para_i)
    for total_demand in range(int(begin), int(end) + 1):
        p1 = stats.poisson.pmf(
            total_demand,
            _warehouse_incremental_poisson_mean(info_input),
        )
        warehouse_backorder = max(0, total_demand - best_warehouse_local_level)
        if warehouse_backorder > 0:
            begin_ii, end_ii = _binom_range_tail_fixed(
                warehouse_backorder,
                theta,
                tail_prob=tail_prob,
            )
            backorders = np.arange(begin_ii, end_ii + 1)
            backorder_pmfs = stats.binom.pmf(backorders, warehouse_backorder, theta)
            retailer_level = backorders[:, None] + retailer_demands[None, :] - current_si
            b_level = np.maximum(retailer_level, 0.0)
            i_level = np.maximum(-retailer_level, 0.0)
            cost = hi * i_level + bi * b_level
            total_cost += float(p1 * np.sum(cost * backorder_pmfs[:, None] * retailer_pmfs[None, :]))
        else:
            retailer_level = retailer_demands - current_si
            b_level = np.maximum(retailer_level, 0.0)
            i_level = np.maximum(-retailer_level, 0.0)
            cost = hi * i_level + bi * b_level
            total_cost += float(p1 * np.sum(cost * retailer_pmfs))
    return total_cost


def cost_warehouse_binomial_backorder_tail_fixed(
        info_input: OWMR_input,
        x,
        order_up_to_level,
        tail_prob=RO_TAIL_FIXED_PROB,
):
    if x >= order_up_to_level[0]:
        return info_input.holding_cost[0] * x + sum([
            cost_function_retailers(info_input, k, order_up_to_level[k])
            for k in info_input.rdc_dict.keys()
        ])

    backlog = int(order_up_to_level[0] - x)
    demand_sum = sum([demand_mean(info_input.demand_feature[k]) for k in info_input.rdc_dict.keys()])
    cost_retailers = 0.0
    for k in info_input.rdc_dict.keys():
        theta = demand_mean(info_input.demand_feature[k]) / demand_sum
        begin, end = _binom_range_tail_fixed(backlog, theta, tail_prob=tail_prob)
        for n in range(begin, end + 1):
            level = order_up_to_level[k] - n
            cost_retailers += stats.binom.pmf(n, backlog, theta) * cost_function_retailers(info_input, k, level)

    return info_input.holding_cost[0] * x + cost_retailers


def cost_warehouse_ro_tail_fixed(
        info_input: OWMR_input,
        x,
        order_up_to_level,
        tail_prob=RO_TAIL_FIXED_PROB,
):
    para = _warehouse_incremental_poisson_mean(info_input)
    begin, end = _poisson_range_tail_fixed(para, tail_prob=tail_prob)
    return np.sum([
        stats.poisson.pmf(n, para)
        * cost_warehouse_binomial_backorder_tail_fixed(
            info_input,
            x - n,
            order_up_to_level,
            tail_prob=tail_prob,
        )
        for n in range(begin, end + 1)
    ])



### Zipkin Central Control Relaxation
'''
Assumptions
1. 平稳需求，服从泊松分布
2. 仅考虑holdng cost和retailer的backorder cost
'''
'''
demand_feature = {1: 20, 2, 30, 3: 25}
rdc_dict = {1: 2, 2: 3, 3: 3}
holding_cost = {0: 1, 1: 1.2, 2, 1.4, 3: 1.6}
backorder_cost = {1: 10, 2: 10, 3: 10, 4: 10}
'''

def relaxation_based_heuristic(info_input: OWMR_input):
    rdcs = list(info_input.rdc_dict.keys())
    ## Figure out each base stock level $s_j^*$ for retailers
    order_up_to_level = {0: 0.}
    for j in rdcs:
        # Newsvendor quantile
        cdf_quantile = (info_input.backorder_cost[j] + info_input.holding_cost[0]) / (
            info_input.backorder_cost[j] + info_input.holding_cost[j]
        )
        para = info_input.Poisson_para[j]
        order_up_to_level[j] = stats.poisson.ppf(min(0.99, cdf_quantile), para)
        order_up_to_level[0] += order_up_to_level[j]

    best_level_retailer_cost = np.sum([cost_function_retailers(info_input, rdc, order_up_to_level[rdc]) for rdc in info_input.rdc_dict.keys()])
    ## Figure out best base stock level for the warehouse
    warehouse_para = _warehouse_incremental_poisson_mean(info_input)
    current_traverse = order_up_to_level[0] + warehouse_para
    best_cost = 1e5
    traverse_interval = max(1, int(current_traverse / 1000))
    while True:
        current_cost = cost_function_warehouse(info_input, current_traverse, order_up_to_level, best_level_retailer_cost)
        print(current_cost, current_traverse)
        if current_cost > best_cost:
            break
        else:
            best_cost = current_cost
            current_traverse -= traverse_interval
    best_warehouse_base_stock_level = current_traverse + 1
    return best_cost, order_up_to_level, best_warehouse_base_stock_level


def _retailer_lead_time_model(info_input: OWMR_input, rdc):
    return info_input.demand_feature[rdc], info_input.rdc_dict[rdc] + 1


def _distribution_mean_from_pmf(support, pmf):
    return float(np.dot(np.asarray(support, dtype=float), np.asarray(pmf, dtype=float)))


def _pmf_survival_at_least(support, pmf, threshold):
    support = np.asarray(support, dtype=float)
    pmf = np.asarray(pmf, dtype=float)
    if threshold <= support[0]:
        return 1.0
    if threshold > support[-1]:
        return 0.0
    return float(pmf[support >= threshold].sum())


def cost_function_retailers_distribution_aware(info_input: OWMR_input, rdc, x):
    hc = info_input.holding_cost[rdc]
    bc = info_input.backorder_cost[rdc]
    h0 = info_input.holding_cost[0]
    support, pmf = _lead_time_pmf_cached(_retailer_lead_time_model(info_input, rdc))
    mean = _distribution_mean_from_pmf(support, pmf)
    shortage = pmf_expected_shortage(support, pmf, x)
    return (hc - h0) * (x - mean) + (bc + hc) * shortage


def _retailer_distribution_aware_allocation_benefit(info_input: OWMR_input, rdc, current_position):
    cache = getattr(info_input, "_distribution_aware_allocation_benefit_cache", None)
    if cache is None:
        cache = {}
        info_input._distribution_aware_allocation_benefit_cache = cache
    key = (int(rdc), int(round(current_position)))
    if key in cache:
        return cache[key]
    benefit = (
        cost_function_retailers_distribution_aware(info_input, rdc, current_position)
        - cost_function_retailers_distribution_aware(info_input, rdc, current_position + 1)
    )
    cache[key] = float(benefit)
    return cache[key]


def rb_distribution_aware_allocation(info_input: OWMR_input, current_positions, target_levels, out_qty):
    rdcs = list(info_input.rdc_dict.keys())
    action_rdc = np.zeros(len(rdcs), dtype=float)
    positions = {rdc: int(round(current_positions[rdc])) for rdc in rdcs}
    targets = {rdc: int(round(target_levels[rdc])) for rdc in rdcs}
    remaining = int(round(max(0.0, out_qty)))
    deficits = {rdc: max(0, targets[rdc] - positions[rdc]) for rdc in rdcs}
    total_deficit = sum(deficits.values())
    if remaining >= total_deficit:
        for rdc in rdcs:
            action_rdc[rdc - 1] = float(deficits[rdc])
        return action_rdc

    heap = []
    for rdc in rdcs:
        if positions[rdc] < targets[rdc]:
            benefit = _retailer_distribution_aware_allocation_benefit(info_input, rdc, positions[rdc])
            heapq.heappush(heap, (-float(benefit), rdc))

    while remaining > 0 and heap:
        neg_benefit, rdc = heapq.heappop(heap)
        if positions[rdc] >= targets[rdc]:
            continue
        positions[rdc] += 1
        action_rdc[rdc - 1] += 1.0
        remaining -= 1
        if positions[rdc] < targets[rdc]:
            benefit = _retailer_distribution_aware_allocation_benefit(info_input, rdc, positions[rdc])
            heapq.heappush(heap, (-float(benefit), rdc))

    return action_rdc


def distribution_aware_fixed_release_allocation(info_input: OWMR_input, current_positions, out_qty):
    """Minimize relaxed downstream cost for a fixed integer release quantity.

    This is the allocation subproblem used by the direct-search policy.  Unlike
    the RB shortcut above, direct search may choose an aggregate retailer level
    above the sum of the retailers' relaxed minimizers.  Its allocator must
    therefore continue assigning units after an individual relaxed target is
    reached.  Convex separable retailer costs make unit-marginal allocation
    exact for the integer problem.
    """
    rdcs = list(info_input.rdc_dict.keys())
    positions = {rdc: int(round(current_positions[rdc])) for rdc in rdcs}
    rdc_index = {rdc: index for index, rdc in enumerate(rdcs)}
    remaining = int(round(max(0.0, out_qty)))
    action_rdc = np.zeros(len(rdcs), dtype=float)
    if remaining <= 0:
        return action_rdc

    heap = []
    for rdc in rdcs:
        benefit = _retailer_distribution_aware_allocation_benefit(
            info_input, rdc, positions[rdc]
        )
        heapq.heappush(heap, (-float(benefit), rdc))

    while remaining > 0:
        _neg_benefit, rdc = heapq.heappop(heap)
        index = rdc_index[rdc]
        positions[rdc] += 1
        action_rdc[index] += 1.0
        remaining -= 1
        benefit = _retailer_distribution_aware_allocation_benefit(
            info_input, rdc, positions[rdc]
        )
        heapq.heappush(heap, (-float(benefit), rdc))

    return action_rdc


def direct_search_distribution_aware_order_qty(
        env,
        info_input: OWMR_input,
        aggregate_retailer_base_stock_level,
        warehouse_base_stock_level,
):
    """Return one action for a DS policy with fixed ``(S0, Sr)`` levels."""
    positions = {
        rdc: env.state_echelon_rdc[rdc][-1]
        for rdc in env.rdcs.keys()
    }
    aggregate_position = float(sum(positions.values()))
    desired_release = max(
        0.0,
        float(aggregate_retailer_base_stock_level) - aggregate_position,
    )
    available = max(float(env.CDC_inventory[0]), 0.0)
    release = min(desired_release, available)
    action_rdc = distribution_aware_fixed_release_allocation(
        info_input,
        positions,
        release,
    )
    action_cdc = max(
        0.0,
        float(warehouse_base_stock_level) - float(env.state_echelon_cdc[-1]),
    )
    return np.hstack([action_rdc, action_cdc])


RB_RETAILER_QUANTILE_CAP = 0.99


def _retailer_relaxed_base_stock_distribution_aware(info_input: OWMR_input, rdc, quantile_cap=None):
    hc = info_input.holding_cost[rdc]
    bc = info_input.backorder_cost[rdc]
    h0 = info_input.holding_cost[0]
    quantile = (bc + h0) / (bc + hc)
    if quantile_cap is not None:
        quantile = min(float(quantile_cap), quantile)
    quantile = min(max(quantile, 0.0), 1.0 - 1e-10)
    spec, periods = _retailer_lead_time_model(info_input, rdc)
    return int(round(lead_time_demand_quantile(spec, quantile, periods)))


def direct_search_distribution_aware_bounds(
        info_input: OWMR_input,
        retailer_quantile_cap=RB_RETAILER_QUANTILE_CAP,
):
    """Construct the integer ``(S0, Sr)`` search box for periodic-review DS.

    ``Sr`` starts at the RB aggregate relaxed level.  The upper endpoint uses
    each retailer's newsvendor level over the joint supplier-to-retailer
    horizon ``L0 + Lj + 1``.  Their sum also supplies the paper-style upper
    bound for the warehouse echelon level ``S0``.
    """
    _objective, relaxed_levels, rb_warehouse_level = (
        relaxation_based_heuristic_distribution_aware_capped(info_input)
    )
    individual_upper = {}
    for rdc in info_input.rdc_dict:
        hc = float(info_input.holding_cost[rdc])
        bc = float(info_input.backorder_cost[rdc])
        h0 = float(info_input.holding_cost[0])
        quantile = (bc + h0) / (bc + hc)
        if retailer_quantile_cap is not None:
            quantile = min(float(retailer_quantile_cap), quantile)
        periods = int(info_input.item_lead_time + info_input.rdc_dict[rdc] + 1)
        upper = int(round(lead_time_demand_quantile(
            info_input.demand_feature[rdc],
            quantile,
            periods,
        )))
        individual_upper[rdc] = max(
            int(round(relaxed_levels[rdc])),
            upper,
        )

    sr_lower = int(round(relaxed_levels[0]))
    sr_upper = max(sr_lower, int(sum(individual_upper.values())))
    s0_upper = max(int(round(rb_warehouse_level)), sr_upper)
    return {
        "warehouse": (0, s0_upper),
        "aggregate_retailer": (sr_lower, sr_upper),
        "rb_warehouse": int(round(rb_warehouse_level)),
        "rb_aggregate_retailer": sr_lower,
        "relaxed_retailer_levels": {
            rdc: int(round(relaxed_levels[rdc])) for rdc in info_input.rdc_dict
        },
        "individual_upper_levels": individual_upper,
    }


def _build_relaxed_retailer_cost_table(info_input: OWMR_input, order_up_to_level, min_aggregate_position):
    rdcs = list(info_input.rdc_dict.keys())
    sr = int(round(order_up_to_level[0]))
    min_x = int(math.floor(min_aggregate_position))
    base_cost = sum(
        cost_function_retailers_distribution_aware(info_input, rdc, int(round(order_up_to_level[rdc])))
        for rdc in rdcs
    )

    table = {sr: float(base_cost)}
    current_level = {rdc: int(round(order_up_to_level[rdc])) for rdc in rdcs}
    pmf_info = {}
    heap = []
    for rdc in rdcs:
        support, pmf = _lead_time_pmf_cached(_retailer_lead_time_model(info_input, rdc))
        pmf_info[rdc] = (support, pmf)
        hc = info_input.holding_cost[rdc]
        bc = info_input.backorder_cost[rdc]
        h0 = info_input.holding_cost[0]
        h_echelon = hc - h0
        y = current_level[rdc]
        removal_cost = (bc + hc) * _pmf_survival_at_least(support, pmf, y) - h_echelon
        heapq.heappush(heap, (float(removal_cost), rdc))

    running_cost = float(base_cost)
    for x in range(sr - 1, min_x - 1, -1):
        removal_cost, rdc = heapq.heappop(heap)
        running_cost += removal_cost
        current_level[rdc] -= 1
        table[x] = float(running_cost)

        support, pmf = pmf_info[rdc]
        hc = info_input.holding_cost[rdc]
        bc = info_input.backorder_cost[rdc]
        h0 = info_input.holding_cost[0]
        h_echelon = hc - h0
        y = current_level[rdc]
        next_removal_cost = (bc + hc) * _pmf_survival_at_least(support, pmf, y) - h_echelon
        heapq.heappush(heap, (float(next_removal_cost), rdc))

    return table


def relaxation_based_heuristic_distribution_aware(info_input: OWMR_input, retailer_quantile_cap=None):
    rdcs = list(info_input.rdc_dict.keys())
    order_up_to_level = {0: 0.0}
    for rdc in rdcs:
        order_up_to_level[rdc] = _retailer_relaxed_base_stock_distribution_aware(
            info_input,
            rdc,
            quantile_cap=retailer_quantile_cap,
        )
        order_up_to_level[0] += order_up_to_level[rdc]

    aggregate_support, aggregate_pmf = aggregate_lead_time_demand_pmf(
        info_input,
        info_input.item_lead_time,
    )
    aggregate_support = np.asarray(aggregate_support, dtype=float)
    aggregate_pmf = np.asarray(aggregate_pmf, dtype=float)

    sr = int(round(order_up_to_level[0]))
    max_warehouse_demand = int(np.ceil(aggregate_support[-1]))
    min_aggregate_position = -max_warehouse_demand
    relaxed_cost_table = _build_relaxed_retailer_cost_table(
        info_input, order_up_to_level, min_aggregate_position
    )
    cr_at_sr = relaxed_cost_table[sr]

    upper_base_stock = int(max(0, np.ceil(sr + aggregate_support[-1])))
    h0 = info_input.holding_cost[0]
    # For integer y, floor(y - d) == y - ceil(d). The expected relaxed
    # retailer cost over every candidate y is therefore one convolution.
    effective_demand = np.ceil(aggregate_support).astype(int)
    demand_min = int(effective_demand.min())
    demand_max = int(effective_demand.max())
    demand_pmf = np.zeros(demand_max - demand_min + 1, dtype=float)
    np.add.at(demand_pmf, effective_demand - demand_min, aggregate_pmf)

    position_min = min_aggregate_position
    positions = np.arange(position_min, upper_base_stock + 1, dtype=int)
    relaxed_costs = np.empty(positions.size, dtype=float)
    at_or_below_sr = positions <= sr
    relaxed_costs[~at_or_below_sr] = cr_at_sr
    relaxed_costs[at_or_below_sr] = np.fromiter(
        (relaxed_cost_table[int(position)] for position in positions[at_or_below_sr]),
        dtype=float,
        count=int(at_or_below_sr.sum()),
    )

    convolved_costs = fftconvolve(relaxed_costs, demand_pmf, mode="full")
    candidates = np.arange(upper_base_stock + 1, dtype=int)
    convolution_indices = candidates - position_min - demand_min
    expected_retailer_costs = convolved_costs[convolution_indices]
    mean_warehouse_demand = float(np.dot(aggregate_support, aggregate_pmf))
    candidate_costs = h0 * (candidates - mean_warehouse_demand) + expected_retailer_costs
    best_index = int(np.argmin(candidate_costs))
    best_cost = float(candidate_costs[best_index])
    best_warehouse_base_stock_level = float(candidates[best_index])

    return best_cost, order_up_to_level, best_warehouse_base_stock_level


def relaxation_based_heuristic_distribution_aware_capped(
        info_input: OWMR_input,
        retailer_quantile_cap=RB_RETAILER_QUANTILE_CAP,
):
    return relaxation_based_heuristic_distribution_aware(
        info_input,
        retailer_quantile_cap=retailer_quantile_cap,
    )


RB_distribution_aware_uncapped = relaxation_based_heuristic_distribution_aware
RB_distribution_aware_capped = relaxation_based_heuristic_distribution_aware_capped
RB_distribution_aware = relaxation_based_heuristic_distribution_aware_capped


def cost_function_retailers(info_input: OWMR_input, rdc, x):
    hc = info_input.holding_cost[rdc]
    bc = info_input.backorder_cost[rdc]
    h0 = info_input.holding_cost[0]
    para = info_input.Poisson_para[rdc]

    # begin, end = stats.poisson.ppf(0.01, para), stats.poisson.ppf(0.99, para)
    # # 创建n值数组
    # n_values = np.arange(begin, end + 1)
    # # 一次性计算所有概率质量函数值
    # pmf_values = stats.poisson.pmf(n_values, para)

    # 向量化计算第一部分
    part1 = (hc - h0) * (x - para)

    # 使用正态近似
    if para > 10:  # 参数足够大时使用近似
        mean = para
        std = np.sqrt(para)
        z = (x - mean) / std
        # 计算期望值
        # 第二部分: E[max(n - x, 0)]
        expected_shortage = (mean - x) * (1 - stats.norm.cdf(z)) + std * stats.norm.pdf(z)
        part2 = (bc + hc) * expected_shortage
    else:
        # 使用缓存的泊松值
        n_values, pmf_values = get_poisson_values(para)
        # 向量化计算第二部分
        mask = n_values >= x  # 创建掩码，只计算n>=x的部分
        part2 = (bc + hc) * np.sum((n_values[mask] - x) * pmf_values[mask])
    
    return part1 + part2
    # return (hc - h0) * np.sum([(x - n) * stats.poisson.pmf(n, para) for n in np.arange(begin, end + 1)]) + (bc + hc) * np.sum([(n - x) * stats.poisson.pmf(n, para) for n in np.arange(max(x, begin), end + 1)])



from functools import lru_cache


# 缓存泊松分布计算
@lru_cache(maxsize=100)
def get_poisson_values(para):
    begin, end = stats.poisson.ppf(0.01, para), stats.poisson.ppf(0.99, para)
    n_values = np.arange(begin, end + 1)
    pmf_values = stats.poisson.pmf(n_values, para)
    return n_values, pmf_values


def cost_function_warehouse(info_input: OWMR_input, current_order_up_to_level,  order_up_to_level, best_cost):
    para = _warehouse_incremental_poisson_mean(info_input)
    h0 = info_input.holding_cost[0]
    begin, end = stats.poisson.ppf(0.01, para), stats.poisson.ppf(0.99, para)

    return h0 * np.sum([(current_order_up_to_level - n) * stats.poisson.pmf(n, para) + heuristic_allocation(info_input, current_order_up_to_level, n, order_up_to_level, best_cost) * stats.poisson.pmf(n, para) for n in np.arange(begin, end)])


# import itertools
def partition_to_k_parts_iterative(n, k, interval=1):
    # 初始化一个栈来模拟递归
    stack = [(n, k, [])]
    results = []
    
    while stack:
        remaining, parts, current = stack.pop() 
        if parts == 1:
            results.append(current + [remaining])
            continue
        for i in range(0, remaining + 1, interval):
            stack.append((remaining - i, parts - 1, current + [i]))
    
    return results

@timer
def distribute_unmet_stock(total, order_up_to_level):
    num_rdcs = len(order_up_to_level) - 1
    outls = np.array(list(order_up_to_level.values())[1:])
    # 生成所有可能的分配方式
    interval = max(int(num_rdcs), int(total / num_rdcs), 1)
    if num_rdcs > 10 and total > 1000:  # redundant total
        interval  = int(total / 1)
    all_combinations = partition_to_k_parts_iterative(total, num_rdcs, interval)
    # 筛选出总和等于 total 的组合
    valid_combinations = [combo for combo in all_combinations if np.all(outls - combo >= 0)]
    return valid_combinations


def func_allocated(info_input, order_up_to_level, allocated):
    return np.sum([cost_function_retailers(info_input, rdc, order_up_to_level[rdc] - allocated[rdc - 1]) for rdc in info_input.rdc_dict.keys()])


@timer
def heuristic_allocation(info_input: OWMR_input, current_order_up_to_level, warehouse_demand, order_up_to_level, best_cost):

    best_level = order_up_to_level[0]
    if current_order_up_to_level - warehouse_demand >= best_level:
        # logger.info('input order-up-to level minus demand is larger than the best level for the retailers and is not appropriate')
        return best_cost
    delta = int(best_level + warehouse_demand - current_order_up_to_level)
    # warehouse_cost = cost_function_warehouse(info_input, current_order_up_to_level)

    # combination of allocation of unmet stock
    allocated_quantities = distribute_unmet_stock(delta, order_up_to_level)
    # logger.info(f'需要计算{len(allocated_quantities)}组成本')
    best_allocation = []
    best_cost = 1e5
    # for allocated in allocated_quantities:
        # current_cost = np.sum([cost_function_retailers(info_input, rdc, order_up_to_level[rdc] - allocated[rdc - 1]) for rdc in info_input.rdc_dict.keys()])
        # if current_cost < best_cost:
        #     best_cost = current_cost
        #     best_allocation = allocated
    costs = [np.sum([cost_function_retailers(info_input, rdc, order_up_to_level[rdc] - allocated[rdc - 1]) for rdc in info_input.rdc_dict.keys()]) for allocated in allocated_quantities]
    best_cost = min(costs)
    return best_cost


# def main_relaxation_based_heu(info_input: OWMR_input, input_args):
#     best_cost, order_up_to_level, best_warehouse_base_stock_level = relaxation_based_heuristic(info_input)
    
#     logger.info('算法给出的策略为:')
#     logger.info(f'warehouse的发货水位为{order_up_to_level[0]}')
#     logger.info(f'warehouse的订货水位为{best_warehouse_base_stock_level}')
#     logger.info(f'rdc的订货水位为{order_up_to_level}')
#     logger.info(f'成本下界为{best_cost}')

#     # Num_of_Instances_Generated = 100
#     # learning_rate, net_dims, reward_scale, holding_weight, penalty_weight, cost_plus, all_demand_list = auto_hyperparameter(info_input, Num_of_Instances_Generated)

#     # 做一个args赋值，这个后续可以直接在auto_hyperparameter()做成需要的格式
#     # input_args = {}
#     # input_args['all_demand_list'] = all_demand_list
#     # input_args['holding_weight'] = holding_weight
#     # input_args['penalty_weight'] = penalty_weight
#     # input_args['cost_plus'] = cost_plus
#     # input_args['info_input'] = info_input

#     env = generate_env(OWMRInventoryEnv, input_args)
#     env.total_periods = 1000
#     env.reset()
#     reward_process = []
#     eval_times = 100

#     heu_reward, std, median = 0., 0., 0.
#     while True:
#         env.step_satisfy_demand()
#         env.get_echelon_state()

#         action = heuristic_order_qty(env, info_input, order_up_to_level, best_warehouse_base_stock_level, method='RB')

#         _, reward, _, _ = env.step_heu(action, 'heu')
#         reward_process.append(-1 * reward)
#         if len(reward_process) > 300 and len(reward_process) % 100 == 0:
#             current_reward = np.mean(np.array(reward_process[-100:]))
#             past_reward = np.mean(np.array(reward_process[-200: -100]))
#             # past_past_reward = np.sum(reward_process[-300: -200]) / 100
#             logger.info(f'最后100期的平均成本为{current_reward}, 最后100-200期的平均成本为{past_reward}')
#             if abs(current_reward - past_reward) / past_reward < 0.01:
#                 heu_reward = current_reward
#                 std = np.std(np.array(reward_process[-100:]))
#                 median = np.median(np.array(reward_process[-100:]))
#                 break
#     return heu_reward, std, median


def env_input_initialize(rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, RDC_inventory, CDC_inventory):
    state_dim = sum([rdc_dict[k] + 1 for k in rdc_dict.keys()]) + item_lead_time + 1
    action_dim = len(rdc_dict) + 1
    logger.info(f'current data:\n\
                rdc lead time: {rdc_dict}\n\
                demand rate: {demand_feature}\n\
                holding cost: {holding_cost}\n\
                backorder cost: {backorder_cost}\n\
                item lead time: {item_lead_time}\n\
                initail RDC inv: {RDC_inventory}\n\
                initial CDC inv: {CDC_inventory}')

    return OWMR_input(rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time,
    RDC_inventory, CDC_inventory), state_dim, action_dim


MAX_WEEK = 1000

def generate_and_save_instances(input_info: OWMR_input, num_of_instances):
    item_demand_info = {rdc: np.zeros((num_of_instances, MAX_WEEK)) for rdc in input_info.rdc_dict.keys()}
    for rdc in input_info.rdc_dict.keys():
        real_demand = sample_demand(input_info.demand_feature[rdc], size=(num_of_instances, MAX_WEEK))
        # real_demand = np.reshape(real_demand, newshape=(num_of_instances, MAX_WEEK))
        item_demand_info[rdc] = real_demand
    all_demand_dict = item_demand_info
    return all_demand_dict


def auto_hyperparameter(input_info: OWMR_input, num_of_instances):
    # 保存随机生成的demand至excel，留档。
    all_demand_list = generate_and_save_instances(input_info, num_of_instances)

    learning_rate = 1e-3
    net_dims = (64, 128)
    # reward_scale = 2 ** -10
    # test_env = MultiEchelonInventoryEnvWeek(all_demand_list, input_env.item_dict, input_env.rdc_dict,
    #                                         input_env.selling_item, input_env.demand_feature, input_env.CDC_inventory,
    #                                         input_env.RDC_inventory)
    # test_env.reset()
    # holding_c, backorder_c, penalty_c = 0, 0, 0
    # repeat_episode = 10

    # for _ in range(repeat_episode):
    #     while True:
    #         state, reward, done, cost_dict = test_env.step(test_env.generate_action())
    #         holding_c += cost_dict['holding_cost']
    #         backorder_c += cost_dict['backorder_cost']
    #         penalty_c += cost_dict['penalty']

    #         if done:
    #             test_env.reset()
    #             break
    # holding_weight = test_env.holding_weight
    # penalty_weight = test_env.penalty_weight
    # if holding_c > 0.:
    #     times_bh = backorder_c / holding_c
    #     times_ph = penalty_c / holding_c
    #     holding_weight = test_env.holding_weight * times_bh / 10
    #     if times_ph > 0:
    #         penalty_weight = test_env.penalty_weight * times_bh / times_ph / 100

    # avg_reward = (holding_c * holding_weight + backorder_c + penalty_c * penalty_weight) / repeat_episode

    # cost_plus = avg_reward / 5
    # reward_scale_index = round(log2(1000 / avg_reward)) - 1
    holding_weight, penalty_weight, cost_plus = 1., 0., 0.
    reward_scale_index = -5
    reward_scale = 2 ** reward_scale_index

    return learning_rate, net_dims, reward_scale, holding_weight, penalty_weight, cost_plus, all_demand_list


# if __name__ == "__main__":
#     datasets = generate_data()

#     for data in datasets:
#         rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, RDC_inventory, CDC_inventory = data
#         input_env, state_dim, action_dim = env_input_initialize(rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, RDC_inventory, CDC_inventory)

#         heu_cost = main_relaxation_based_heu(input_env)
#         logger.info(f'the cost by RB is {heu_cost}')

if __name__ == '__main__':
    datasets = generate_data()
    # datasets = generate_certain_data()
    # datasets = generate_ratio_data()
    # datasets = generate_data_with_random_para(rdc_num=20, sample_num=10, random_seed=10)
    instance_name = 'dataset'
    instance_no = 0
    # arg_name = f'OWMRInventoryEnv_{instance_name}'
    np.random.seed(0)
    for data in datasets:
        instance_no += 1
        # if instance_no in [10]:
        instance_n = f'{instance_name}_{instance_no}'
        # cwd = f'./PPO_0625_ratio/{instance_n}'

        rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, RDC_inventory, CDC_inventory = data
        input_env, state_dim, action_dim = env_input_initialize(rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, RDC_inventory, CDC_inventory)
        
        Num_of_Instances_Generated = 100
        learning_rate, net_dims, reward_scale, holding_weight, penalty_weight, cost_plus, all_demand_list = auto_hyperparameter(input_env, Num_of_Instances_Generated)
        input_args = {}
        # input_args['initial_inventory_cdc'] = input_args.pop('CDC_inventory', None)
        # input_args['initial_inventory_rdc'] = input_args.pop('RDC_inventory', None)
        input_args['all_demand_list'] = all_demand_list
        input_args['holding_weight'] = holding_weight
        input_args['penalty_weight'] = penalty_weight
        input_args['cost_plus'] = cost_plus
        input_args['info_input'] = input_env
        input_args['if_generate_demand'] = False

        calc_heu_results(input_args, instance_name = instance_n, method='DA', output_path='output_OWMR_0918_newsequence', detail=False)
        # generate certain demand
        # demand_feature = input_env.demand_feature
        # all_demand_list = {k: stats.poisson.rvs(demand_feature[k], size=(Num_of_Instances_Generated, 1000)) for k in demand_feature.keys()}

            # agent_class = [AgentPPO.AgentPPO, AgentA2C.AgentA2C, AgentDDPG.AgentDDPG, AgentTD3.AgentTD3, AgentSAC, AgentConvexSAC.AgentConvexSAC][4]
            # agent_name = agent_class.__name__[5:]
            # instance_name = 'dataset_2'
            # actor_path = f'{cwd}/act.pth'

            # evaluate_single_agent(all_demand_list, input_args, agent_class, actor_path, eval_times=100, instance_name=instance_n)
