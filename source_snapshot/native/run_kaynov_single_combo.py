import argparse
import json
import math
import numpy as np
import torch
from pathlib import Path

from Benchmark import (
    OWMRInventoryEnv,
    auto_hyperparameter,
    compute_adaptive_action_space_overrides,
    env_input_initialize,
)
from dataset import data_kaynov, data_small_rvi
from elegantrl.agents import AgentANSAC, AgentLagANSAC, AgentPenaltyANSAC, AgentRecommendedSAC, AgentRecommendedSACReallocationV1Backup, AgentSAC, AgentSACLeadTimeNstep, AgentSACLogProbFixed, AgentSACLogProbFixedOnlineMinQ, AgentSACMechanism, AgentSACResidualSafe, AgentSACV2
from elegantrl.agents.AgentPPO import AgentPPO, AgentDiscreteMultiPPO
from elegantrl.agents.AgentTD3 import AgentTD3
from elegantrl.train.config import Config, configure_training_evaluation
from elegantrl.train.run import train_agent
from elegantrl.train.rb_bootstrap import (
    add_rb_bootstrap_arguments,
    configure_rb_bootstrap,
)
from elegantrl.train.checkpoint_validation import (
    add_checkpoint_validation_arguments,
    configure_checkpoint_validation,
)
from main_DRL_agents import get_env_args
from output_paths import output_path
from utils import my_mkdir


AGENTS = {
    "recommended": AgentRecommendedSAC,
    "recommended_v1": AgentRecommendedSACReallocationV1Backup,
    "an_sac": AgentANSAC,
    "penalty_an_sac": AgentPenaltyANSAC,
    "lag_an_sac": AgentLagANSAC,
    "penalty_pa_sac": AgentPenaltyANSAC,
    "lag_pa_sac": AgentLagANSAC,
    "sac": AgentSAC,
    "sac_leadtime_nstep": AgentSACLeadTimeNstep,
    "sac_residual_safe": AgentSACResidualSafe,
    "sac_mechanism": AgentSACMechanism,
    "sac_logprob_fixed": AgentSACLogProbFixed,
    "sac_logprob_fixed_online_min": AgentSACLogProbFixedOnlineMinQ,
    "sac_v2": AgentSACV2,
    "ppo_continuous": AgentPPO,
    "ppo": AgentDiscreteMultiPPO,
    "discrete_multi_ppo": AgentDiscreteMultiPPO,
    "td3": AgentTD3,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train one SAC/RecommendedSAC combo on a data_kaynov instance.")
    parser.add_argument("--agent", choices=AGENTS.keys(), default="recommended")
    parser.add_argument("--kaynov-index", type=int, default=0)
    parser.add_argument(
        "--small-rvi-index",
        type=int,
        default=None,
        help="Use data_small_rvi()[index] instead of data_kaynov()[kaynov_index].",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--alpha-lr", type=float, default=None)
    parser.add_argument("--reward-scale", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument(
        "--critic-n-step",
        type=int,
        default=1,
        help="Critic return horizon for the opt-in sac_leadtime_nstep agent.",
    )
    parser.add_argument("--batch", type=str, required=True)
    parser.add_argument("--break-step", type=int, required=True)
    parser.add_argument("--eval-per-step", type=int, required=True)
    parser.add_argument(
        "--training-eval-profile",
        choices=("short", "legacy", "custom"),
        default="short",
    )
    parser.add_argument("--eval-times", type=int, default=20)
    parser.add_argument("--eval-episode-steps", type=int, default=300)
    parser.add_argument("--eval-burn-in-steps", type=int, default=150)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "auto", "cuda", "mps"), default="cpu")
    parser.add_argument("--rollout-device", choices=("cpu", "auto", "cuda", "mps"), default=None)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--net-dims", type=str, default="64,32")
    parser.add_argument("--actor-net-dims", type=parse_net_dims, default=None)
    parser.add_argument("--critic-net-dims", type=parse_net_dims, default=None)
    parser.add_argument("--critic-ensemble-size", type=int, default=2)
    parser.add_argument("--critic-target-min-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--repeat-times", type=float, default=None)
    parser.add_argument("--horizon-len", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--soft-update-tau", type=float, default=None)
    parser.add_argument("--clip-grad-norm", type=float, default=None)
    parser.add_argument("--replay-warmup-steps", type=int, default=None)
    parser.add_argument("--freeze-state-normalization-from-warmup", action="store_true")
    parser.add_argument(
        "--replay-sampler",
        choices=("uniform", "paired_uniform", "inventory_stratified"),
        default="uniform",
    )
    parser.add_argument("--replay-focus-fraction", type=float, default=0.25)
    parser.add_argument("--replay-sampler-refresh-updates", type=int, default=10)
    parser.add_argument("--replay-strata-rdc-lead-times", type=parse_net_dims, default=None)
    parser.add_argument("--replay-strata-cdc-lead-time", type=int, default=None)
    add_rb_bootstrap_arguments(parser)
    add_checkpoint_validation_arguments(parser)
    parser.add_argument("--target-entropy", type=float, default=None)
    parser.add_argument(
        "--save-replay-buffer",
        action="store_true",
        help="Persist the off-policy replay buffer for an exact later resume.",
    )
    parser.add_argument(
        "--save-best-training-state",
        action="store_true",
        help="Save a resumable full agent and replay-buffer snapshot whenever evaluation improves.",
    )
    parser.add_argument("--lambda-gae-adv", type=float, default=None)
    parser.add_argument("--lambda-entropy", type=float, default=None)
    parser.add_argument("--ratio-clip", type=float, default=None)
    parser.add_argument("--policy-activation", choices=("relu", "tanh", "gelu", "elu"), default=None)
    parser.add_argument("--explore-noise-std", type=float, default=None)
    parser.add_argument("--policy-noise-std", type=float, default=None)
    parser.add_argument("--update-freq", type=int, default=None)
    parser.add_argument("--an-penalty-coef", "--pa-penalty-coef", dest="an_penalty_coef", type=float, default=None)
    parser.add_argument("--an-penalty-power", "--pa-penalty-power", dest="an_penalty_power", type=float, default=None)
    parser.add_argument("--an-violation-eps", "--pa-violation-eps", dest="an_violation_eps", type=float, default=None)
    parser.add_argument("--an-positive-tau", "--pa-positive-tau", dest="an_positive_tau", type=float, default=None)
    parser.add_argument("--lag-lambda-init", type=float, default=None)
    parser.add_argument("--lag-lambda-lr", type=float, default=None)
    parser.add_argument("--lag-delta", type=float, default=None)
    parser.add_argument("--lag-lambda-max", type=float, default=None)
    parser.add_argument("--lag-update-interval", type=int, default=None)
    parser.add_argument("--paper-preset", choices=("none", "kaynov2024"), default="none")
    parser.add_argument("--resume-from-cwd", type=str, default=None)
    parser.add_argument("--resume-extra-step", type=int, default=None)
    parser.add_argument(
        "--stationary-action-mask-protocol",
        choices=("auto", "masked", "legacy"),
        default="auto",
        help=(
            "State protocol for a stationary case with collapsed action dimensions. "
            "Auto preserves the established unmasked protocol for fresh runs and "
            "infers the protocol from the saved actor input dimension for resumed runs."
        ),
    )
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--allocation-rule", type=str, default="prioritized_greedy")
    parser.add_argument("--demand-fulfillment", type=str, default="partial_backorder")
    parser.add_argument("--emergency-shipment-prob", type=float, default=None)
    parser.add_argument("--rdc-bsl-min-override", type=parse_bound_override, default=None)
    parser.add_argument("--cdc-bsl-min-override", type=float, default=None)
    parser.add_argument("--rdc-bsl-max-override", type=parse_bound_override, default=None)
    parser.add_argument("--cdc-bsl-max-override", type=float, default=None)
    parser.add_argument(
        "--adaptive-action-space",
        action="store_true",
        help="Use distribution-aware lead-time demand quantiles to set continuous-action BSL upper bounds.",
    )
    parser.add_argument("--adaptive-action-mode", choices=("cost_aware", "fixed"), default="cost_aware")
    parser.add_argument("--adaptive-action-quantile", type=float, default=0.92)
    parser.add_argument("--adaptive-action-cost-margin", type=float, default=0.03)
    parser.add_argument("--adaptive-action-min-quantile", type=float, default=0.85)
    parser.add_argument("--adaptive-action-max-quantile", type=float, default=0.95)
    parser.add_argument("--adaptive-cdc-multiplier", type=float, default=1.0)
    parser.add_argument("--adaptive-cdc-buffer-alpha", type=float, default=0.0)
    parser.add_argument(
        "--adaptive-cdc-buffer-mode",
        choices=("required", "legacy_scaled"),
        default="required",
        help=(
            "In required mode, always include CDC lead-time demand and use alpha as extra margin. "
            "Use legacy_scaled only to reproduce historical action bounds."
        ),
    )
    parser.add_argument(
        "--adaptive-cdc-horizon-mode",
        choices=("additive", "joint"),
        default="joint",
        help=(
            "Use joint (default) to protect one L0+Lk+1 demand horizon at the CDC; "
            "use additive only to reproduce the historical sum of two horizons."
        ),
    )
    parser.add_argument(
        "--adaptive-cdc-upper-tail-retention",
        type=float,
        default=0.25,
        help=(
            "Fraction of residual upper-tail probability retained above the CDC "
            "operating quantile when constructing the joint-horizon search envelope."
        ),
    )
    parser.add_argument(
        "--adaptive-cdc-upper-quantile-cap",
        type=float,
        default=0.99,
        help="Maximum quantile used for the joint-horizon CDC action upper bound.",
    )
    parser.add_argument("--adaptive-cdc-min-span", type=float, default=None)
    parser.add_argument("--adaptive-cdc-cost-margin", type=float, default=None)
    parser.add_argument("--adaptive-cdc-min-quantile", type=float, default=None)
    parser.add_argument("--adaptive-cdc-max-quantile", type=float, default=None)
    parser.add_argument("--adaptive-rdc-lower-fraction", "--adaptive-rdc-lower-frac", type=float, default=0.0)
    parser.add_argument("--adaptive-cdc-lower-fraction", "--adaptive-cdc-lower-frac", type=float, default=0.25)
    parser.add_argument("--adaptive-rdc-lower-quantile", type=float, default=None)
    parser.add_argument("--adaptive-cdc-lower-quantile", type=float, default=None)
    parser.add_argument("--adaptive-small-case-beta", type=float, default=0.0)
    parser.add_argument("--adaptive-small-case-threshold", type=int, default=3)
    parser.add_argument("--adaptive-rdc-round-to", type=float, default=1.0)
    parser.add_argument("--adaptive-cdc-round-to", type=float, default=1.0)
    parser.add_argument(
        "--if-generate-demand",
        action="store_true",
        help="Generate fresh stochastic demand during environment steps instead of cycling pre-generated demand traces.",
    )
    parser.add_argument(
        "--kaynov-k5-compatible",
        action="store_true",
        help="For data_kaynov()[5], align initial inventory, floor demand, and MultiDiscrete branch sizes with Kaynov scenario 13.",
    )
    parser.add_argument(
        "--kaynov-k5-state-demand-compatible",
        action="store_true",
        help=(
            "For data_kaynov()[5], align initial inventory and floor demand with Kaynov scenario 13 "
            "while keeping this project's original quantile-based discrete action space."
        ),
    )
    parser.add_argument("--run-tag", type=str, default="original_eval")
    parser.add_argument("--cwd-name", type=str, default=None)
    return parser.parse_args()


def parse_bound_override(text):
    """Parse a scalar or a JSON object for RDC-specific BSL bounds."""
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    if raw.startswith("{"):
        data = json.loads(raw)
        return {int(k): float(v) for k, v in data.items()}
    return float(raw)


def infer_saved_actor_state_dim(resume_cwd: str | Path) -> int:
    """Read a saved actor's input dimension without replacing a live network."""
    resume_cwd = Path(resume_cwd)
    training_state_path = resume_cwd / "agent_training_state.pth"
    if training_state_path.is_file():
        payload = torch.load(training_state_path, map_location="cpu", weights_only=False)
        state_dict = (
            payload.get("attributes", {}).get("act", {}).get("state_dict", {})
        )
    else:
        actor_path = resume_cwd / "act.pth"
        if not actor_path.is_file():
            raise FileNotFoundError(
                f"Cannot infer resume actor state dimension; missing {training_state_path} "
                f"and {actor_path}."
            )
        actor = torch.load(actor_path, map_location="cpu", weights_only=False)
        state_dict = actor.state_dict() if hasattr(actor, "state_dict") else actor

    for key in ("state_avg", "state_std"):
        value = state_dict.get(key)
        if value is not None and value.ndim == 1:
            return int(value.shape[0])
    weight = state_dict.get("net_s.0.weight")
    if weight is not None and weight.ndim == 2:
        return int(weight.shape[1])
    raise ValueError(f"Cannot infer actor input dimension from {resume_cwd}.")


def resolve_stationary_action_mask_protocol(cli, input_args: dict) -> dict:
    """Preserve the standard state protocol or match a resumed checkpoint."""
    protocol = str(cli.stationary_action_mask_protocol)
    if protocol in {"legacy", "masked"}:
        input_args["include_stationary_action_mask_in_state"] = protocol == "masked"
        return {"selected": protocol, "saved_state_dim": None}
    if cli.resume_from_cwd is None:
        input_args["include_stationary_action_mask_in_state"] = False
        return {"selected": "legacy", "saved_state_dim": None}

    saved_state_dim = infer_saved_actor_state_dim(cli.resume_from_cwd)
    masked_probe = OWMRInventoryEnv(
        **{**input_args, "include_stationary_action_mask_in_state": True}
    )
    masked_state_dim = int(masked_probe.state_dim)
    legacy_state_dim = masked_state_dim
    if masked_probe.action_mask_state_start is not None:
        legacy_state_dim -= int(masked_probe.action_mask_dim)

    if saved_state_dim == masked_state_dim:
        selected = "masked"
    elif saved_state_dim == legacy_state_dim:
        selected = "legacy"
    else:
        raise ValueError(
            "Resume actor state dimension is incompatible with both stationary protocols: "
            f"saved={saved_state_dim}, legacy={legacy_state_dim}, masked={masked_state_dim}, "
            f"resume_cwd={cli.resume_from_cwd}."
        )
    input_args["include_stationary_action_mask_in_state"] = selected == "masked"
    return {
        "selected": selected,
        "saved_state_dim": saved_state_dim,
        "legacy_state_dim": legacy_state_dim,
        "masked_state_dim": masked_state_dim,
    }


def parse_net_dims(text):
    dims = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not dims:
        raise ValueError("--net-dims must contain at least one integer")
    return dims


def apply_kaynov2024_paper_preset(args, cli):
    """Apply Table B.6 PPO hyperparameters from Kaynov et al. (2024)."""
    if cli.paper_preset != "kaynov2024":
        return
    args.learning_rate = 1e-4 if cli.lr is None else cli.lr
    args.reward_scale = 1.0 if cli.reward_scale is None else cli.reward_scale
    args.gamma = 0.95 if cli.gamma is None else cli.gamma
    args.net_dims = (64, 64) if cli.net_dims == "64,32" else parse_net_dims(cli.net_dims)
    args.batch_size = 32768 if cli.batch_size is None else cli.batch_size
    args.horizon_len = 327680 if cli.horizon_len is None else cli.horizon_len
    args.repeat_times = 30.0 if cli.repeat_times is None else cli.repeat_times
    args.clip_grad_norm = 40.0 if cli.clip_grad_norm is None else cli.clip_grad_norm
    args.lambda_gae_adv = 0.98 if cli.lambda_gae_adv is None else cli.lambda_gae_adv
    args.lambda_entropy = 0.0 if cli.lambda_entropy is None else cli.lambda_entropy
    args.policy_activation = "tanh" if cli.policy_activation is None else cli.policy_activation


def _raw_kaynov_mean(spec):
    if isinstance(spec, (int, float)):
        return float(spec)
    dist_name = str(spec[0]).lower()
    if dist_name in {"poisson", "normal"}:
        return float(spec[1])
    if dist_name == "uniform":
        return 0.5 * (float(spec[1]) + float(spec[2]))
    raise ValueError(f"Unsupported demand specification for Kaynov compatibility: {spec!r}")


def _sample_kaynov_floor_demand(spec, size):
    if isinstance(spec, (int, float)):
        samples = np.random.poisson(float(spec), size=size)
    else:
        dist_name = str(spec[0]).lower()
        if dist_name == "poisson":
            samples = np.random.poisson(float(spec[1]), size=size)
        elif dist_name == "normal":
            mean = float(spec[1])
            sigma = math.sqrt(max(float(spec[2]), 0.0))
            samples = np.random.normal(mean, sigma, size=size)
        elif dist_name == "uniform":
            samples = np.random.uniform(float(spec[1]), float(spec[2]), size=size)
        else:
            raise ValueError(f"Unsupported demand specification for Kaynov compatibility: {spec!r}")
    return np.clip(np.floor(samples), 0, 500).astype(int)


def generate_kaynov_floor_instances(input_env, num_of_instances, max_week=1000):
    return {
        rdc: _sample_kaynov_floor_demand(input_env.demand_feature[rdc], (num_of_instances, max_week))
        for rdc in input_env.rdc_dict.keys()
    }


def apply_kaynov_k5_compatibility(cli, rdc_dict, demand_feature, item_lead_time):
    if not (cli.kaynov_k5_compatible or cli.kaynov_k5_state_demand_compatible):
        return None, None
    if cli.kaynov_index != 5:
        raise ValueError(
            "--kaynov-k5-compatible and --kaynov-k5-state-demand-compatible are intended "
            "for data_kaynov()[5] / Kaynov scenario 13."
        )

    rdc_inventory = {}
    for rdc, lead_time in rdc_dict.items():
        start_level = int(_raw_kaynov_mean(demand_feature[rdc]) * lead_time)
        rdc_inventory[rdc] = np.hstack((start_level, np.zeros(lead_time)))

    warehouse_start = int(item_lead_time * sum(_raw_kaynov_mean(demand_feature[k]) for k in rdc_dict))
    cdc_inventory = np.hstack((warehouse_start, np.zeros(item_lead_time)))

    discrete_action_lengths = None
    if cli.kaynov_k5_compatible:
        # Kaynov scenario 13 uses MultiDiscrete([101, 26, ..., 26]) in warehouse-first order.
        # Our environment keeps the existing RDC-first, CDC-last action convention.
        discrete_action_lengths = [26 for _ in rdc_dict] + [101]
    return (rdc_inventory, cdc_inventory), discrete_action_lengths


def main():
    cli = parse_args()
    if cli.paper_preset == "none" and (cli.lr is None or cli.reward_scale is None or cli.gamma is None):
        raise ValueError("--lr, --reward-scale, and --gamma are required unless --paper-preset kaynov2024 is used.")
    if cli.small_rvi_index is not None:
        if cli.kaynov_k5_compatible or cli.kaynov_k5_state_demand_compatible:
            raise ValueError("Kaynov K5 compatibility flags cannot be used with --small-rvi-index.")
        data = data_small_rvi()[cli.small_rvi_index]
        instance_name = f"data_small_rvi_{cli.small_rvi_index}"
        instance_index_tag = f"small{cli.small_rvi_index}"
    else:
        data = data_kaynov()[cli.kaynov_index]
        instance_name = f"data_kaynov_{cli.kaynov_index}"
        instance_index_tag = f"idx{cli.kaynov_index}"
    rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, RDC_inventory, CDC_inventory = data
    compatible_inventory, discrete_action_lengths = apply_kaynov_k5_compatibility(
        cli, rdc_dict, demand_feature, item_lead_time
    )
    if compatible_inventory is not None:
        RDC_inventory, CDC_inventory = compatible_inventory
    input_env, _, _ = env_input_initialize(
        rdc_dict,
        demand_feature,
        holding_cost,
        backorder_cost,
        item_lead_time,
        RDC_inventory,
        CDC_inventory,
    )
    _, _, _, holding_weight, penalty_weight, cost_plus, all_demand_list = auto_hyperparameter(input_env, 100)
    if cli.kaynov_k5_compatible or cli.kaynov_k5_state_demand_compatible:
        all_demand_list = generate_kaynov_floor_instances(input_env, 100)
    input_args = {
        "all_demand_list": all_demand_list,
        "holding_weight": holding_weight,
        "penalty_weight": penalty_weight,
        "cost_plus": cost_plus,
        "info_input": input_env,
        "if_generate_demand": cli.if_generate_demand,
        "rho": cli.rho,
        "allocation_rule": cli.allocation_rule,
        "demand_fulfillment": cli.demand_fulfillment,
        "emergency_shipment_prob": cli.emergency_shipment_prob,
        "rdc_bsl_min_override": cli.rdc_bsl_min_override,
        "cdc_bsl_min_override": cli.cdc_bsl_min_override,
        "rdc_bsl_max_override": cli.rdc_bsl_max_override,
        "cdc_bsl_max_override": cli.cdc_bsl_max_override,
    }
    adaptive_rdc_bsl_min = None
    adaptive_cdc_bsl_min = None
    adaptive_rdc_bsl_max = None
    adaptive_cdc_bsl_max = None
    adaptive_rdc_quantiles = None
    if cli.adaptive_action_space:
        (
            adaptive_rdc_bsl_min,
            adaptive_cdc_bsl_min,
            adaptive_rdc_bsl_max,
            adaptive_cdc_bsl_max,
            adaptive_rdc_quantiles,
        ) = compute_adaptive_action_space_overrides(
            input_env,
            mode=cli.adaptive_action_mode,
            quantile=cli.adaptive_action_quantile,
            cost_margin=cli.adaptive_action_cost_margin,
            min_quantile=cli.adaptive_action_min_quantile,
            max_quantile=cli.adaptive_action_max_quantile,
            cdc_multiplier=cli.adaptive_cdc_multiplier,
            cdc_buffer_alpha=cli.adaptive_cdc_buffer_alpha,
            cdc_buffer_mode=cli.adaptive_cdc_buffer_mode,
            cdc_horizon_mode=cli.adaptive_cdc_horizon_mode,
            cdc_upper_tail_retention=cli.adaptive_cdc_upper_tail_retention,
            cdc_upper_quantile_cap=cli.adaptive_cdc_upper_quantile_cap,
            cdc_cost_margin=cli.adaptive_cdc_cost_margin,
            cdc_min_quantile=cli.adaptive_cdc_min_quantile,
            cdc_max_quantile=cli.adaptive_cdc_max_quantile,
            small_case_beta=cli.adaptive_small_case_beta,
            small_case_threshold=cli.adaptive_small_case_threshold,
            rdc_round_to=cli.adaptive_rdc_round_to,
            cdc_round_to=cli.adaptive_cdc_round_to,
            rdc_lower_fraction=cli.adaptive_rdc_lower_fraction,
            cdc_lower_fraction=cli.adaptive_cdc_lower_fraction,
            rdc_lower_quantile=cli.adaptive_rdc_lower_quantile,
            cdc_lower_quantile=cli.adaptive_cdc_lower_quantile,
            return_bounds=True,
            return_quantiles=True,
        )
        if cli.rdc_bsl_min_override is None and adaptive_rdc_bsl_min is not None:
            input_args["rdc_bsl_min_override"] = adaptive_rdc_bsl_min
        if cli.cdc_bsl_min_override is None and adaptive_cdc_bsl_min is not None:
            input_args["cdc_bsl_min_override"] = adaptive_cdc_bsl_min
        if cli.rdc_bsl_max_override is None:
            input_args["rdc_bsl_max_override"] = adaptive_rdc_bsl_max
        if cli.cdc_bsl_max_override is None:
            input_args["cdc_bsl_max_override"] = adaptive_cdc_bsl_max
        effective_cdc_min = input_args.get("cdc_bsl_min_override")
        effective_cdc_max = input_args.get("cdc_bsl_max_override")
        if effective_cdc_min is not None and effective_cdc_max is not None:
            min_span = cli.adaptive_cdc_min_span
            if min_span is None:
                min_span = max(float(cli.adaptive_cdc_round_to), 1e-6)
            if float(effective_cdc_max) - float(effective_cdc_min) < float(min_span) - 1e-9:
                raise ValueError(
                    "Adaptive CDC action range collapsed: "
                    f"lower={effective_cdc_min}, upper={effective_cdc_max}, required_span={min_span}. "
                    "Increase the CDC buffer or provide explicit CDC bounds."
                )
    if discrete_action_lengths is not None:
        input_args["discrete_action_lengths"] = discrete_action_lengths
    stationary_mask_protocol = resolve_stationary_action_mask_protocol(cli, input_args)

    agent_class = AGENTS[cli.agent]
    if agent_class.__name__.split(".")[-1] == "AgentDiscreteMultiPPO":
        input_args["method"] = "paper"

    env_args = get_env_args(input_args)
    args = Config(agent_class, env_class=OWMRInventoryEnv, env_args=env_args)
    args.gamma = cli.gamma if cli.gamma is not None else args.gamma
    args.critic_n_step = int(cli.critic_n_step)
    if cli.agent == "sac_leadtime_nstep" and args.critic_n_step < 2:
        raise ValueError("--agent sac_leadtime_nstep requires --critic-n-step >= 2")
    args.learning_rate = cli.lr if cli.lr is not None else args.learning_rate
    if cli.alpha_lr is not None:
        args.alpha_learning_rate = cli.alpha_lr
    args.reward_scale = cli.reward_scale if cli.reward_scale is not None else args.reward_scale
    args.net_dims = parse_net_dims(cli.net_dims)
    args.actor_net_dims = cli.actor_net_dims
    args.critic_net_dims = cli.critic_net_dims
    args.critic_ensemble_size = cli.critic_ensemble_size
    args.critic_target_min_size = cli.critic_target_min_size
    if cli.agent == "sac_mechanism":
        if args.critic_ensemble_size < 2:
            raise ValueError("--critic-ensemble-size must be at least two")
        if not 1 <= args.critic_target_min_size <= args.critic_ensemble_size:
            raise ValueError(
                "--critic-target-min-size must be in [1, critic-ensemble-size]"
            )
    args.break_step = cli.break_step
    configure_training_evaluation(
        args,
        cli.training_eval_profile,
        eval_per_step=cli.eval_per_step,
        eval_times=cli.eval_times,
        eval_episode_steps=cli.eval_episode_steps,
        eval_burn_in_steps=cli.eval_burn_in_steps,
    )
    args.num_threads = cli.num_threads
    args.random_seed = cli.random_seed
    args.if_save_buffer = cli.save_replay_buffer
    args.if_save_best_training_state = cli.save_best_training_state
    args.device = cli.device
    args.rollout_device = cli.rollout_device
    args.gpu_id = -1 if cli.device == "cpu" else cli.gpu_id

    if cli.batch_size is not None:
        args.batch_size = cli.batch_size
    if cli.repeat_times is not None:
        args.repeat_times = cli.repeat_times
    if cli.horizon_len is not None:
        args.horizon_len = cli.horizon_len
    if cli.buffer_size is not None:
        args.buffer_size = cli.buffer_size
    if cli.soft_update_tau is not None:
        args.soft_update_tau = cli.soft_update_tau
    if cli.clip_grad_norm is not None:
        args.clip_grad_norm = cli.clip_grad_norm
    if cli.replay_warmup_steps is not None:
        args.replay_warmup_steps = cli.replay_warmup_steps
    args.freeze_state_normalization_from_warmup = (
        cli.freeze_state_normalization_from_warmup
    )
    args.replay_sampler = cli.replay_sampler
    args.replay_focus_fraction = cli.replay_focus_fraction
    args.replay_sampler_refresh_updates = cli.replay_sampler_refresh_updates
    args.replay_strata_rdc_lead_times = cli.replay_strata_rdc_lead_times
    args.replay_strata_cdc_lead_time = cli.replay_strata_cdc_lead_time
    configure_rb_bootstrap(args, cli)
    configure_checkpoint_validation(args, cli)
    if cli.target_entropy is not None:
        args.target_entropy = cli.target_entropy
    if cli.lambda_gae_adv is not None:
        args.lambda_gae_adv = cli.lambda_gae_adv
    if cli.lambda_entropy is not None:
        args.lambda_entropy = cli.lambda_entropy
    if cli.ratio_clip is not None:
        args.ratio_clip = cli.ratio_clip
    if cli.policy_activation is not None:
        args.policy_activation = cli.policy_activation
    if cli.explore_noise_std is not None:
        args.explore_noise_std = cli.explore_noise_std
    if cli.policy_noise_std is not None:
        args.policy_noise_std = cli.policy_noise_std
    if cli.update_freq is not None:
        args.update_freq = cli.update_freq
    if cli.an_penalty_coef is not None:
        args.an_penalty_coef = cli.an_penalty_coef
        args.pa_penalty_coef = cli.an_penalty_coef
    if cli.an_penalty_power is not None:
        args.an_penalty_power = cli.an_penalty_power
        args.pa_penalty_power = cli.an_penalty_power
    if cli.an_violation_eps is not None:
        args.an_violation_eps = cli.an_violation_eps
        args.pa_violation_eps = cli.an_violation_eps
    if cli.an_positive_tau is not None:
        args.an_positive_tau = cli.an_positive_tau
        args.pa_positive_tau = cli.an_positive_tau
    if cli.lag_lambda_init is not None:
        args.lag_lambda_init = cli.lag_lambda_init
    if cli.lag_lambda_lr is not None:
        args.lag_lambda_lr = cli.lag_lambda_lr
    if cli.lag_delta is not None:
        args.lag_delta = cli.lag_delta
    if cli.lag_lambda_max is not None:
        args.lag_lambda_max = cli.lag_lambda_max
    if cli.lag_update_interval is not None:
        args.lag_update_interval = cli.lag_update_interval
    if cli.agent == "an_sac":
        args.an_penalty_coef = 0.0
        args.pa_penalty_coef = 0.0
    apply_kaynov2024_paper_preset(args, cli)

    if cli.resume_from_cwd is not None:
        resume_cwd = Path(cli.resume_from_cwd)
        if not resume_cwd.exists():
            raise FileNotFoundError(f"--resume-from-cwd does not exist: {resume_cwd}")
        args.resume_cwd = str(resume_cwd)
        args.resume_recorder = True
        if cli.resume_extra_step is not None:
            recorder_path = resume_cwd / "recorder.npy"
            if not recorder_path.exists():
                raise FileNotFoundError(f"Cannot infer resume break_step; missing {recorder_path}")
            recorder = np.load(recorder_path)
            if recorder.size == 0:
                raise ValueError(f"Cannot infer resume break_step from empty recorder: {recorder_path}")
            args.break_step = int(recorder[-1][0]) + int(cli.resume_extra_step)

    net_tag = "x".join(str(x) for x in args.net_dims)
    policy_activation = getattr(args, "policy_activation", None)
    hp_tag = (
        f"{cli.agent}_{cli.run_tag}_{cli.paper_preset}_{cli.batch}_{instance_index_tag}_"
        f"lr{args.learning_rate}_alr{getattr(args, 'alpha_learning_rate', None)}_"
        f"rs{args.reward_scale}_g{args.gamma}_net{net_tag}_act{policy_activation}_"
        f"nstep{getattr(args, 'critic_n_step', 1)}_"
        f"replay{getattr(args, 'replay_sampler', 'uniform')}_"
        f"bs{args.batch_size}_rt{args.repeat_times}_hz{args.horizon_len}_buf{args.buffer_size}_"
        f"tau{args.soft_update_tau}_clip{args.clip_grad_norm}_"
        f"gae{getattr(args, 'lambda_gae_adv', None)}_ent{getattr(args, 'lambda_entropy', None)}_"
        f"ratio{getattr(args, 'ratio_clip', None)}_"
        f"explore{getattr(args, 'explore_noise_std', None)}_"
        f"policy{getattr(args, 'policy_noise_std', None)}_"
        f"uf{getattr(args, 'update_freq', None)}_"
        f"ancoef{getattr(args, 'an_penalty_coef', getattr(args, 'pa_penalty_coef', None))}_"
        f"anpow{getattr(args, 'an_penalty_power', getattr(args, 'pa_penalty_power', None))}_"
        f"aneps{getattr(args, 'an_violation_eps', getattr(args, 'pa_violation_eps', None))}_"
        f"antau{getattr(args, 'an_positive_tau', getattr(args, 'pa_positive_tau', None))}_"
        f"laginit{getattr(args, 'lag_lambda_init', None)}_"
        f"laglr{getattr(args, 'lag_lambda_lr', None)}_"
        f"lagdelta{getattr(args, 'lag_delta', None)}_"
        f"lagmax{getattr(args, 'lag_lambda_max', None)}_"
        f"lagint{getattr(args, 'lag_update_interval', None)}"
    )
    cwd_name = cli.cwd_name if cli.cwd_name is not None else f"Kaynov_{hp_tag}"
    args.cwd = str(output_path("train", cwd_name, instance_name))
    my_mkdir(args.cwd)
    Path(args.cwd).mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print(
        f"Training instance={instance_name}, kaynov_index={cli.kaynov_index}, "
        f"small_rvi_index={cli.small_rvi_index}, agent={cli.agent}, "
        f"paper_preset={cli.paper_preset}, lr={args.learning_rate}, "
        f"alpha_learning_rate={getattr(args, 'alpha_learning_rate', None)}, "
        f"reward_scale={args.reward_scale}, gamma={args.gamma}, "
        f"critic_n_step={getattr(args, 'critic_n_step', 1)}, "
        f"net_dims={args.net_dims}, actor_net_dims={args.actor_net_dims}, "
        f"critic_net_dims={args.critic_net_dims}, "
        f"critic_ensemble_size={args.critic_ensemble_size}, "
        f"critic_target_min_size={args.critic_target_min_size}, "
        f"batch_size={args.batch_size}, "
        f"repeat_times={args.repeat_times}, horizon_len={args.horizon_len}, "
        f"buffer_size={args.buffer_size}, soft_update_tau={args.soft_update_tau}, "
        f"clip_grad_norm={args.clip_grad_norm}, "
        f"warmup_steps={args.replay_warmup_steps}, "
        f"fixed_state_norm={args.freeze_state_normalization_from_warmup}, "
        f"save_best_training_state={args.if_save_best_training_state}, "
        f"lambda_gae_adv={getattr(args, 'lambda_gae_adv', None)}, "
        f"lambda_entropy={getattr(args, 'lambda_entropy', None)}, "
        f"ratio_clip={getattr(args, 'ratio_clip', None)}, rho={cli.rho}, "
        f"explore_noise_std={getattr(args, 'explore_noise_std', None)}, "
        f"policy_noise_std={getattr(args, 'policy_noise_std', None)}, "
        f"update_freq={getattr(args, 'update_freq', None)}, "
        f"an_penalty_coef={getattr(args, 'an_penalty_coef', getattr(args, 'pa_penalty_coef', None))}, "
        f"an_penalty_power={getattr(args, 'an_penalty_power', getattr(args, 'pa_penalty_power', None))}, "
        f"an_violation_eps={getattr(args, 'an_violation_eps', getattr(args, 'pa_violation_eps', None))}, "
        f"an_positive_tau={getattr(args, 'an_positive_tau', getattr(args, 'pa_positive_tau', None))}, "
        f"lag_lambda_init={getattr(args, 'lag_lambda_init', None)}, "
        f"lag_lambda_lr={getattr(args, 'lag_lambda_lr', None)}, "
        f"lag_delta={getattr(args, 'lag_delta', None)}, "
        f"lag_lambda_max={getattr(args, 'lag_lambda_max', None)}, "
        f"lag_update_interval={getattr(args, 'lag_update_interval', None)}, "
        f"policy_activation={policy_activation}, "
        f"random_seed={args.random_seed}, "
        f"device={args.device}, gpu_id={args.gpu_id}, "
        f"rollout_device={args.rollout_device}, "
        f"rb_bootstrap_mode={args.rb_bootstrap_mode}, "
        f"rb_bootstrap_steps={args.rb_bootstrap_steps}, "
        f"actor_learning_rate={getattr(args, 'actor_learning_rate', None)}, "
        f"critic_learning_rate={getattr(args, 'critic_learning_rate', None)}, "
        f"replay_sampler={getattr(args, 'replay_sampler', 'uniform')}, "
        f"replay_focus_fraction={getattr(args, 'replay_focus_fraction', 0.0)}, "
        f"replay_sampler_refresh_updates="
        f"{getattr(args, 'replay_sampler_refresh_updates', None)}, "
        f"replay_strata_rdc_lead_times="
        f"{getattr(args, 'replay_strata_rdc_lead_times', None)}, "
        f"replay_strata_cdc_lead_time="
        f"{getattr(args, 'replay_strata_cdc_lead_time', None)}, "
        f"actor_freeze_steps={getattr(args, 'actor_freeze_steps', 0)}, "
        f"actor_update_interval={getattr(args, 'actor_update_interval', 1)}, "
        f"rb_anchor=({getattr(args, 'rb_anchor_coefficient_initial', 0.0)},"
        f"{getattr(args, 'rb_anchor_coefficient_final', 0.0)}), "
        f"rb_teacher_replay_fraction={getattr(args, 'rb_teacher_replay_fraction', 0.0)}, "
        f"rb_teacher_replay_fraction_final="
        f"{getattr(args, 'rb_teacher_replay_fraction_final', None)}, "
        f"rb_teacher_replay_decay_steps="
        f"{getattr(args, 'rb_teacher_replay_decay_steps', 0)}, "
        f"target_entropy={getattr(args, 'target_entropy', None)}, "
        f"checkpoint_validation={getattr(args, 'checkpoint_validation_enabled', False)}, "
        f"allocation_rule={cli.allocation_rule}, "
        f"if_generate_demand={cli.if_generate_demand}, "
        f"demand_fulfillment={cli.demand_fulfillment}, "
        f"emergency_shipment_prob={cli.emergency_shipment_prob}, "
        f"rdc_bsl_min_override={cli.rdc_bsl_min_override}, "
        f"cdc_bsl_min_override={cli.cdc_bsl_min_override}, "
        f"rdc_bsl_max_override={cli.rdc_bsl_max_override}, "
        f"cdc_bsl_max_override={cli.cdc_bsl_max_override}, "
        f"adaptive_action_space={cli.adaptive_action_space}, "
        f"adaptive_action_mode={cli.adaptive_action_mode}, "
        f"adaptive_action_quantile={cli.adaptive_action_quantile}, "
        f"adaptive_action_cost_margin={cli.adaptive_action_cost_margin}, "
        f"adaptive_action_min_quantile={cli.adaptive_action_min_quantile}, "
        f"adaptive_action_max_quantile={cli.adaptive_action_max_quantile}, "
        f"adaptive_cdc_multiplier={cli.adaptive_cdc_multiplier}, "
        f"adaptive_cdc_buffer_alpha={cli.adaptive_cdc_buffer_alpha}, "
        f"adaptive_cdc_buffer_mode={cli.adaptive_cdc_buffer_mode}, "
        f"adaptive_cdc_horizon_mode={cli.adaptive_cdc_horizon_mode}, "
        f"adaptive_cdc_upper_tail_retention={cli.adaptive_cdc_upper_tail_retention}, "
        f"adaptive_cdc_upper_quantile_cap={cli.adaptive_cdc_upper_quantile_cap}, "
        f"adaptive_cdc_cost_margin={cli.adaptive_cdc_cost_margin}, "
        f"adaptive_cdc_min_quantile={cli.adaptive_cdc_min_quantile}, "
        f"adaptive_cdc_max_quantile={cli.adaptive_cdc_max_quantile}, "
        f"adaptive_rdc_lower_fraction={cli.adaptive_rdc_lower_fraction}, "
        f"adaptive_cdc_lower_fraction={cli.adaptive_cdc_lower_fraction}, "
        f"adaptive_rdc_lower_quantile={cli.adaptive_rdc_lower_quantile}, "
        f"adaptive_cdc_lower_quantile={cli.adaptive_cdc_lower_quantile}, "
        f"adaptive_small_case_beta={cli.adaptive_small_case_beta}, "
        f"adaptive_small_case_threshold={cli.adaptive_small_case_threshold}, "
        f"adaptive_rdc_quantiles={adaptive_rdc_quantiles}, "
        f"adaptive_rdc_bsl_min={adaptive_rdc_bsl_min}, "
        f"adaptive_cdc_bsl_min={adaptive_cdc_bsl_min}, "
        f"adaptive_rdc_bsl_max={adaptive_rdc_bsl_max}, "
        f"adaptive_cdc_bsl_max={adaptive_cdc_bsl_max}, "
        f"stationary_action_mask_protocol={stationary_mask_protocol}, "
        f"kaynov_k5_compatible={cli.kaynov_k5_compatible}, "
        f"discrete_action_lengths={discrete_action_lengths}",
        flush=True,
    )
    if cli.resume_from_cwd is not None:
        print(
            f"resume_from_cwd={args.resume_cwd}, resume_extra_step={cli.resume_extra_step}, "
            f"target_break_step={args.break_step}",
            flush=True,
        )
    print(f"cwd={args.cwd}", flush=True)
    print("=" * 80, flush=True)
    train_agent(args)


if __name__ == "__main__":
    main()
