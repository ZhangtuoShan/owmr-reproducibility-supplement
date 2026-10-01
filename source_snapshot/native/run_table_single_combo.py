import argparse
import json
import math
from pathlib import Path

import numpy as np

from Benchmark import (
    OWMRInventoryEnv,
    auto_hyperparameter,
    compute_adaptive_action_space_overrides,
    compute_joint_horizon_cdc_lower_bound,
    compute_table15_size_aware_action_space_overrides,
    env_input_initialize,
)
from dataset import (
    data_random_table_benchmarks,
    data_random_table_benchmark_metadata,
    data_sensitivity_analysis_13,
    data_sensitivity_analysis_13_corrected,
    data_sensitivity_analysis_13_corrected_metadata,
    data_sensitivity_analysis_13_metadata,
    data_structural_sensitivity,
    data_structural_sensitivity_metadata,
)
from elegantrl.agents import (
    AgentANSAC,
    AgentLagANSAC,
    AgentPenaltyANSAC,
    AgentSAC,
    AgentSACMechanism,
)
from elegantrl.agents.AgentPPO import AgentDiscreteMultiPPO, AgentPPO
from elegantrl.agents.AgentTD3 import AgentTD3
from elegantrl.train.config import Config, configure_training_evaluation
from elegantrl.train.run import train_agent
from main_DRL_agents import get_env_args
from output_paths import output_path
from utils import my_mkdir


BENCHMARKS = {
    "ra_sac": {"agent": AgentSAC, "allocation_rule": "prioritized_greedy"},
    "ra_sac_ineq": {"agent": AgentSAC, "allocation_rule": "prioritized_greedy_inequality"},
    "ppo_kaynov": {"agent": AgentDiscreteMultiPPO, "allocation_rule": "random_sequential", "method": "paper"},
    "ra_ppo": {"agent": AgentPPO, "allocation_rule": "prioritized_greedy"},
    "ra_td3": {"agent": AgentTD3, "allocation_rule": "prioritized_greedy"},
    "pa_sac": {"agent": AgentSAC, "allocation_rule": "proportion"},
    "sa_sac": {"agent": AgentSAC, "allocation_rule": "random_sequential"},
    "an_sac": {"agent": AgentANSAC, "allocation_rule": "proportion"},
    "penalty_an_sac": {"agent": AgentPenaltyANSAC, "allocation_rule": "proportion"},
    "lag_an_sac": {"agent": AgentLagANSAC, "allocation_rule": "proportion"},
}


def parse_bound_override(text):
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    if raw.startswith("{"):
        data = json.loads(raw)
        return {int(k): float(v) for k, v in data.items()}
    return float(raw)


def parse_net_dims(text):
    dims = tuple(int(x.strip()) for x in str(text).split(",") if x.strip())
    if not dims:
        raise ValueError("--net-dims must contain at least one integer")
    return dims


def parse_args():
    parser = argparse.ArgumentParser(description="Train one benchmark combo on a supported OWMR dataset.")
    parser.add_argument("--benchmark", choices=BENCHMARKS.keys(), required=True)
    parser.add_argument(
        "--sac-agent",
        choices=("standard", "mechanism"),
        default="standard",
        help=(
            "Use the opt-in SAC mechanism implementation for SAC-family "
            "benchmarks. The default preserves historical AgentSAC behavior."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=(
            "table15",
            "sensitivity13",
            "sensitivity13_corrected",
            "structural_sensitivity",
        ),
        default="table15",
    )
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--reward-scale", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--paper-preset", choices=("none", "kaynov2024"), default="none")
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
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help=(
            "Seed used for the pre-generated demand bank, environment RNG, "
            "network initialization, and replay sampling."
        ),
    )
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "auto", "cuda", "mps"), default="cpu")
    parser.add_argument("--rollout-device", choices=("cpu", "auto", "cuda", "mps"), default=None)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--net-dims", default="64,32")
    parser.add_argument("--actor-net-dims", default=None)
    parser.add_argument("--critic-net-dims", default=None)
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--actor-update-interval", type=int, default=1)
    parser.add_argument("--actor-freeze-steps", type=int, default=0)
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
    parser.add_argument("--replay-store-after-period", type=int, default=0)
    parser.add_argument("--training-episode-steps", type=int, default=None)
    parser.add_argument(
        "--rb-bootstrap-mode",
        choices=("none", "replay", "behavior_clone", "replay_and_clone"),
        default="none",
        help="Initialize SAC from RB trajectories, RB behavior cloning, or both.",
    )
    parser.add_argument("--rb-bootstrap-steps", type=int, default=None)
    parser.add_argument(
        "--rb-bootstrap-method",
        choices=(
            "RB", "RB_distribution_aware", "RB_distribution_aware_capped",
            "RB_distribution_aware_uncapped",
        ),
        default="RB_distribution_aware_capped",
    )
    parser.add_argument("--rb-bootstrap-action-noise-std", type=float, default=0.0)
    parser.add_argument("--rb-bc-updates", type=int, default=1000)
    parser.add_argument("--rb-bc-batch-size", type=int, default=256)
    parser.add_argument("--rb-bc-learning-rate", type=float, default=None)
    parser.add_argument("--rb-bc-dataset-size", type=int, default=50000)
    parser.add_argument(
        "--save-best-training-state",
        action="store_true",
        help="Save a resumable full agent and replay-buffer snapshot whenever evaluation improves.",
    )
    parser.add_argument(
        "--save-replay-buffer",
        action="store_true",
        help="Save the final off-policy replay buffer so later continuation can restore it exactly.",
    )
    parser.add_argument("--alpha-learning-rate", type=float, default=None)
    parser.add_argument(
        "--resume-learning-rate-override",
        type=float,
        default=None,
        help="After loading a complete training state, set actor/critic optimizer learning rates.",
    )
    parser.add_argument(
        "--resume-alpha-learning-rate-override",
        type=float,
        default=None,
        help="After loading a complete training state, set the temperature optimizer learning rate.",
    )
    parser.add_argument("--lambda-gae-adv", type=float, default=None)
    parser.add_argument("--lambda-entropy", type=float, default=None)
    parser.add_argument("--ratio-clip", type=float, default=None)
    parser.add_argument("--policy-activation", choices=("relu", "tanh", "gelu", "elu"), default=None)
    parser.add_argument("--explore-noise-std", type=float, default=None)
    parser.add_argument("--policy-noise-std", type=float, default=None)
    parser.add_argument("--update-freq", type=int, default=None)
    parser.add_argument("--an-penalty-coef", type=float, default=None)
    parser.add_argument("--an-penalty-power", type=float, default=None)
    parser.add_argument("--an-violation-eps", type=float, default=None)
    parser.add_argument("--an-positive-tau", type=float, default=None)
    parser.add_argument("--lag-lambda-init", type=float, default=None)
    parser.add_argument("--lag-lambda-lr", type=float, default=None)
    parser.add_argument("--lag-delta", type=float, default=None)
    parser.add_argument("--lag-lambda-max", type=float, default=None)
    parser.add_argument("--lag-update-interval", type=int, default=None)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--allocation-rule", default=None, help="Override the benchmark default allocation rule.")
    parser.add_argument("--if-generate-demand", action="store_true")
    parser.add_argument("--rdc-bsl-min-override", type=parse_bound_override, default=None)
    parser.add_argument("--cdc-bsl-min-override", type=float, default=None)
    parser.add_argument("--rdc-bsl-max-override", type=parse_bound_override, default=None)
    parser.add_argument("--cdc-bsl-max-override", type=float, default=None)
    parser.add_argument(
        "--stationary-action-mask-protocol",
        choices=("legacy", "masked"),
        default="legacy",
        help=(
            "Append a constant action mask to the stationary state only for "
            "experiments that intentionally collapse action dimensions."
        ),
    )
    parser.add_argument("--adaptive-action-space", action="store_true")
    parser.add_argument(
        "--table15-action-space-profile",
        choices=(
            "none",
            "size_aware_v1",
            "size_aware_v2_joint_cdc",
            "size_aware_v3_joint_cdc_tail",
        ),
        default="none",
        help="Use a validated table15 action-space profile before explicit bound overrides.",
    )
    parser.add_argument("--adaptive-action-mode", choices=("cost_aware", "fixed"), default="cost_aware")
    parser.add_argument("--adaptive-action-quantile", type=float, default=0.92)
    parser.add_argument("--adaptive-action-cost-margin", type=float, default=0.03)
    parser.add_argument("--adaptive-action-min-quantile", type=float, default=0.85)
    parser.add_argument("--adaptive-action-max-quantile", type=float, default=0.95)
    parser.add_argument("--adaptive-cdc-buffer-alpha", type=float, default=0.0)
    parser.add_argument(
        "--adaptive-cdc-buffer-mode",
        choices=("required", "legacy_scaled"),
        default="required",
    )
    parser.add_argument(
        "--adaptive-cdc-horizon-mode",
        choices=("additive", "joint"),
        default="joint",
    )
    parser.add_argument("--adaptive-cdc-upper-tail-retention", type=float, default=0.25)
    parser.add_argument("--adaptive-cdc-upper-quantile-cap", type=float, default=0.99)
    parser.add_argument("--adaptive-cdc-min-span", type=float, default=None)
    parser.add_argument("--adaptive-rdc-lower-fraction", type=float, default=0.0)
    parser.add_argument("--adaptive-cdc-lower-fraction", type=float, default=0.25)
    parser.add_argument(
        "--adaptive-cdc-lower-rule",
        choices=("legacy", "joint_horizon"),
        default="legacy",
        help="Optionally replace the adaptive CDC lower bound with one joint-horizon quantile.",
    )
    parser.add_argument("--run-tag", default="table15")
    parser.add_argument("--cwd-name", default=None)
    parser.add_argument("--resume-from-cwd", default=None)
    parser.add_argument("--resume-extra-step", type=int, default=None)
    parser.add_argument("--resume-total-step", type=int, default=None)
    return parser.parse_args()


def apply_kaynov2024_paper_preset(args, cli):
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


def main():
    cli = parse_args()
    if cli.paper_preset == "none" and (cli.lr is None or cli.reward_scale is None or cli.gamma is None):
        raise ValueError("--lr, --reward-scale, and --gamma are required unless --paper-preset kaynov2024 is used.")

    # Demand is generated before Config.init_before_training() seeds NumPy.
    # Seed here so jobs with the same training seed use the same demand bank.
    np.random.seed(cli.random_seed)

    if cli.dataset == "table15":
        cases = data_random_table_benchmarks()
        metadata = data_random_table_benchmark_metadata()
    elif cli.dataset == "sensitivity13":
        cases = data_sensitivity_analysis_13()
        metadata = data_sensitivity_analysis_13_metadata()
    elif cli.dataset == "sensitivity13_corrected":
        cases = data_sensitivity_analysis_13_corrected()
        metadata = data_sensitivity_analysis_13_corrected_metadata()
    else:
        cases = data_structural_sensitivity()
        metadata = data_structural_sensitivity_metadata()
    if not 0 <= cli.case_index < len(cases):
        raise IndexError(f"--case-index must be in [0, {len(cases)-1}], got {cli.case_index}")
    data = cases[cli.case_index]
    case_name = metadata[cli.case_index]["case"]
    rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, rdc_inv, cdc_inv = data
    input_env, _, _ = env_input_initialize(
        rdc_dict,
        demand_feature,
        holding_cost,
        backorder_cost,
        item_lead_time,
        rdc_inv,
        cdc_inv,
    )
    _, _, _, holding_weight, penalty_weight, cost_plus, all_demand_list = auto_hyperparameter(input_env, 100)

    spec = BENCHMARKS[cli.benchmark]
    allocation_rule = cli.allocation_rule or spec["allocation_rule"]
    input_args = {
        "all_demand_list": all_demand_list,
        "holding_weight": holding_weight,
        "penalty_weight": penalty_weight,
        "cost_plus": cost_plus,
        "info_input": input_env,
        "if_generate_demand": cli.if_generate_demand,
        "rho": cli.rho,
        "allocation_rule": allocation_rule,
        "rdc_bsl_min_override": cli.rdc_bsl_min_override,
        "cdc_bsl_min_override": cli.cdc_bsl_min_override,
        "rdc_bsl_max_override": cli.rdc_bsl_max_override,
        "cdc_bsl_max_override": cli.cdc_bsl_max_override,
        "include_stationary_action_mask_in_state": (
            cli.stationary_action_mask_protocol == "masked"
        ),
    }
    if cli.training_episode_steps is not None:
        if cli.training_episode_steps <= 0:
            raise ValueError("--training-episode-steps must be positive.")
        input_args["total_periods"] = int(cli.training_episode_steps)
    if cli.replay_store_after_period < 0:
        raise ValueError("--replay-store-after-period must be nonnegative.")
    if (
            cli.training_episode_steps is not None
            and cli.replay_store_after_period > 0
            and cli.replay_store_after_period >= cli.training_episode_steps):
        raise ValueError(
            "--replay-store-after-period must be smaller than "
            "--training-episode-steps."
        )
    if spec.get("method") == "paper":
        input_args["method"] = "paper"

    use_adaptive_action_space = (
        cli.adaptive_action_space or cli.table15_action_space_profile != "none"
    )
    if use_adaptive_action_space and spec.get("method") != "paper":
        if cli.table15_action_space_profile in {
                "size_aware_v1",
                "size_aware_v2_joint_cdc",
                "size_aware_v3_joint_cdc_tail",
        }:
            profile_is_joint = cli.table15_action_space_profile != "size_aware_v1"
            profile_tail_retention = (
                1.0
                if cli.table15_action_space_profile == "size_aware_v2_joint_cdc"
                else 0.25
            )
            rdc_min, cdc_min, rdc_max, cdc_max = (
                compute_table15_size_aware_action_space_overrides(
                    input_env,
                    cdc_horizon_mode="joint" if profile_is_joint else "additive",
                    cdc_upper_tail_retention=profile_tail_retention,
                    cdc_upper_quantile_cap=0.99,
                )
            )
        else:
            rdc_min, cdc_min, rdc_max, cdc_max = compute_adaptive_action_space_overrides(
                input_env,
                mode=cli.adaptive_action_mode,
                quantile=cli.adaptive_action_quantile,
                cost_margin=cli.adaptive_action_cost_margin,
                min_quantile=cli.adaptive_action_min_quantile,
                max_quantile=cli.adaptive_action_max_quantile,
                cdc_buffer_alpha=cli.adaptive_cdc_buffer_alpha,
                cdc_buffer_mode=cli.adaptive_cdc_buffer_mode,
                cdc_horizon_mode=cli.adaptive_cdc_horizon_mode,
                cdc_upper_tail_retention=cli.adaptive_cdc_upper_tail_retention,
                cdc_upper_quantile_cap=cli.adaptive_cdc_upper_quantile_cap,
                rdc_lower_fraction=cli.adaptive_rdc_lower_fraction,
                cdc_lower_fraction=cli.adaptive_cdc_lower_fraction,
                return_bounds=True,
            )
        if (
                cli.adaptive_cdc_lower_rule == "joint_horizon"
                and cli.cdc_bsl_min_override is None):
            cdc_min = min(
                compute_joint_horizon_cdc_lower_bound(input_env),
                float(cdc_max),
            )
        if cli.rdc_bsl_min_override is None:
            input_args["rdc_bsl_min_override"] = rdc_min
        if cli.cdc_bsl_min_override is None:
            input_args["cdc_bsl_min_override"] = cdc_min
        if cli.rdc_bsl_max_override is None:
            input_args["rdc_bsl_max_override"] = rdc_max
        if cli.cdc_bsl_max_override is None:
            input_args["cdc_bsl_max_override"] = cdc_max
        effective_cdc_min = input_args.get("cdc_bsl_min_override")
        effective_cdc_max = input_args.get("cdc_bsl_max_override")
        if effective_cdc_min is not None and effective_cdc_max is not None:
            min_span = 1.0 if cli.adaptive_cdc_min_span is None else float(cli.adaptive_cdc_min_span)
            if float(effective_cdc_max) - float(effective_cdc_min) < min_span - 1e-9:
                raise ValueError(
                    "Adaptive CDC action range collapsed: "
                    f"lower={effective_cdc_min}, upper={effective_cdc_max}, required_span={min_span}."
                )

    env_args = get_env_args(input_args)
    sac_benchmarks = {"ra_sac", "ra_sac_ineq", "pa_sac", "sa_sac"}
    if cli.sac_agent == "mechanism" and cli.benchmark not in sac_benchmarks:
        raise ValueError("--sac-agent mechanism is only valid for SAC-family benchmarks.")
    agent_class = (
        AgentSACMechanism
        if cli.sac_agent == "mechanism"
        else spec["agent"]
    )
    args = Config(agent_class, env_class=OWMRInventoryEnv, env_args=env_args)
    args.random_seed = cli.random_seed
    args.gamma = cli.gamma if cli.gamma is not None else args.gamma
    args.learning_rate = cli.lr if cli.lr is not None else args.learning_rate
    args.reward_scale = cli.reward_scale if cli.reward_scale is not None else args.reward_scale
    args.net_dims = parse_net_dims(cli.net_dims)
    args.actor_net_dims = (
        None if cli.actor_net_dims is None else parse_net_dims(cli.actor_net_dims)
    )
    args.critic_net_dims = (
        None if cli.critic_net_dims is None else parse_net_dims(cli.critic_net_dims)
    )
    args.actor_learning_rate = cli.actor_lr
    args.critic_learning_rate = cli.critic_lr
    args.actor_update_interval = int(cli.actor_update_interval)
    args.actor_freeze_steps = int(cli.actor_freeze_steps)
    args.critic_ensemble_size = int(cli.critic_ensemble_size)
    args.critic_target_min_size = int(cli.critic_target_min_size)
    if cli.sac_agent == "mechanism":
        if args.critic_ensemble_size < 2:
            raise ValueError("--critic-ensemble-size must be at least two.")
        if not 1 <= args.critic_target_min_size <= args.critic_ensemble_size:
            raise ValueError(
                "--critic-target-min-size must be in [1, critic-ensemble-size]."
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
    args.device = cli.device
    args.rollout_device = cli.rollout_device
    args.gpu_id = -1 if cli.device == "cpu" else cli.gpu_id

    if cli.resume_extra_step is not None and cli.resume_total_step is not None:
        raise ValueError("Use only one of --resume-extra-step and --resume-total-step.")
    if cli.resume_from_cwd is not None:
        resume_cwd = Path(cli.resume_from_cwd)
        if not resume_cwd.exists():
            raise FileNotFoundError(f"--resume-from-cwd does not exist: {resume_cwd}")
        recorder_path = resume_cwd / "recorder.npy"
        if not recorder_path.exists():
            raise FileNotFoundError(f"Cannot resume without recorder: {recorder_path}")
        recorder = np.load(recorder_path)
        if recorder.size == 0:
            raise ValueError(f"Cannot resume from empty recorder: {recorder_path}")
        args.resume_cwd = str(resume_cwd)
        args.resume_recorder = True
        if cli.resume_extra_step is not None:
            args.break_step = int(recorder[-1][0]) + int(cli.resume_extra_step)
        elif cli.resume_total_step is not None:
            args.break_step = int(cli.resume_total_step)

    for attr, value in {
        "batch_size": cli.batch_size,
        "repeat_times": cli.repeat_times,
        "horizon_len": cli.horizon_len,
        "buffer_size": cli.buffer_size,
        "soft_update_tau": cli.soft_update_tau,
        "clip_grad_norm": cli.clip_grad_norm,
        "replay_warmup_steps": cli.replay_warmup_steps,
        "replay_store_after_period": cli.replay_store_after_period,
        "rb_bootstrap_steps": cli.rb_bootstrap_steps,
        "rb_bc_learning_rate": cli.rb_bc_learning_rate,
        "alpha_learning_rate": cli.alpha_learning_rate,
        "resume_learning_rate_override": cli.resume_learning_rate_override,
        "resume_alpha_learning_rate_override": cli.resume_alpha_learning_rate_override,
        "lambda_gae_adv": cli.lambda_gae_adv,
        "lambda_entropy": cli.lambda_entropy,
        "ratio_clip": cli.ratio_clip,
        "policy_activation": cli.policy_activation,
        "explore_noise_std": cli.explore_noise_std,
        "policy_noise_std": cli.policy_noise_std,
        "update_freq": cli.update_freq,
        "an_penalty_coef": cli.an_penalty_coef,
        "pa_penalty_coef": cli.an_penalty_coef,
        "an_penalty_power": cli.an_penalty_power,
        "pa_penalty_power": cli.an_penalty_power,
        "an_violation_eps": cli.an_violation_eps,
        "pa_violation_eps": cli.an_violation_eps,
        "an_positive_tau": cli.an_positive_tau,
        "pa_positive_tau": cli.an_positive_tau,
        "lag_lambda_init": cli.lag_lambda_init,
        "lag_lambda_lr": cli.lag_lambda_lr,
        "lag_delta": cli.lag_delta,
        "lag_lambda_max": cli.lag_lambda_max,
        "lag_update_interval": cli.lag_update_interval,
    }.items():
        if value is not None:
            setattr(args, attr, value)
    args.freeze_state_normalization_from_warmup = (
        cli.freeze_state_normalization_from_warmup
    )
    args.replay_sampler = cli.replay_sampler
    args.rb_bootstrap_mode = cli.rb_bootstrap_mode
    args.rb_bootstrap_method = cli.rb_bootstrap_method
    args.rb_bootstrap_action_noise_std = cli.rb_bootstrap_action_noise_std
    args.rb_bc_updates = cli.rb_bc_updates
    args.rb_bc_batch_size = cli.rb_bc_batch_size
    args.rb_bc_dataset_size = cli.rb_bc_dataset_size
    args.if_save_best_training_state = cli.save_best_training_state
    args.if_save_buffer = cli.save_replay_buffer
    if cli.benchmark == "an_sac":
        args.an_penalty_coef = 0.0
        args.pa_penalty_coef = 0.0
    apply_kaynov2024_paper_preset(args, cli)

    net_tag = "x".join(str(x) for x in args.net_dims)
    cwd_name = cli.cwd_name or (
        f"Table15_{cli.benchmark}_{cli.run_tag}_{case_name}_"
        f"lr{args.learning_rate}_rs{args.reward_scale}_g{args.gamma}_net{net_tag}_"
        f"bs{args.batch_size}_rt{args.repeat_times}_hz{args.horizon_len}"
    )
    args.cwd = str(output_path("train", cwd_name, case_name))
    my_mkdir(args.cwd)
    Path(args.cwd).mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print(
        f"Training dataset={cli.dataset} case {case_name} index={cli.case_index}, "
        f"benchmark={cli.benchmark}, "
        f"agent={agent_class.__name__}, sac_agent={cli.sac_agent}, "
        f"allocation_rule={allocation_rule}, "
        f"lr={args.learning_rate}, reward_scale={args.reward_scale}, gamma={args.gamma}, "
        f"net_dims={args.net_dims}, actor_net_dims={args.actor_net_dims}, "
        f"critic_net_dims={args.critic_net_dims}, "
        f"actor_lr={args.actor_learning_rate}, critic_lr={args.critic_learning_rate}, "
        f"actor_update_interval={args.actor_update_interval}, "
        f"actor_freeze_steps={args.actor_freeze_steps}, "
        f"critic_ensemble_size={args.critic_ensemble_size}, "
        f"critic_target_min_size={args.critic_target_min_size}, "
        f"replay_sampler={args.replay_sampler}, "
        f"batch_size={args.batch_size}, repeat_times={args.repeat_times}, "
        f"horizon_len={args.horizon_len}, break_step={args.break_step}, "
        f"warmup_steps={args.replay_warmup_steps}, "
        f"replay_store_after_period={args.replay_store_after_period}, "
        f"training_episode_steps={input_args.get('total_periods', 100)}, "
        f"rb_bootstrap={args.rb_bootstrap_mode}, rb_steps={args.rb_bootstrap_steps}, "
        f"rb_bc_updates={args.rb_bc_updates}, "
        f"fixed_state_norm={args.freeze_state_normalization_from_warmup}, "
        f"alpha_lr={args.alpha_learning_rate}, "
        f"resume_lr_override={args.resume_learning_rate_override}, "
        f"resume_alpha_lr_override={args.resume_alpha_learning_rate_override}, "
        f"save_best_training_state={args.if_save_best_training_state}, "
        f"save_replay_buffer={args.if_save_buffer}, "
        f"training_eval_profile={args.training_eval_profile}, "
        f"eval_per_step={args.eval_per_step}, eval_times={args.eval_times}, "
        f"eval_episode_steps={args.eval_episode_steps}, "
        f"eval_burn_in_steps={args.eval_burn_in_steps}, "
        f"device={args.device}, cwd={args.cwd}",
        flush=True,
    )
    if cli.resume_from_cwd is not None:
        print(
            f"Resume config: from={args.resume_cwd}, target_step={args.break_step}, "
            f"resume_extra_step={cli.resume_extra_step}, "
            f"resume_total_step={cli.resume_total_step}",
            flush=True,
        )
    print(
        "Action-space config: "
        f"adaptive={use_adaptive_action_space}, "
        f"profile={cli.table15_action_space_profile}, "
        f"cdc_buffer_mode={cli.adaptive_cdc_buffer_mode}, "
        f"cdc_buffer_alpha={cli.adaptive_cdc_buffer_alpha}, "
        f"cdc_lower_rule={cli.adaptive_cdc_lower_rule}, "
        f"rdc_min={input_args.get('rdc_bsl_min_override')}, "
        f"rdc_max={input_args.get('rdc_bsl_max_override')}, "
        f"cdc_min={input_args.get('cdc_bsl_min_override')}, "
        f"cdc_max={input_args.get('cdc_bsl_max_override')}",
        flush=True,
    )
    print("=" * 80, flush=True)
    train_agent(args)


if __name__ == "__main__":
    main()
