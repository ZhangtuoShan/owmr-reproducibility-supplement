#!/usr/bin/env python3
"""Train one SAC policy on a 13-week nonstationary forecast case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from Benchmark import FiniteHorizonNonstationaryOWMREnv, env_input_initialize
from dataset import data_nonstationary_quantile_skus
from elegantrl.agents.AgentSAC import AgentSAC, AgentSACMechanism
from elegantrl.agents.AgentSACLogProbFixed import AgentSACLogProbFixed
from main_DRL_agents import (
    get_nonstationary_env_args,
    get_nonstationary_training_defaults,
    train_nonstationary,
)


ROOT = Path(__file__).resolve().parent
DA_RESULT = ROOT / "outputs/evaluation/nonstationary_rolling_da_validation_20260801/validation.json"
RUN_TAG = "nonstationary_sac_two_cases_20260801"
AGENT_CLASSES = {
    "sac": AgentSAC,
    "sac_mechanism": AgentSACMechanism,
    "sac_logprob_fixed": AgentSACLogProbFixed,
}


def da_reference_cost(item_code: str, result_path: Path) -> float:
    rows = json.loads(result_path.read_text(encoding="utf-8"))
    row = next(item for item in rows if str(item["item_code"]) == str(item_code))
    return float(row["controllable_weekly_cost"]["mean"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument(
        "--item-code",
        help="Explicit SKU to load. Omit to retain the original candidate-index behavior.",
    )
    parser.add_argument("--da-result", type=Path, default=DA_RESULT)
    parser.add_argument("--benchmark", choices=("ra_sac", "ra_sac_ineq"), required=True)
    parser.add_argument("--agent", choices=tuple(AGENT_CLASSES), default="sac")
    parser.add_argument("--alpha-learning-rate", "--alpha-lr", type=float)
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--actor-update-interval", type=int, default=1)
    parser.add_argument("--actor-freeze-steps", type=int, default=0)
    parser.add_argument("--critic-net-dims", default="64,32")
    parser.add_argument("--critic-ensemble-size", type=int, default=2)
    parser.add_argument("--critic-target-min-size", type=int, default=2)
    parser.add_argument(
        "--replay-sampler",
        choices=("uniform", "paired_uniform", "inventory_stratified"),
        default="uniform",
    )
    parser.add_argument("--break-step", type=int, default=500_000)
    parser.add_argument(
        "--initial-inventory-quantile",
        type=float,
        default=None,
        help=(
            "Cold-start service quantile used for initial RDC/CDC on-hand "
            "inventory. Pipelines remain empty."
        ),
    )
    parser.add_argument("--cwd-tag")
    parser.add_argument("--init-actor")
    parser.add_argument("--resume-from-cwd", type=Path)
    parser.add_argument("--initial-total-step", type=int, default=0)
    parser.add_argument("--initial-max-r", type=float, default=float("-inf"))
    parser.add_argument("--save-best-training-state", action="store_true")
    parser.add_argument("--save-buffer", action="store_true")
    parser.add_argument("--eval-times", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if (
        args.initial_inventory_quantile is not None
        and not 0.0 <= args.initial_inventory_quantile <= 1.0
    ):
        raise ValueError("--initial-inventory-quantile must be in [0, 1].")

    critic_net_dims = tuple(
        int(value.strip())
        for value in args.critic_net_dims.split(",")
        if value.strip()
    )
    if not critic_net_dims:
        raise ValueError("--critic-net-dims must contain at least one integer.")
    if args.agent == "sac_mechanism":
        if args.critic_ensemble_size < 2:
            raise ValueError("--critic-ensemble-size must be at least two.")
        if not 1 <= args.critic_target_min_size <= args.critic_ensemble_size:
            raise ValueError(
                "--critic-target-min-size must be in [1, critic-ensemble-size]."
            )

    if args.init_actor and args.resume_from_cwd:
        raise ValueError("Use only one of --init-actor and --resume-from-cwd.")
    if args.resume_from_cwd:
        args.resume_from_cwd = args.resume_from_cwd.resolve()
        required = (
            "act.pth",
            "cri.pth",
            "act_optimizer.pth",
            "cri_optimizer.pth",
            "alpha_log.pth",
            "alpha_optimizer.pth",
            "agent_training_state.pth",
            "recorder.npy",
        )
        missing = [name for name in required if not (args.resume_from_cwd / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Incomplete final training state at {args.resume_from_cwd}: {missing}"
            )

    torch.set_num_threads(1)
    if args.item_code:
        datasets, metadata = data_nonstationary_quantile_skus(
            item_codes=(str(args.item_code),),
            initial_inventory_quantile=args.initial_inventory_quantile,
        )
        data = datasets[0]
        case_metadata = metadata[0]
    else:
        datasets, metadata = data_nonstationary_quantile_skus(
            candidate_count=max(2, args.case_index + 1),
            initial_inventory_quantile=args.initial_inventory_quantile,
        )
        data = datasets[args.case_index]
        case_metadata = metadata[args.case_index]
    item_code = str(case_metadata["item_code"])
    info = env_input_initialize(*data)[0]
    allocation_rule = (
        "prioritized_greedy"
        if args.benchmark == "ra_sac"
        else "prioritized_greedy_inequality"
    )
    da_result = args.da_result if args.da_result.is_absolute() else ROOT / args.da_result
    reference_cost = da_reference_cost(item_code, da_result)
    defaults = get_nonstationary_training_defaults(horizon=13)
    reward_scale = 1.0 / reference_cost
    instance_name = f"nonstationary_{args.case_index}_{item_code}"
    cwd_tag = args.cwd_tag or f"{RUN_TAG}_{args.benchmark}_{item_code}_500k"
    eval_times = defaults["eval_times"] if args.eval_times is None else args.eval_times

    protocol = {
        "case_index": args.case_index,
        "item_code": item_code,
        "benchmark": args.benchmark,
        "agent": args.agent,
        "allocation_rule": allocation_rule,
        "horizon": 13,
        "train_cost_mode": "controllable",
        "action_bound_profile": "joint_quantile_v2",
        "action_lower_quantile": 0.5,
        "action_upper_quantile": 0.99,
        "cdc_protection_horizon": "joint L0+Lk+1 by RDC",
        "reference_rolling_da_weekly_cost": reference_cost,
        "reference_rolling_da_result": str(da_result),
        "learning_rate": 1e-4,
        "actor_learning_rate": args.actor_lr,
        "critic_learning_rate": args.critic_lr,
        "actor_update_interval": args.actor_update_interval,
        "actor_freeze_steps": args.actor_freeze_steps,
        "critic_net_dims": critic_net_dims,
        "critic_ensemble_size": args.critic_ensemble_size,
        "critic_target_min_size": args.critic_target_min_size,
        "replay_sampler": args.replay_sampler,
        "alpha_learning_rate": args.alpha_learning_rate,
        "reward_scale": reward_scale,
        "gamma": defaults["gamma"],
        "horizon_len": defaults["horizon_len"],
        "replay_warmup_steps": defaults["replay_warmup_steps"],
        "fixed_state_normalization": defaults["freeze_state_normalization_from_warmup"],
        "buffer_size": defaults["buffer_size"],
        "batch_size": defaults["batch_size"],
        "repeat_times": defaults["repeat_times"],
        "break_step": int(args.break_step),
        "eval_per_step": defaults["eval_per_step"],
        "eval_times": eval_times,
        "initialization": (
            "final_state_resume"
            if args.resume_from_cwd
            else ("best_actor_warm_restart" if args.init_actor else "from_scratch")
        ),
        "init_actor": args.init_actor,
        "resume_from_cwd": str(args.resume_from_cwd) if args.resume_from_cwd else None,
        "initial_total_step": args.initial_total_step,
        "initial_max_r": args.initial_max_r,
        "save_best_training_state": args.save_best_training_state,
        "initial_inventory_protocol": case_metadata["initial_inventory_protocol"],
        "initial_inventory_quantile": case_metadata["initial_inventory_quantile"],
        "initial_inventory_start_mode": "cold_empty_pipeline",
        "initial_service_levels": case_metadata["initial_service_levels"],
        "cdc_initial_service_level": case_metadata["cdc_initial_service_level"],
    }
    print("NONSTATIONARY_PROTOCOL " + json.dumps(protocol, ensure_ascii=False), flush=True)

    if args.dry_run:
        env_args = get_nonstationary_env_args(
            info,
            horizon=13,
            allocation_rule=allocation_rule,
            train_cost_mode="controllable",
            seed=0,
        )
        env = FiniteHorizonNonstationaryOWMREnv(**{
            key: value
            for key, value in env_args.items()
            if key not in {"env_name", "max_step", "state_dim", "action_dim", "if_discrete"}
        })
        print(json.dumps({
            "state_dim": env.state_dim,
            "action_dim": env.action_dim,
            "rdc_global_bounds": {
                str(k): [env.RDC_min_bsl[k], env.RDC_max_bsl[k]] for k in env.rdcs
            },
            "cdc_global_bounds": [env.CDC_min_bsl, env.CDC_max_bsl],
            "initial_inventory": {
                "RDC": {
                    str(k): [float(value) for value in info.RDC_inventory[k]]
                    for k in sorted(info.RDC_inventory)
                },
                "CDC": [float(value) for value in info.CDC_inventory],
            },
            "active_actions_by_period": [int(env.get_action_mask(t).sum()) for t in range(13)],
        }, ensure_ascii=False), flush=True)
        return

    train_nonstationary(
        agent_class=AGENT_CLASSES[args.agent],
        input_env=info,
        instance_name=instance_name,
        cwd=cwd_tag,
        horizon=13,
        if_generate_demand=True,
        allocation_rule=allocation_rule,
        train_cost_mode="controllable",
        reference_weekly_cost=reference_cost,
        learning_rate=1e-4,
        actor_learning_rate=args.actor_lr,
        critic_learning_rate=args.critic_lr,
        actor_update_interval=args.actor_update_interval,
        actor_freeze_steps=args.actor_freeze_steps,
        actor_net_dims=(64, 32),
        critic_net_dims=critic_net_dims,
        critic_ensemble_size=args.critic_ensemble_size,
        critic_target_min_size=args.critic_target_min_size,
        replay_sampler=args.replay_sampler,
        alpha_learning_rate=args.alpha_learning_rate,
        reward_scale=reward_scale,
        gamma=defaults["gamma"],
        horizon_len=defaults["horizon_len"],
        replay_warmup_steps=defaults["replay_warmup_steps"],
        freeze_state_normalization_from_warmup=True,
        buffer_size=defaults["buffer_size"],
        batch_size=defaults["batch_size"],
        repeat_times=defaults["repeat_times"],
        net_dims=(64, 32),
        break_step=args.break_step,
        eval_per_step=defaults["eval_per_step"],
        eval_times=eval_times,
        init_actor_path=args.init_actor,
        resume_cwd=str(args.resume_from_cwd) if args.resume_from_cwd else None,
        resume_recorder=True,
        initial_total_step=args.initial_total_step,
        initial_max_r=args.initial_max_r,
        if_save_best_training_state=args.save_best_training_state,
        if_save_buffer=args.save_buffer,
        device="cpu",
        gpu_id=-1,
    )


if __name__ == "__main__":
    main()
