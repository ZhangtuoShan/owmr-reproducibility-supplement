"""RB-guided replay initialization and behavior cloning for SAC.

The RB heuristic chooses echelon base-stock levels.  SAC uses the same
decision semantics, represented as normalized actions in [-1, 1].  The
teacher therefore projects the RB levels into the active SAC action bounds
and lets the target SAC environment apply its own allocation rule.  This
keeps every replay tuple consistent with the transition kernel used later by
SAC.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from my_log import logger


RB_BOOTSTRAP_MODES = {
    "none",
    "replay",
    "behavior_clone",
    "replay_and_clone",
}


def add_rb_bootstrap_arguments(parser) -> None:
    """Expose the same opt-in RB protocol on project training CLIs."""
    parser.add_argument(
        "--rb-bootstrap-mode",
        choices=tuple(sorted(RB_BOOTSTRAP_MODES)),
        default="none",
        help="Initialize SAC from RB replay, behavior cloning, or both.",
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
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--actor-freeze-steps", type=int, default=0)
    parser.add_argument("--actor-update-interval", type=int, default=1)
    parser.add_argument("--rb-anchor-coef-initial", type=float, default=0.0)
    parser.add_argument("--rb-anchor-coef-final", type=float, default=0.0)
    parser.add_argument("--rb-anchor-decay-steps", type=int, default=0)
    parser.add_argument("--rb-anchor-batch-size", type=int, default=256)
    parser.add_argument("--rb-teacher-replay-fraction", type=float, default=0.0)
    parser.add_argument("--rb-teacher-replay-fraction-final", type=float, default=None)
    parser.add_argument("--rb-teacher-replay-decay-steps", type=int, default=0)
    parser.add_argument("--rb-guidance-dataset-size", type=int, default=50000)


def configure_rb_bootstrap(args, cli) -> None:
    for name in (
        "rb_bootstrap_mode",
        "rb_bootstrap_steps",
        "rb_bootstrap_method",
        "rb_bootstrap_action_noise_std",
        "rb_bc_updates",
        "rb_bc_batch_size",
        "rb_bc_learning_rate",
        "rb_bc_dataset_size",
        "actor_learning_rate",
        "critic_learning_rate",
        "actor_freeze_steps",
        "actor_update_interval",
        "rb_anchor_coefficient_initial",
        "rb_anchor_coefficient_final",
        "rb_anchor_decay_steps",
        "rb_anchor_batch_size",
        "rb_teacher_replay_fraction",
        "rb_teacher_replay_fraction_final",
        "rb_teacher_replay_decay_steps",
        "rb_guidance_dataset_size",
    ):
        cli_name = {
            "actor_learning_rate": "actor_lr",
            "critic_learning_rate": "critic_lr",
            "rb_anchor_coefficient_initial": "rb_anchor_coef_initial",
            "rb_anchor_coefficient_final": "rb_anchor_coef_final",
        }.get(name, name)
        setattr(args, name, getattr(cli, cli_name))


def _normalize_target(target: float, lower: float, upper: float) -> tuple[float, bool]:
    lower = float(lower)
    upper = float(upper)
    if upper < lower:
        raise ValueError(f"Invalid action bounds: lower={lower}, upper={upper}")
    if upper - lower <= 1e-12:
        return 0.0, abs(float(target) - lower) > 1e-9
    raw = 2.0 * (float(target) - lower) / (upper - lower) - 1.0
    clipped = float(np.clip(raw, -1.0, 1.0))
    return clipped, abs(raw - clipped) > 1e-9


@dataclass(frozen=True)
class RBTeacherPolicy:
    """Constant RB base-stock targets expressed in SAC action coordinates."""

    action: np.ndarray
    retailer_levels: dict
    warehouse_level: float
    projected_dimensions: tuple[str, ...]
    method: str

    @classmethod
    def from_levels(
            cls,
            env,
            retailer_levels: dict,
            warehouse_level: float,
            method: str = "RB_distribution_aware_capped",
    ) -> "RBTeacherPolicy":
        actions = []
        projected = []
        for rdc in env.rdcs.keys():
            value, was_projected = _normalize_target(
                retailer_levels[rdc],
                env.RDC_min_bsl[rdc],
                env.RDC_max_bsl[rdc],
            )
            actions.append(value)
            if was_projected:
                projected.append(f"RDC{rdc}")

        value, was_projected = _normalize_target(
            warehouse_level,
            env.CDC_min_bsl,
            env.CDC_max_bsl,
        )
        actions.append(value)
        if was_projected:
            projected.append("CDC")

        return cls(
            action=np.asarray(actions, dtype=np.float32),
            retailer_levels={int(key): float(value) for key, value in retailer_levels.items()},
            warehouse_level=float(warehouse_level),
            projected_dimensions=tuple(projected),
            method=str(method),
        )

    @classmethod
    def solve(cls, env, method: str = "RB_distribution_aware_capped") -> "RBTeacherPolicy":
        if getattr(env, "demand_type", None) == "quantile":
            raise ValueError(
                "Stationary RB bootstrap does not support period-dependent quantile demand."
            )
        info_input = getattr(env, "info_input", None)
        if info_input is None:
            # OWMRInventoryEnv keeps the input data through its constituent
            # attributes, while the original object remains in env_args.
            info_input = getattr(env, "_rb_info_input", None)
        if info_input is None:
            raise ValueError("RB bootstrap requires env.info_input or env._rb_info_input.")

        from Benchmark import (
            relaxation_based_heuristic,
            relaxation_based_heuristic_distribution_aware,
            relaxation_based_heuristic_distribution_aware_capped,
        )

        solvers = {
            "RB": relaxation_based_heuristic,
            "RB_distribution_aware": relaxation_based_heuristic_distribution_aware_capped,
            "RB_distribution_aware_capped": relaxation_based_heuristic_distribution_aware_capped,
            "RB_distribution_aware_uncapped": relaxation_based_heuristic_distribution_aware,
        }
        try:
            solver = solvers[method]
        except KeyError as exc:
            supported = ", ".join(sorted(solvers))
            raise ValueError(f"Unsupported RB bootstrap method={method!r}; choose {supported}.") from exc

        _, retailer_levels, warehouse_level = solver(info_input)
        return cls.from_levels(env, retailer_levels, warehouse_level, method=method)

    def describe(self) -> dict:
        return {
            "method": self.method,
            "retailer_levels": self.retailer_levels,
            "warehouse_level": self.warehouse_level,
            "normalized_action": self.action.tolist(),
            "projected_dimensions": list(self.projected_dimensions),
        }


class _RunningStateMoments:
    def __init__(self, state_dim: int, device: torch.device):
        self.total = torch.zeros(state_dim, dtype=torch.float64, device=device)
        self.square_total = torch.zeros(state_dim, dtype=torch.float64, device=device)
        self.count = 0

    def update(self, states: Tensor) -> None:
        states = states.reshape((-1, states.shape[-1])).to(dtype=torch.float64)
        self.total += states.sum(dim=0)
        self.square_total += states.square().sum(dim=0)
        self.count += int(states.shape[0])

    def mean_std(self) -> tuple[Tensor, Tensor]:
        if self.count <= 0:
            raise ValueError("Cannot estimate state normalization from zero states.")
        mean = self.total / self.count
        variance = (self.square_total / self.count - mean.square()).clamp_min(0.0)
        return mean.float(), variance.sqrt().clamp_min(1.0).float()


def set_fixed_state_normalization(agent, state_mean: Tensor, state_std: Tensor) -> None:
    """Apply one state transform to every unique actor/critic network."""
    state_mean = state_mean.to(device=agent.device, dtype=torch.float32)
    state_std = state_std.to(device=agent.device, dtype=torch.float32)
    visited = set()
    for network_name in ("act", "act_target", "cri", "cri_target"):
        network = getattr(agent, network_name, None)
        if network is None or id(network) in visited:
            continue
        visited.add(id(network))
        if hasattr(network, "state_avg") and hasattr(network, "state_std"):
            network.state_avg.copy_(state_mean)
            network.state_std.copy_(state_std)


def collect_rb_trajectory_chunk(
        agent,
        env,
        teacher: RBTeacherPolicy,
        steps: int,
        action_noise_std: float = 0.0,
) -> tuple[tuple[Tensor, ...], Tensor]:
    """Collect SAC-compatible transitions and clean teacher labels."""
    if agent.num_envs != 1:
        raise NotImplementedError("RB bootstrap currently supports num_envs=1 only.")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if action_noise_std < 0:
        raise ValueError("RB bootstrap action noise must be nonnegative.")

    device = agent.rollout_device
    states = torch.empty((steps, 1, agent.state_dim), dtype=torch.float32, device=device)
    actions = torch.empty((steps, 1, agent.action_dim), dtype=torch.float32, device=device)
    labels = torch.empty_like(actions)
    rewards = torch.empty((steps, 1), dtype=torch.float32, device=device)
    dones = torch.empty((steps, 1), dtype=torch.bool, device=device)

    state = agent._get_rollout_state()
    base_action = torch.as_tensor(teacher.action, dtype=torch.float32, device=device).unsqueeze(0)
    for index in range(steps):
        clean_action = agent.apply_action_mask(state, base_action)
        action = clean_action
        if action_noise_std > 0:
            action = (clean_action + torch.randn_like(clean_action) * action_noise_std).clamp(-1.0, 1.0)
            action = agent.apply_action_mask(state, action)

        states[index] = state
        actions[index] = action
        labels[index] = clean_action
        next_state, reward, done, _ = env.step(action[0].detach().cpu().numpy())
        next_state = env.reset() if done else next_state
        state = torch.as_tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
        rewards[index] = float(reward)
        dones[index] = bool(done)

    agent._set_last_rollout_state(state)
    rewards *= agent.reward_scale
    undones = 1.0 - dones.float()
    items = agent._rollout_items_to_train_device((states, actions, rewards, undones))
    labels = labels.to(agent.device) if device != agent.device else labels
    return items, labels


def behavior_clone_actor(
        agent,
        states: Tensor,
        target_actions: Tensor,
        updates: int,
        batch_size: int,
        learning_rate: float,
) -> dict:
    """Fit the deterministic SAC actor mean to RB actions using MSE."""
    states = states.reshape((-1, agent.state_dim)).to(agent.device)
    target_actions = target_actions.reshape((-1, agent.action_dim)).to(agent.device)
    sample_count = int(states.shape[0])
    if sample_count <= 0:
        raise ValueError("Behavior cloning requires at least one state-action pair.")
    if updates <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("BC updates, batch size, and learning rate must be positive.")

    diagnostic_ids = torch.arange(min(sample_count, 8192), device=agent.device)
    with torch.no_grad():
        initial_mse = torch.nn.functional.mse_loss(
            agent.act(states[diagnostic_ids]), target_actions[diagnostic_ids]
        ).item()

    optimizer = torch.optim.Adam(agent.act.parameters(), lr=float(learning_rate))
    agent.act.train()
    with torch.enable_grad():
        for _ in range(int(updates)):
            ids = torch.randint(sample_count, size=(min(int(batch_size), sample_count),), device=agent.device)
            prediction = agent.act(states[ids])
            loss = torch.nn.functional.mse_loss(prediction, target_actions[ids])
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(agent.act.parameters(), max_norm=agent.clip_grad_norm)
            optimizer.step()

    with torch.no_grad():
        final_mse = torch.nn.functional.mse_loss(
            agent.act(states[diagnostic_ids]), target_actions[diagnostic_ids]
        ).item()

    act_target = getattr(agent, "act_target", None)
    if act_target is not None and act_target is not agent.act:
        act_target.load_state_dict(agent.act.state_dict())
    agent._rollout_act = None
    return {
        "samples": sample_count,
        "updates": int(updates),
        "batch_size": min(int(batch_size), sample_count),
        "learning_rate": float(learning_rate),
        "initial_mse": float(initial_mse),
        "final_mse": float(final_mse),
    }


def initialize_sac_from_rb(agent, env, buffer, args) -> dict:
    """Run the configured RB replay/behavior-cloning initialization."""
    mode = str(getattr(args, "rb_bootstrap_mode", "none"))
    if mode not in RB_BOOTSTRAP_MODES or mode == "none":
        raise ValueError(f"initialize_sac_from_rb received inactive/invalid mode={mode!r}")
    if not hasattr(agent, "alpha_optimizer"):
        raise TypeError("RB bootstrap is implemented for SAC-family agents only.")
    if len(buffer.get_history_items()) != 4:
        raise TypeError(
            "RB bootstrap currently supports the standard SAC replay schema only; "
            f"got {buffer.__class__.__name__}."
        )

    steps = getattr(args, "rb_bootstrap_steps", None)
    if steps is None:
        steps = getattr(args, "replay_warmup_steps", None) or args.horizon_len
    steps = int(steps)
    if steps < 2:
        raise ValueError("RB bootstrap requires at least two transitions.")

    teacher = RBTeacherPolicy.solve(
        env,
        method=getattr(args, "rb_bootstrap_method", "RB_distribution_aware_capped"),
    )
    use_replay = mode in {"replay", "replay_and_clone"}
    use_clone = mode in {"behavior_clone", "replay_and_clone"}
    noise_std = float(getattr(args, "rb_bootstrap_action_noise_std", 0.0))
    bc_limit = int(getattr(args, "rb_bc_dataset_size", 50000))
    if use_clone and bc_limit <= 0:
        raise ValueError("rb_bc_dataset_size must be positive when behavior cloning is enabled.")
    moments = _RunningStateMoments(agent.state_dim, agent.device)
    bc_states = []
    bc_actions = []
    bc_count = 0
    keep_persistent_guidance = hasattr(agent, "set_rb_guidance")
    guidance_limit = int(getattr(args, "rb_guidance_dataset_size", 50000))
    if keep_persistent_guidance and guidance_limit <= 1:
        raise ValueError("rb_guidance_dataset_size must exceed one transition")
    guidance_items = [[], [], [], []]
    guidance_states = []
    guidance_actions = []
    guidance_count = 0

    remaining = steps
    while remaining > 0:
        chunk_steps = min(int(args.horizon_len), remaining)
        items, labels = collect_rb_trajectory_chunk(
            agent, env, teacher, chunk_steps, action_noise_std=noise_std
        )
        if use_replay:
            buffer.update(items)
        moments.update(items[0])
        if keep_persistent_guidance and guidance_count < guidance_limit:
            keep = min(chunk_steps, guidance_limit - guidance_count)
            for index in range(4):
                guidance_items[index].append(items[index][:keep].detach())
            guidance_states.append(items[0][:keep].detach())
            guidance_actions.append(labels[:keep].detach())
            guidance_count += keep
        if use_clone and bc_count < bc_limit:
            keep = min(chunk_steps, bc_limit - bc_count)
            bc_states.append(items[0][:keep].detach())
            bc_actions.append(labels[:keep].detach())
            bc_count += keep
        remaining -= chunk_steps

    state_mean, state_std = moments.mean_std()
    if getattr(args, "freeze_state_normalization_from_warmup", False):
        set_fixed_state_normalization(agent, state_mean, state_std)
        logger.info(
            f"| Frozen state normalization from {moments.count} RB states: "
            f"mean_abs={state_mean.abs().mean().item():.3f}, "
            f"std_mean={state_std.mean().item():.3f}"
        )

    bc_metrics = None
    if use_clone:
        bc_metrics = behavior_clone_actor(
            agent,
            states=torch.cat(bc_states, dim=0),
            target_actions=torch.cat(bc_actions, dim=0),
            updates=int(getattr(args, "rb_bc_updates", 1000)),
            batch_size=int(getattr(args, "rb_bc_batch_size", 256)),
            learning_rate=float(
                getattr(args, "rb_bc_learning_rate", None) or args.learning_rate
            ),
        )
        logger.info(f"| RB behavior cloning: {bc_metrics}")

    if keep_persistent_guidance:
        if guidance_count <= 1:
            raise RuntimeError("Persistent RB guidance collected too few transitions")
        agent.set_rb_guidance(
            replay_items=tuple(torch.cat(parts, dim=0) for parts in guidance_items),
            anchor_states=torch.cat(guidance_states, dim=0),
            anchor_actions=torch.cat(guidance_actions, dim=0),
        )
        logger.info(
            f"| Persistent RB guidance retained: transitions={guidance_count}, "
            f"teacher_replay_fraction={getattr(agent, 'rb_teacher_replay_fraction', 0.0):.3f}"
        )

    summary = {
        "mode": mode,
        "steps": steps,
        "replay_size": int(buffer.cur_size) if use_replay else 0,
        "action_noise_std": noise_std,
        "teacher": teacher.describe(),
        "behavior_cloning": bc_metrics,
        "persistent_guidance_size": guidance_count if keep_persistent_guidance else 0,
    }
    cwd = getattr(args, "cwd", None)
    if cwd:
        summary_path = Path(cwd) / "rb_bootstrap_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    logger.info(f"| RB bootstrap complete: {summary}")
    return summary
