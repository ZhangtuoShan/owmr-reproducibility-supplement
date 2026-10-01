import math
from functools import partial
import torch
import numpy as np
from typing import Tuple
from copy import deepcopy
from types import MethodType
from scipy import stats
from torch import Tensor

from demand_utils import (
    demand_mean,
    demand_std,
    lead_time_demand_quantile,
    smooth_lead_time_quantile_table,
)
from elegantrl.agents.AgentBase import AgentBase, get_optim_param
from elegantrl.agents.net import ActorSAC, ActorFixSAC, CriticTwin, CriticEnsemble
from elegantrl.train.config import Config
from elegantrl.train.replay_buffer import ReplayBuffer, ReplayBufferRecommendedSAC, ReplayBufferANSAC


def build_truncated_td_lambda_target(
        rewards: Tensor,
        undones: Tensor,
        next_soft_values: Tensor,
        gamma: float,
        trace_lambda: float,
) -> Tensor:
    """Return an episode-safe finite-horizon TD(lambda) target.

    ``next_soft_values[:, k]`` is the frozen-policy soft value at ``s_{t+k+1}``.
    The final component uses the full-horizon bootstrap; earlier components
    blend one-step bootstrapping with the recursively accumulated return.
    """
    if rewards.ndim != 2 or rewards.shape != undones.shape:
        raise ValueError("rewards and undones must have the same [batch, horizon] shape")
    if next_soft_values.shape != rewards.shape:
        raise ValueError("next_soft_values must match rewards shape")
    if rewards.shape[1] < 1:
        raise ValueError("TD(lambda) requires a positive horizon")
    trace_lambda = float(trace_lambda)
    if not 0.0 <= trace_lambda <= 1.0:
        raise ValueError(f"trace_lambda must be in [0, 1], got {trace_lambda}.")

    target = next_soft_values[:, -1]
    for index in range(rewards.shape[1] - 1, -1, -1):
        if index == rewards.shape[1] - 1:
            continuation = next_soft_values[:, index]
        else:
            continuation = (
                (1.0 - trace_lambda) * next_soft_values[:, index]
                + trace_lambda * target
            )
        target = rewards[:, index] + float(gamma) * undones[:, index] * continuation
    return target


class AgentSAC(AgentBase):
    """Standard one-step SAC; allocation variants share this learning backbone."""
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, 'act_class', ActorSAC)
        self.cri_class = getattr(self, 'cri_class', CriticTwin)
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        self.cri_target = deepcopy(self.cri)

        self.alpha_log = torch.tensor((-1,), dtype=torch.float32, requires_grad=True, device=self.device)  # trainable
        configured_alpha_lr = getattr(args, 'alpha_learning_rate', None)
        self.alpha_learning_rate = (
            self.learning_rate * 4 if configured_alpha_lr is None else float(configured_alpha_lr)
        )
        self.optimizer_weight_decay = 0.0
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)
        self.alpha_optimizer = torch.optim.Adam((self.alpha_log,), lr=self.alpha_learning_rate)
        for optimizer in (self.act_optimizer, self.cri_optimizer, self.alpha_optimizer):
            optimizer.parameters = MethodType(get_optim_param, optimizer)
        self.criterion = torch.nn.MSELoss(reduction="none" if self.if_use_per else "mean")
        self.target_entropy = float(getattr(args, 'target_entropy', action_dim))
        self.target_entropy_per_action = self.target_entropy / max(int(action_dim), 1)
        self.save_attr_names.update({'alpha_log', 'alpha_optimizer'})

    def save_or_load_agent(self, cwd: str, if_save: bool):
        super().save_or_load_agent(cwd, if_save)
        if not if_save:
            # Legacy optimizer state must not reintroduce weight decay.
            for optimizer in (self.act_optimizer, self.cri_optimizer, self.alpha_optimizer):
                for group in optimizer.param_groups:
                    group["weight_decay"] = 0.0

    def _actor_q_min(self, states: Tensor, actions: Tensor) -> Tensor:
        # Keep gradients through the action while treating critic weights as fixed.
        parameters = tuple(self.cri.parameters())
        original_flags = tuple(parameter.requires_grad for parameter in parameters)
        try:
            for parameter in parameters:
                parameter.requires_grad_(False)
            return self.cri.get_q_min(states, actions)
        finally:
            for parameter, flag in zip(parameters, original_flags):
                parameter.requires_grad_(flag)

    def get_target_entropy(self, states: Tensor):
        action_mask = self.get_action_mask(states)
        if action_mask is None:
            return self.target_entropy
        return action_mask.sum(dim=1) * self.target_entropy_per_action

    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
        with torch.no_grad():
            states, actions, rewards, undones = buffer.add_item
            self.update_avg_std_for_normalization(
                states=states.reshape((-1, self.state_dim)),
                returns=self.get_cumulative_rewards(rewards=rewards, undones=undones).reshape((-1,))
            )

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0
        alphas = 0.0

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        for _ in range(update_times):
            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            '''objective of alpha (temperature parameter automatic adjustment)'''
            action_pg, log_prob = self.act.get_action_logprob(state)  # policy gradient
            target_entropy = self.get_target_entropy(state)
            obj_alpha = (self.alpha_log * (target_entropy - log_prob).detach()).mean()
            self.optimizer_update(self.alpha_optimizer, obj_alpha)

            '''objective of actor'''
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2)
            alpha = self.alpha_log.exp().detach()
            alphas += alpha.item()

            q_value_pg = self._actor_q_min(state, action_pg)
            obj_actor = (q_value_pg - log_prob * alpha).mean()
            obj_actors += obj_actor.item()
            self.optimizer_update(self.act_optimizer, -obj_actor)

        return obj_critics / update_times, obj_actors / update_times, alphas / update_times

    def get_obj_critic_raw(self, buffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            # while True:
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)  # next_ss: next states
                # if torch.isnan(next_ss).any():
                #     continue
                # break

            # try:
            next_as, next_logprobs = self.act.get_action_logprob(next_ss)  # next actions
            # except ValueError:
            #     print(next_ss)
            #     print(states, actions, rewards, undones)
                # else:
                #     break
            next_qs = self.cri_target.get_q_min(next_ss, next_as)  # next q values

            alpha = self.alpha_log.exp().detach()
            q_labels = rewards + undones * self.gamma * (next_qs - next_logprobs * alpha)

        q1, q2 = self.cri.get_q1_q2(states, actions)
        obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)  # twin critics
        return obj_critic, states

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            states, actions, rewards, undones, next_ss, is_weights, is_indices = buffer.sample_for_per(batch_size)
            next_as, next_logprobs = self.act.get_action_logprob(next_ss)
            next_qs = self.cri_target.get_q_min(next_ss, next_as)

            alpha = self.alpha_log.exp().detach()
            q_labels = rewards + undones * self.gamma * (next_qs - next_logprobs * alpha)

        q1, q2 = self.cri.get_q1_q2(states, actions)
        td_errors = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)
        obj_critic = (td_errors * is_weights).mean()

        buffer.td_error_update_for_per(is_indices.detach(), td_errors.detach())
        return obj_critic, states


class AgentSACLeadTimeNstep(AgentSAC):
    """Opt-in SAC critic with a lead-time-aware n-step Bellman target.

    The actor, temperature update, optimizers, and network classes remain the
    same as :class:`AgentSAC`.  Only the critic target changes from one-step TD
    to an episode-safe n-step reward return plus a soft-Q bootstrap at ``t+n``.
    """

    def __init__(self, net_dims, state_dim, action_dim, gpu_id=0, args=Config()):
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.critic_n_step = int(getattr(args, "critic_n_step", 1) or 1)
        if self.critic_n_step < 2:
            raise ValueError(
                "AgentSACLeadTimeNstep requires critic_n_step >= 2; "
                f"got {self.critic_n_step}."
            )
        if self.if_use_per:
            raise NotImplementedError(
                "Prioritized replay is not implemented for lead-time-aware n-step SAC."
            )

    def get_obj_critic_raw(self, buffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            states, actions, n_step_returns, bootstrap_discounts, next_ss = buffer.sample_n_step(
                batch_size=batch_size,
                n_step=self.critic_n_step,
                gamma=self.gamma,
            )
            next_as, next_logprobs = self.act.get_action_logprob(next_ss)
            next_qs = self.cri_target.get_q_min(next_ss, next_as)
            alpha = self.alpha_log.exp().detach()
            q_labels = n_step_returns + bootstrap_discounts * (
                next_qs - next_logprobs * alpha
            )

        q1, q2 = self.cri.get_q1_q2(states, actions)
        obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)
        return obj_critic, states


class AgentSACLeadTimeTDLambda(AgentSAC):
    """Opt-in SAC critic using an episode-safe truncated TD(lambda) target."""

    def __init__(self, net_dims, state_dim, action_dim, gpu_id=0, args=Config()):
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.critic_lambda_horizon = int(
            getattr(args, "critic_lambda_horizon", 9) or 9
        )
        self.critic_trace_lambda = float(
            getattr(args, "critic_trace_lambda", 0.8)
        )
        if self.critic_lambda_horizon < 1:
            raise ValueError("critic_lambda_horizon must be positive")
        if not 0.0 <= self.critic_trace_lambda <= 1.0:
            raise ValueError("critic_trace_lambda must be in [0, 1]")
        if self.if_use_per:
            raise NotImplementedError(
                "Prioritized replay is not implemented for TD(lambda) SAC."
            )

    def get_obj_critic_raw(self, buffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            states, actions, rewards, undones, next_states = buffer.sample_sequences(
                batch_size=batch_size,
                horizon=self.critic_lambda_horizon,
            )
            flat_next_states = next_states.reshape((-1, self.state_dim))
            flat_next_actions, flat_next_logprobs = self.act.get_action_logprob(
                flat_next_states
            )
            flat_next_qs = self.cri_target.get_q_min(
                flat_next_states, flat_next_actions
            )
            alpha = self.alpha_log.exp().detach()
            next_soft_values = (
                flat_next_qs - flat_next_logprobs * alpha
            ).reshape((batch_size, self.critic_lambda_horizon))
            q_labels = build_truncated_td_lambda_target(
                rewards=rewards,
                undones=undones,
                next_soft_values=next_soft_values,
                gamma=self.gamma,
                trace_lambda=self.critic_trace_lambda,
            )

        q1, q2 = self.cri.get_q1_q2(states, actions)
        obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)
        return obj_critic, states


class AgentSACResidualSafe(AgentSAC):
    """SAC variant for conservative fine-tuning around an RB teacher.

    All deviations from :class:`AgentSAC` are opt-in: separate actor/critic
    learning rates, an initial critic-only phase, delayed policy updates, a
    persistent behavior-cloning anchor, and a fixed fraction of RB trajectories
    refreshed into replay.  The class is deliberately separate so historical
    SAC experiments retain their exact update semantics.
    """

    def __init__(self, net_dims, state_dim, action_dim, gpu_id=0, args=Config()):
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.actor_learning_rate = float(
            getattr(args, "actor_learning_rate", None) or self.learning_rate
        )
        self.critic_learning_rate = float(
            getattr(args, "critic_learning_rate", None) or self.learning_rate
        )
        for group in self.act_optimizer.param_groups:
            group["lr"] = self.actor_learning_rate
        for group in self.cri_optimizer.param_groups:
            group["lr"] = self.critic_learning_rate

        self.actor_freeze_steps = int(getattr(args, "actor_freeze_steps", 0) or 0)
        self.actor_update_interval = int(
            getattr(args, "actor_update_interval", 1) or 1
        )
        if self.actor_freeze_steps < 0 or self.actor_update_interval < 1:
            raise ValueError("actor freeze/update interval settings are invalid")

        self.rb_anchor_coefficient_initial = float(
            getattr(args, "rb_anchor_coefficient_initial", 0.0) or 0.0
        )
        self.rb_anchor_coefficient_final = float(
            getattr(args, "rb_anchor_coefficient_final", 0.0) or 0.0
        )
        self.rb_anchor_decay_steps = int(
            getattr(args, "rb_anchor_decay_steps", 0) or 0
        )
        self.rb_anchor_batch_size = int(
            getattr(args, "rb_anchor_batch_size", 256) or 256
        )
        self.rb_teacher_replay_fraction = float(
            getattr(args, "rb_teacher_replay_fraction", 0.0) or 0.0
        )
        teacher_final = getattr(args, "rb_teacher_replay_fraction_final", None)
        self.rb_teacher_replay_fraction_final = (
            self.rb_teacher_replay_fraction
            if teacher_final is None
            else float(teacher_final)
        )
        self.rb_teacher_replay_decay_steps = int(
            getattr(args, "rb_teacher_replay_decay_steps", 0) or 0
        )
        if not (
            0.0 <= self.rb_teacher_replay_fraction < 1.0
            and 0.0 <= self.rb_teacher_replay_fraction_final < 1.0
        ):
            raise ValueError("RB teacher replay fractions must be in [0, 1)")
        if self.rb_teacher_replay_decay_steps < 0:
            raise ValueError("rb_teacher_replay_decay_steps must be nonnegative")

        self.training_env_steps = 0
        self.critic_update_count = 0
        self.rb_replay_pointer = 0
        self.rb_anchor_states = torch.empty((0, state_dim), device=self.device)
        self.rb_anchor_actions = torch.empty((0, action_dim), device=self.device)
        self.rb_replay_states = torch.empty((0, 1, state_dim), device=self.device)
        self.rb_replay_actions = torch.empty((0, 1, action_dim), device=self.device)
        self.rb_replay_rewards = torch.empty((0, 1), device=self.device)
        self.rb_replay_undones = torch.empty((0, 1), device=self.device)
        self.save_attr_names.update({
            "training_env_steps", "critic_update_count", "rb_replay_pointer",
            "rb_anchor_states", "rb_anchor_actions", "rb_replay_states",
            "rb_replay_actions", "rb_replay_rewards", "rb_replay_undones",
        })

    def set_rb_guidance(self, replay_items, anchor_states, anchor_actions):
        self.rb_replay_states = replay_items[0].detach().to(self.device)
        self.rb_replay_actions = replay_items[1].detach().to(self.device)
        self.rb_replay_rewards = replay_items[2].detach().to(self.device)
        self.rb_replay_undones = replay_items[3].detach().to(self.device)
        self.rb_anchor_states = anchor_states.reshape(-1, self.state_dim).detach().to(self.device)
        self.rb_anchor_actions = anchor_actions.reshape(-1, self.action_dim).detach().to(self.device)
        self.rb_replay_pointer = 0

    def mix_rb_replay_items(self, online_items):
        """Prepend a contiguous teacher segment while preserving transition order."""
        fraction = self._teacher_replay_fraction()
        teacher_size = int(self.rb_replay_rewards.shape[0])
        online_size = int(online_items[2].shape[0])
        if fraction <= 0.0 or teacher_size < 2 or online_size < 1:
            return online_items

        teacher_count = max(1, int(round(online_size * fraction / (1.0 - fraction))))
        teacher_count = min(teacher_count, teacher_size - 1)
        max_start = teacher_size - teacher_count
        start = self.rb_replay_pointer % max_start
        stop = start + teacher_count
        self.rb_replay_pointer = stop % max_start
        teacher = [
            self.rb_replay_states[start:stop],
            self.rb_replay_actions[start:stop],
            self.rb_replay_rewards[start:stop],
            self.rb_replay_undones[start:stop].clone(),
        ]
        # ReplayBuffer infers next state from adjacency. Mark the splice as a
        # terminal boundary so the last teacher transition cannot bootstrap
        # from the first online state.
        teacher[3][-1] = 0.0
        return tuple(torch.cat((teacher[i], online_items[i]), dim=0) for i in range(4))

    def _teacher_replay_fraction(self) -> float:
        if self.rb_teacher_replay_decay_steps <= 0:
            return self.rb_teacher_replay_fraction_final
        progress = min(1.0, self.training_env_steps / self.rb_teacher_replay_decay_steps)
        return (
            self.rb_teacher_replay_fraction * (1.0 - progress)
            + self.rb_teacher_replay_fraction_final * progress
        )

    def _anchor_coefficient(self) -> float:
        if self.rb_anchor_decay_steps <= 0:
            return self.rb_anchor_coefficient_final
        progress = min(1.0, self.training_env_steps / self.rb_anchor_decay_steps)
        return (
            self.rb_anchor_coefficient_initial * (1.0 - progress)
            + self.rb_anchor_coefficient_final * progress
        )

    def _anchor_loss(self) -> Tensor:
        count = int(self.rb_anchor_states.shape[0])
        if count == 0:
            return torch.zeros((), device=self.device)
        batch_size = min(self.rb_anchor_batch_size, count)
        ids = torch.randint(count, (batch_size,), device=self.device)
        predicted = self.act(self.rb_anchor_states[ids])
        return torch.nn.functional.mse_loss(predicted, self.rb_anchor_actions[ids])

    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
        online_add_size = int(getattr(buffer, "online_add_size", buffer.add_size))
        self.training_env_steps += online_add_size
        with torch.no_grad():
            states, _, rewards, undones = buffer.add_item
            self.update_avg_std_for_normalization(
                states=states.reshape((-1, self.state_dim)),
                returns=self.get_cumulative_rewards(rewards, undones).reshape((-1,)),
            )

        update_times = max(1, int(online_add_size * self.repeat_times))
        critic_sum = actor_sum = alpha_sum = 0.0
        actor_updates = 0
        for _ in range(update_times):
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            critic_sum += obj_critic.item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            self.critic_update_count += 1

            policy_due = (
                self.training_env_steps > self.actor_freeze_steps
                and self.critic_update_count % self.actor_update_interval == 0
            )
            if not policy_due:
                continue

            action_pg, log_prob = self.act.get_action_logprob(state)
            target_entropy = self.get_target_entropy(state)
            obj_alpha = (self.alpha_log * (target_entropy - log_prob).detach()).mean()
            self.optimizer_update(self.alpha_optimizer, obj_alpha)
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2)
            alpha = self.alpha_log.exp().detach()

            q_value_pg = self._actor_q_min(state, action_pg)
            obj_actor = (q_value_pg - log_prob * alpha).mean()
            anchor_coefficient = self._anchor_coefficient()
            actor_loss = -obj_actor
            if anchor_coefficient > 0.0:
                actor_loss = actor_loss + anchor_coefficient * self._anchor_loss()
            self.optimizer_update(self.act_optimizer, actor_loss)
            actor_sum += obj_actor.item()
            alpha_sum += alpha.item()
            actor_updates += 1

        alpha_value = float(self.alpha_log.exp().detach().item())
        return (
            critic_sum / update_times,
            actor_sum / max(actor_updates, 1),
            alpha_sum / max(actor_updates, 1) if actor_updates else alpha_value,
        )


class AgentSACMechanism(AgentSACResidualSafe):
    """Opt-in SAC with delayed actor updates and a configurable Q ensemble.

    No RB trajectories or behavior-cloning targets are used.  The class reuses
    the conservative update scheduler from :class:`AgentSACResidualSafe` while
    keeping all teacher-related coefficients at their inert defaults.
    """

    def __init__(self, net_dims, state_dim, action_dim, gpu_id=0, args=Config()):
        ensemble_size = int(getattr(args, "critic_ensemble_size", 2) or 2)
        target_min_size = int(getattr(args, "critic_target_min_size", 2) or 2)
        if ensemble_size < 2:
            raise ValueError("critic_ensemble_size must be at least two")
        if not 1 <= target_min_size <= ensemble_size:
            raise ValueError(
                "critic_target_min_size must be in [1, critic_ensemble_size]"
            )
        self.critic_ensemble_size = ensemble_size
        self.critic_target_min_size = target_min_size
        self.cri_class = partial(CriticEnsemble, ensemble_size=ensemble_size)
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        if self.rb_anchor_coefficient_initial or self.rb_anchor_coefficient_final:
            raise ValueError("AgentSACMechanism does not accept an RB actor anchor")
        if self.rb_teacher_replay_fraction or self.rb_teacher_replay_fraction_final:
            raise ValueError("AgentSACMechanism does not accept RB teacher replay")

    def _target_q(self, next_states: Tensor, next_actions: Tensor) -> Tensor:
        values = self.cri_target.get_q_values(next_states, next_actions)
        if self.critic_target_min_size == self.critic_ensemble_size:
            selected = values
        else:
            indices = torch.randperm(values.shape[1], device=values.device)[
                :self.critic_target_min_size
            ]
            selected = values.index_select(dim=1, index=indices)
        return selected.min(dim=1).values

    def get_obj_critic_raw(self, buffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            states, actions, rewards, undones, next_states = buffer.sample(batch_size)
            next_actions, next_logprobs = self.act.get_action_logprob(next_states)
            next_qs = self._target_q(next_states, next_actions)
            alpha = self.alpha_log.exp().detach()
            q_labels = rewards + undones * self.gamma * (
                next_qs - next_logprobs * alpha
            )

        q_values = self.cri.get_q_values(states, actions)
        losses = torch.stack(
            [self.criterion(q_values[:, index], q_labels) for index in range(q_values.shape[1])]
        )
        # Preserve the historical twin-Q loss scale when ensemble_size=2.
        obj_critic = losses.mean() * 2.0
        return obj_critic, states

    def get_obj_critic_per(self, buffer, batch_size: int):
        raise NotImplementedError("PER is not used by AgentSACMechanism experiments")


class AgentRecommendedSAC(AgentSAC):
    """RA-SAC recommended training with a feasible-action critic.

    The actor outputs a virtual action in [-1, 1]. A differentiable soft
    Lagrangian reallocation layer maps the virtual action to feasible order
    quantities. The environment executes the feasible quantities, and the critic
    is trained on the reallocated feasible action.
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        self.buffer_class = ReplayBufferRecommendedSAC
        self._init_reallocation_tensors(args)

    def _init_reallocation_tensors(self, args: Config):
        env_args = getattr(args, 'env_args', None) or {}
        info_input = env_args.get('info_input', None)
        if info_input is None:
            raise ValueError("AgentRecommendedSAC requires env_args['info_input'] for differentiable reallocation.")

        self.rdc_keys = list(info_input.rdc_dict.keys())
        self.rdc_size = len(self.rdc_keys)
        self.rdc_slices = []
        cursor = 0
        for k in self.rdc_keys:
            length = int(info_input.rdc_dict[k]) + 1
            self.rdc_slices.append((cursor, cursor + length))
            cursor += length
        self.cdc_slice = (cursor, cursor + int(info_input.item_lead_time) + 1)

        demand_type = env_args.get('demand_type', 'poisson')
        rdc_min_bsl, rdc_max_bsl, demand_para, demand_sigma = [], [], [], []
        quantile_tables = []
        quantile_p_grid = None
        quantile_grid_size = int(getattr(args, 'reallocation_quantile_grid_size', 1025))
        self.quantile_tau = float(getattr(args, 'reallocation_quantile_tau', 0.25))
        self.prob_eps = float(getattr(args, 'reallocation_prob_eps', 1e-5))
        for k in self.rdc_keys:
            lead_periods = int(info_input.rdc_dict[k]) + 1
            if demand_type == 'quantile':
                demand_info = info_input.demand_feature[k]
                max_demand = sum([demand_info['0.9'][w] for w in range(13)]) / 13 / 7
                min_demand = sum([demand_info['0.1'][w] for w in range(13)]) / 13 / 7
                mean_demand = 0.5 * (max_demand + min_demand)
                std_demand = math.sqrt(max(mean_demand, 1e-4))
                rdc_min_bsl.append(lead_periods * min_demand)
                rdc_max_bsl.append(lead_periods * max_demand)
            else:
                demand_spec = info_input.demand_feature[k]
                mean_demand = demand_mean(demand_spec)
                std_demand = demand_std(demand_spec)
                rdc_min_bsl.append(lead_time_demand_quantile(demand_spec, 0.01, lead_periods))
                rdc_max_bsl.append(lead_time_demand_quantile(demand_spec, 0.99, lead_periods))
                p_grid, q_grid = smooth_lead_time_quantile_table(
                    demand_spec,
                    lead_periods,
                    grid_size=quantile_grid_size,
                    tau=self.quantile_tau,
                    prob_eps=self.prob_eps,
                )
                if quantile_p_grid is None:
                    quantile_p_grid = p_grid
                quantile_tables.append(q_grid)
            demand_para.append(mean_demand * lead_periods)
            demand_sigma.append(std_demand * math.sqrt(lead_periods))

        cdc_min_bsl = float(np.sum(rdc_min_bsl))
        cdc_max_bsl = float(np.sum(rdc_max_bsl))
        if demand_type == 'quantile':
            max_demand_sum = 0.0
            min_demand_sum = 0.0
            for k in self.rdc_keys:
                demand_info = info_input.demand_feature[k]
                max_demand_sum += sum([demand_info['0.9'][w] for w in range(13)]) / 13 / 7
                min_demand_sum += sum([demand_info['0.1'][w] for w in range(13)]) / 13 / 7
            cdc_max_bsl += (int(info_input.item_lead_time) + 1) * max_demand_sum
            cdc_min_bsl += (int(info_input.item_lead_time) + 1) * min_demand_sum
        else:
            cdc_periods = int(info_input.item_lead_time) + 1
            max_demand_sum = sum([
                lead_time_demand_quantile(info_input.demand_feature[k], 0.99, cdc_periods)
                for k in self.rdc_keys
            ])
            min_demand_sum = sum([
                lead_time_demand_quantile(info_input.demand_feature[k], 0.01, cdc_periods)
                for k in self.rdc_keys
            ])
            cdc_max_bsl += max_demand_sum
            cdc_min_bsl += min_demand_sum

        self.rdc_min_bsl = torch.as_tensor(rdc_min_bsl, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.rdc_max_bsl = torch.as_tensor(rdc_max_bsl, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.cdc_min_bsl = torch.as_tensor([cdc_min_bsl], dtype=torch.float32, device=self.device).unsqueeze(0)
        self.cdc_max_bsl = torch.as_tensor([cdc_max_bsl], dtype=torch.float32, device=self.device).unsqueeze(0)

        self.demand_mu = torch.as_tensor(demand_para, dtype=torch.float32, device=self.device).unsqueeze(0).clamp_min(1e-4)
        self.demand_sigma = torch.as_tensor(demand_sigma, dtype=torch.float32, device=self.device).unsqueeze(0).clamp_min(1.0)
        self.use_quantile_table = bool(quantile_tables)
        if self.use_quantile_table:
            self.quantile_p_grid = torch.as_tensor(quantile_p_grid, dtype=torch.float32, device=self.device)
            self.quantile_q_grid = torch.as_tensor(np.stack(quantile_tables, axis=0),
                                                   dtype=torch.float32, device=self.device)
            self.quantile_rdc_index = torch.arange(self.rdc_size, device=self.device).unsqueeze(0)
        self.holding_cost = torch.as_tensor([info_input.holding_cost[k] for k in self.rdc_keys],
                                            dtype=torch.float32, device=self.device).unsqueeze(0).clamp_min(1e-6)
        self.backorder_cost = torch.as_tensor([info_input.backorder_cost[k] for k in self.rdc_keys],
                                              dtype=torch.float32, device=self.device).unsqueeze(0).clamp_min(1e-6)

        order_scale = list(np.maximum(rdc_max_bsl, 1.0)) + [max(cdc_max_bsl, 1.0)]
        self.order_scale = torch.as_tensor(order_scale, dtype=torch.float32, device=self.device).unsqueeze(0)
        lambda_margin = float((self.holding_cost + self.backorder_cost).max().item() * 1e-4 + 1e-6)
        self.lambda_low_bound = -float(self.holding_cost.max().item()) - lambda_margin
        self.lambda_high_bound = float(self.backorder_cost.max().item()) + lambda_margin
        self.lambda_span = max(abs(self.lambda_low_bound), abs(self.lambda_high_bound))
        self.reallocation_iters = int(getattr(args, 'reallocation_iters', 32))
        self.softplus_tau = float(getattr(args, 'reallocation_softplus_tau', 1.0))
        self.implicit_lambda_eps = float(getattr(args, 'reallocation_implicit_lambda_eps', 1e-6))

    def _parse_inventory_positions(self, states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        rdc_positions = [states[:, beg:end].sum(dim=1) for beg, end in self.rdc_slices]
        rdc_positions = torch.stack(rdc_positions, dim=1)
        cdc_beg, cdc_end = self.cdc_slice
        cdc_inventory = states[:, cdc_beg:cdc_end]
        cdc_on_hand = cdc_inventory[:, 0:1]
        cdc_position = rdc_positions.sum(dim=1, keepdim=True) + cdc_inventory.sum(dim=1, keepdim=True)
        return rdc_positions, cdc_on_hand, cdc_position

    def _smooth_positive(self, value: Tensor) -> Tensor:
        tau = self.softplus_tau
        return torch.nn.functional.softplus(value / tau) * tau

    def _table_quantile_from_prob(self, prob: Tensor) -> Tuple[Tensor, Tensor]:
        prob = prob.clamp(0.0, 1.0)
        idx = torch.searchsorted(self.quantile_p_grid, prob.reshape(-1)).reshape_as(prob)
        idx = idx.clamp(1, self.quantile_p_grid.shape[0] - 1)
        idx_low = idx - 1

        rdc_idx = self.quantile_rdc_index.expand_as(idx)
        p_low = self.quantile_p_grid[idx_low]
        p_high = self.quantile_p_grid[idx]
        q_low = self.quantile_q_grid[rdc_idx, idx_low]
        q_high = self.quantile_q_grid[rdc_idx, idx]

        dq_dp = (q_high - q_low) / (p_high - p_low).clamp_min(1e-12)
        quantile = q_low + (prob - p_low) * dq_dp
        return quantile, dq_dp

    def _demand_position_from_prob(self, prob: Tensor) -> Tuple[Tensor, Tensor]:
        if self.use_quantile_table:
            return self._table_quantile_from_prob(prob)

        normal_quantile = torch.erfinv(prob * 2.0 - 1.0) * math.sqrt(2.0)
        position = self.demand_mu + self.demand_sigma * normal_quantile
        normal_pdf = torch.exp(-0.5 * normal_quantile.square()) / math.sqrt(2.0 * math.pi)
        dposition_dprob = self.demand_sigma / normal_pdf.clamp_min(1e-12)
        return position, dposition_dprob

    def _allocation_given_lambda(self, lambda_value: Tensor, rdc_positions: Tensor, upper_alloc: Tensor) -> Tensor:
        raw_prob = (self.backorder_cost - lambda_value) / (self.backorder_cost + self.holding_cost)
        prob = raw_prob.clamp(0.0, 1.0) if self.use_quantile_table \
            else raw_prob.clamp(self.prob_eps, 1.0 - self.prob_eps)
        unconstrained_position, _ = self._demand_position_from_prob(prob)
        lower = rdc_positions
        upper = rdc_positions + upper_alloc
        position = torch.max(torch.min(unconstrained_position, upper), lower)
        return position - rdc_positions

    def _allocation_total_slope_lambda(self, lambda_value: Tensor, rdc_positions: Tensor, upper_alloc: Tensor) -> Tensor:
        raw_prob = (self.backorder_cost - lambda_value) / (self.backorder_cost + self.holding_cost)
        prob = raw_prob.clamp(0.0, 1.0) if self.use_quantile_table \
            else raw_prob.clamp(self.prob_eps, 1.0 - self.prob_eps)
        unconstrained_position, dposition_dprob = self._demand_position_from_prob(prob)

        lower = rdc_positions
        upper = rdc_positions + upper_alloc
        in_prob = (raw_prob > 0.0) & (raw_prob < 1.0) if self.use_quantile_table \
            else (raw_prob > self.prob_eps) & (raw_prob < 1.0 - self.prob_eps)
        in_bounds = (unconstrained_position > lower) & (unconstrained_position < upper)

        slope = -dposition_dprob / (self.backorder_cost + self.holding_cost)
        slope = torch.where(in_prob & in_bounds, slope, torch.zeros_like(slope))
        return slope.sum(dim=1, keepdim=True)

    def _lambda_with_implicit_gradient(
            self,
            lambda_forward: Tensor,
            rdc_positions: Tensor,
            upper_alloc: Tensor,
            target_capacity: Tensor,
    ) -> Tensor:
        lambda_forward = lambda_forward.detach()
        alloc_forward = self._allocation_given_lambda(lambda_forward, rdc_positions, upper_alloc)
        residual = alloc_forward.sum(dim=1, keepdim=True) - target_capacity

        slope = self._allocation_total_slope_lambda(
            lambda_forward,
            rdc_positions.detach(),
            upper_alloc.detach(),
        ).detach()
        valid_slope = slope.abs() > self.implicit_lambda_eps
        safe_slope = torch.where(valid_slope, slope, torch.ones_like(slope))

        max_correction = self.lambda_high_bound - self.lambda_low_bound
        correction = (residual / safe_slope).clamp(-max_correction, max_correction)
        lambda_implicit = lambda_forward - correction
        lambda_implicit = lambda_implicit.clamp(self.lambda_low_bound, self.lambda_high_bound)
        lambda_for_grad = lambda_forward + (lambda_implicit - lambda_implicit.detach())
        return torch.where(valid_slope, lambda_for_grad, lambda_forward)

    def soft_lagrangian_reallocation(self, states: Tensor, raw_actions: Tensor) -> Tuple[Tensor, Tensor]:
        rdc_positions, cdc_on_hand, cdc_position = self._parse_inventory_positions(states)
        action_mask = self.get_action_mask(states)
        if action_mask is None:
            action_mask = torch.ones_like(raw_actions)
        rdc_mask = action_mask[:, :self.rdc_size]
        cdc_mask = action_mask[:, -1:]
        normalized_action = (raw_actions + 1.0) * 0.5

        target_rdc = self.rdc_min_bsl + normalized_action[:, :self.rdc_size] * (self.rdc_max_bsl - self.rdc_min_bsl)
        target_cdc = self.cdc_min_bsl + normalized_action[:, -1:] * (self.cdc_max_bsl - self.cdc_min_bsl)

        upper_alloc = self._smooth_positive(target_rdc - rdc_positions) * rdc_mask
        capacity = torch.relu(cdc_on_hand)
        total_upper = upper_alloc.sum(dim=1, keepdim=True)
        target_capacity = torch.minimum(capacity, total_upper)

        lambda_low = torch.full_like(target_capacity, self.lambda_low_bound)
        lambda_high = torch.full_like(target_capacity, self.lambda_high_bound)
        for _ in range(self.reallocation_iters):
            lambda_mid = (lambda_low + lambda_high) * 0.5
            alloc_mid = self._allocation_given_lambda(lambda_mid, rdc_positions, upper_alloc)
            total_mid = alloc_mid.sum(dim=1, keepdim=True)
            too_much = total_mid > target_capacity
            lambda_low = torch.where(too_much, lambda_mid, lambda_low)
            lambda_high = torch.where(too_much, lambda_high, lambda_mid)

        lambda_star = self._lambda_with_implicit_gradient(
            (lambda_low + lambda_high) * 0.5,
            rdc_positions,
            upper_alloc,
            target_capacity,
        )
        lagrangian_alloc = self._allocation_given_lambda(lambda_star, rdc_positions, upper_alloc)
        feasible_rdc = torch.where(total_upper <= capacity + 1e-6, upper_alloc, lagrangian_alloc)
        feasible_cdc = self._smooth_positive(target_cdc - cdc_position) * cdc_mask
        feasible_action = torch.cat((feasible_rdc, feasible_cdc), dim=1)
        feasible_action_norm = self.apply_action_mask(
            states, feasible_action / self.order_scale * 2.0 - 1.0
        )
        return feasible_action_norm, feasible_action

    def explore_one_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        raw_actions = torch.zeros((horizon_len, self.num_envs, self.action_dim), dtype=torch.float32).to(self.device)
        feasible_actions = torch.zeros_like(raw_actions)
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        state = self.last_state
        get_action = self.act.get_action
        for t in range(horizon_len):
            raw_action = torch.rand(1, self.action_dim, device=self.device) * 2 - 1.0 if if_random else get_action(state)
            raw_action = self.apply_action_mask(state, raw_action)
            feasible_action_norm, feasible_action = self.soft_lagrangian_reallocation(state, raw_action)

            states[t] = state
            ary_action = feasible_action[0].detach().cpu().numpy()
            ary_state, reward, done, _ = env.step_feasible_action(ary_action)
            ary_state = env.reset() if done else ary_state
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device).unsqueeze(0)

            raw_actions[t] = raw_action
            feasible_actions[t] = feasible_action_norm.detach()
            rewards[t] = reward
            dones[t] = done

        self.last_state = state
        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, raw_actions, feasible_actions, rewards, undones

    def update_net(self, buffer: ReplayBufferRecommendedSAC) -> Tuple[float, ...]:
        with torch.no_grad():
            states, _, _, rewards, undones = buffer.add_item
            self.update_avg_std_for_normalization(
                states=states.reshape((-1, self.state_dim)),
                returns=self.get_cumulative_rewards(rewards=rewards, undones=undones).reshape((-1,))
            )

        obj_critics = 0.0
        obj_actors = 0.0
        alphas = 0.0

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        for _ in range(update_times):
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            raw_action_pg, log_prob = self.act.get_action_logprob(state)
            target_entropy = self.get_target_entropy(state)
            obj_alpha = (self.alpha_log * (target_entropy - log_prob).detach()).mean()
            self.optimizer_update(self.alpha_optimizer, obj_alpha)

            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2)
            alpha = self.alpha_log.exp().detach()
            alphas += alpha.item()

            feasible_action_pg, _ = self.soft_lagrangian_reallocation(state, raw_action_pg)
            q_value_pg = self._actor_q_min(state, feasible_action_pg)
            obj_actor = (q_value_pg - log_prob * alpha).mean()
            obj_actors += obj_actor.item()
            self.optimizer_update(self.act_optimizer, -obj_actor)

        return obj_critics / update_times, obj_actors / update_times, alphas / update_times

    def get_obj_critic_raw(self, buffer: ReplayBufferRecommendedSAC, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            states, _, feasible_actions, rewards, undones, next_ss = buffer.sample(batch_size)
            next_raw_actions, next_logprobs = self.act.get_action_logprob(next_ss)
            next_feasible_actions, _ = self.soft_lagrangian_reallocation(next_ss, next_raw_actions)
            next_qs = self.cri_target.get_q_min(next_ss, next_feasible_actions)

            alpha = self.alpha_log.exp().detach()
            q_labels = rewards + undones * self.gamma * (next_qs - next_logprobs * alpha)

        q1, q2 = self.cri.get_q1_q2(states, feasible_actions)
        obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)
        return obj_critic, states


class AgentPenaltyANSAC(AgentRecommendedSAC):
    """Penalty-AN-SAC with proportional action normalization.

    The actor outputs relaxed base-stock targets. A torch proportional
    allocation layer converts them to feasible order quantities. The critic is
    trained on the original actor output, while a fixed penalty in its target
    discourages relaxed warehouse-availability violations.
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        self.buffer_class = ReplayBufferANSAC
        self.pa_eps = float(getattr(args, 'an_eps', getattr(args, 'pa_eps', 1e-6)))
        self.pa_violation_eps = float(getattr(args, 'an_violation_eps', getattr(args, 'pa_violation_eps', 1.0)))
        self.pa_penalty_coef = float(getattr(args, 'an_penalty_coef', getattr(args, 'pa_penalty_coef', 1.0)))
        self.pa_penalty_power = float(getattr(args, 'an_penalty_power', getattr(args, 'pa_penalty_power', 2.0)))
        self.pa_positive_tau = float(getattr(args, 'an_positive_tau', getattr(args, 'pa_positive_tau', 0.0)))
        self.pa_penalty_scale_by_reward = bool(
            getattr(args, 'an_penalty_scale_by_reward', getattr(args, 'pa_penalty_scale_by_reward', True))
        )
        self.critic_uses_feasible_action = False

    def _pa_positive(self, value: Tensor) -> Tensor:
        if self.pa_positive_tau > 0:
            tau = self.pa_positive_tau
            return torch.nn.functional.softplus(value / tau) * tau
        return torch.relu(value)

    @staticmethod
    def _largest_remainder_integer_ste(
            continuous_allocation: Tensor,
            capacity: Tensor,
    ) -> Tensor:
        """Use integer actions in the forward pass and continuous gradients backward."""
        quantities = continuous_allocation.clamp_min(0.0)
        integer_capacity = torch.floor(capacity.clamp_min(0.0) + 1e-9)
        rounded_total = torch.floor(quantities.sum(dim=1, keepdim=True) + 0.5)
        target_total = torch.minimum(integer_capacity, rounded_total).clamp_min(0.0)

        floors = torch.floor(quantities)
        residual = (target_total - floors.sum(dim=1, keepdim=True)).clamp_min(0.0).to(torch.long)
        fractions = quantities - floors
        order = torch.argsort(fractions, dim=1, descending=True, stable=True)
        ranks = torch.empty_like(order)
        rank_values = torch.arange(order.shape[1], device=order.device).expand_as(order)
        ranks.scatter_(1, order, rank_values)
        increments = (ranks < residual).to(quantities.dtype)
        integer_allocation = floors + increments
        return quantities + (integer_allocation - quantities).detach()

    @staticmethod
    def _nearest_integer_ste(quantity: Tensor) -> Tensor:
        nonnegative = quantity.clamp_min(0.0)
        integer_quantity = torch.floor(nonnegative + 0.5)
        return nonnegative + (integer_quantity - nonnegative).detach()

    def soft_proportional_allocation(self, states: Tensor, raw_actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        rdc_positions, cdc_on_hand, cdc_position = self._parse_inventory_positions(states)
        action_mask = self.get_action_mask(states)
        if action_mask is None:
            action_mask = torch.ones_like(raw_actions)
        rdc_mask = action_mask[:, :self.rdc_size]
        cdc_mask = action_mask[:, -1:]
        normalized_action = (raw_actions + 1.0) * 0.5

        target_rdc = self.rdc_min_bsl + normalized_action[:, :self.rdc_size] * (self.rdc_max_bsl - self.rdc_min_bsl)
        target_cdc = self.cdc_min_bsl + normalized_action[:, -1:] * (self.cdc_max_bsl - self.cdc_min_bsl)

        relaxed_rdc = self._pa_positive(target_rdc - rdc_positions) * rdc_mask
        capacity = torch.relu(cdc_on_hand)
        total_relaxed = relaxed_rdc.sum(dim=1, keepdim=True)
        scale = torch.minimum(
            torch.ones_like(total_relaxed),
            capacity / total_relaxed.clamp_min(self.pa_eps),
        )
        continuous_rdc = relaxed_rdc * scale
        feasible_rdc = self._largest_remainder_integer_ste(continuous_rdc, capacity)
        continuous_cdc = self._pa_positive(target_cdc - cdc_position) * cdc_mask
        feasible_cdc = self._nearest_integer_ste(continuous_cdc)
        feasible_action = torch.cat((feasible_rdc, feasible_cdc), dim=1)
        feasible_action_norm = self.apply_action_mask(
            states, feasible_action / self.order_scale * 2.0 - 1.0
        )

        denominator = (capacity + self.pa_violation_eps).clamp_min(self.pa_eps)
        violation = torch.relu(total_relaxed - capacity) / denominator
        return feasible_action_norm, feasible_action, violation

    def soft_lagrangian_reallocation(self, states: Tensor, raw_actions: Tensor) -> Tuple[Tensor, Tensor]:
        """Compatibility hook used by the existing evaluator.

        This agent does proportional allocation, not Lagrangian reallocation,
        but keeping the method name avoids changing the evaluator interface.
        """
        feasible_action_norm, feasible_action, _ = self.soft_proportional_allocation(states, raw_actions)
        return feasible_action_norm, feasible_action

    def _violation_penalty(self, violations: Tensor) -> Tensor:
        penalty = self.pa_penalty_coef * violations.clamp_min(0.0).pow(self.pa_penalty_power)
        if self.pa_penalty_scale_by_reward:
            penalty = penalty * self.reward_scale
        return penalty

    def _critic_policy_action(self, states: Tensor, raw_actions: Tensor) -> Tensor:
        if self.critic_uses_feasible_action:
            feasible_actions, _, _ = self.soft_proportional_allocation(states, raw_actions)
            return feasible_actions
        return raw_actions

    def explore_one_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        raw_actions = torch.zeros((horizon_len, self.num_envs, self.action_dim), dtype=torch.float32).to(self.device)
        feasible_actions = torch.zeros_like(raw_actions)
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        violations = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        state = self.last_state
        get_action = self.act.get_action
        for t in range(horizon_len):
            raw_action = torch.rand(1, self.action_dim, device=self.device) * 2 - 1.0 if if_random else get_action(state)
            feasible_action_norm, feasible_action, violation = self.soft_proportional_allocation(state, raw_action)

            states[t] = state
            ary_action = feasible_action[0].detach().cpu().numpy()
            ary_state, reward, done, _ = env.step_feasible_action(ary_action)
            ary_state = env.reset() if done else ary_state
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device).unsqueeze(0)

            raw_actions[t] = raw_action
            feasible_actions[t] = feasible_action_norm.detach()
            rewards[t] = reward
            violations[t] = violation.squeeze(-1).detach()
            dones[t] = done

        self.last_state = state
        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, raw_actions, feasible_actions, rewards, violations, undones

    def update_net(self, buffer: ReplayBufferANSAC) -> Tuple[float, ...]:
        with torch.no_grad():
            states, _, _, rewards, violations, undones = buffer.add_item
            train_rewards = rewards - self._violation_penalty(violations)
            self.update_avg_std_for_normalization(
                states=states.reshape((-1, self.state_dim)),
                returns=self.get_cumulative_rewards(rewards=train_rewards, undones=undones).reshape((-1,))
            )

        obj_critics = 0.0
        obj_actors = 0.0
        alphas = 0.0
        avg_violations = 0.0

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        for _ in range(update_times):
            obj_critic, state, sampled_violation = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            avg_violations += sampled_violation.mean().item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            raw_action_pg, log_prob = self.act.get_action_logprob(state)
            target_entropy = self.get_target_entropy(state)
            obj_alpha = (self.alpha_log * (target_entropy - log_prob).detach()).mean()
            self.optimizer_update(self.alpha_optimizer, obj_alpha)

            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2)
            alpha = self.alpha_log.exp().detach()
            alphas += alpha.item()

            critic_action_pg = self._critic_policy_action(state, raw_action_pg)
            q_value_pg = self._actor_q_min(state, critic_action_pg)
            obj_actor = (q_value_pg - log_prob * alpha).mean()
            obj_actors += obj_actor.item()
            self.optimizer_update(self.act_optimizer, -obj_actor)

        return (
            obj_critics / update_times,
            obj_actors / update_times,
            alphas / update_times,
            avg_violations / update_times,
        )

    def get_obj_critic_raw(self, buffer: ReplayBufferANSAC, batch_size: int) -> Tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            states, raw_actions, feasible_actions, rewards, violations, undones, next_ss = buffer.sample(batch_size)
            next_raw_actions, next_logprobs = self.act.get_action_logprob(next_ss)
            next_critic_actions = self._critic_policy_action(next_ss, next_raw_actions)
            next_qs = self.cri_target.get_q_min(next_ss, next_critic_actions)

            alpha = self.alpha_log.exp().detach()
            aug_rewards = rewards - self._violation_penalty(violations)
            q_labels = aug_rewards + undones * self.gamma * (next_qs - next_logprobs * alpha)

        critic_actions = feasible_actions if self.critic_uses_feasible_action else raw_actions
        q1, q2 = self.cri.get_q1_q2(states, critic_actions)
        obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)
        return obj_critic, states, violations


class AgentANSAC(AgentPenaltyANSAC):
    """Action-Normalized SAC.

    AN-SAC shares the feasible-action critic protocol with RA-SAC but replaces
    the cost-aware reallocation layer with proportional action normalization.
    Unlike Penalty-AN-SAC, it uses no relaxed-action penalty in the SAC objective.
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        self.pa_penalty_coef = 0.0
        self.critic_uses_feasible_action = True


class AgentLagANSAC(AgentPenaltyANSAC):
    """Lag-AN-SAC with dynamic relaxed-violation penalty.

    The executed action is generated by the same proportional normalization
    layer as AN-SAC. The critic uses the original actor output; the multiplier
    is updated from sampled violations and applied when forming critic targets.
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        self.lag_delta = float(getattr(args, 'lag_delta', 0.0))
        self.lag_lambda_lr = float(getattr(args, 'lag_lambda_lr', 1e-3))
        self.lag_lambda_max = float(getattr(args, 'lag_lambda_max', 1e6))
        self.lag_update_interval = max(1, int(getattr(args, 'lag_update_interval', 1)))
        self.lag_penalty_scale_by_reward = bool(getattr(args, 'lag_penalty_scale_by_reward', True))
        lag_lambda_init = float(getattr(args, 'lag_lambda_init', 0.0))
        self.lag_lambda = torch.tensor(lag_lambda_init, dtype=torch.float32, device=self.device)
        self.save_attr_names.add('lag_lambda')

    def _violation_penalty(self, violations: Tensor) -> Tensor:
        penalty = self.lag_lambda.detach() * violations.clamp_min(0.0)
        if self.lag_penalty_scale_by_reward:
            penalty = penalty * self.reward_scale
        return penalty

    def _update_lagrange_multiplier(self, violations: Tensor):
        with torch.no_grad():
            violation_gap = violations.detach().mean() - self.lag_delta
            next_lambda = self.lag_lambda + self.lag_lambda_lr * violation_gap
            next_lambda = next_lambda.clamp(0.0, self.lag_lambda_max)
            self.lag_lambda.copy_(next_lambda)

    def update_net(self, buffer: ReplayBufferANSAC) -> Tuple[float, ...]:
        with torch.no_grad():
            states, _, _, rewards, violations, undones = buffer.add_item
            train_rewards = rewards - self._violation_penalty(violations)
            self.update_avg_std_for_normalization(
                states=states.reshape((-1, self.state_dim)),
                returns=self.get_cumulative_rewards(rewards=train_rewards, undones=undones).reshape((-1,))
            )

        obj_critics = 0.0
        obj_actors = 0.0
        alphas = 0.0
        avg_violations = 0.0
        avg_lambdas = 0.0

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        for update_i in range(update_times):
            obj_critic, state, sampled_violation = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            avg_violations += sampled_violation.mean().item()
            if update_i % self.lag_update_interval == 0:
                self._update_lagrange_multiplier(sampled_violation)
            avg_lambdas += self.lag_lambda.item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            raw_action_pg, log_prob = self.act.get_action_logprob(state)
            target_entropy = self.get_target_entropy(state)
            obj_alpha = (self.alpha_log * (target_entropy - log_prob).detach()).mean()
            self.optimizer_update(self.alpha_optimizer, obj_alpha)

            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2)
            alpha = self.alpha_log.exp().detach()
            alphas += alpha.item()

            critic_action_pg = self._critic_policy_action(state, raw_action_pg)
            q_value_pg = self._actor_q_min(state, critic_action_pg)
            obj_actor = (q_value_pg - log_prob * alpha).mean()
            obj_actors += obj_actor.item()
            self.optimizer_update(self.act_optimizer, -obj_actor)

        return (
            obj_critics / update_times,
            obj_actors / update_times,
            alphas / update_times,
            avg_violations / update_times,
            avg_lambdas / update_times,
        )


AgentPenaltyPASAC = AgentPenaltyANSAC
AgentLagPASAC = AgentLagANSAC


class AgentModSAC(AgentSAC):  # Modified SAC using reliable_lambda and Two Time-scale Update Rule
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, "act_class", ActorFixSAC)
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        self.obj_c = 1.0  # for reliable_lambda

    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
        with torch.no_grad():
            states, actions, rewards, undones = buffer.add_item
            self.update_avg_std_for_normalization(
                states=states.reshape((-1, self.state_dim)),
                returns=self.get_cumulative_rewards(rewards=rewards, undones=undones).reshape((-1,))
            )

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0
        alphas = 0.0

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        update_a = 0
        for update_c in range(1, update_times + 1):
            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            self.obj_c = 0.995 * self.obj_c + 0.005 * obj_critic.item()  # for reliable_lambda

            reliable_lambda = math.exp(-self.obj_c ** 2)  # for reliable_lambda
            if update_a / update_c < 1 / (2 - reliable_lambda):  # auto TTUR
                '''objective of alpha (temperature parameter automatic adjustment)'''
                action_pg, log_prob = self.act.get_action_logprob(state)  # policy gradient
                target_entropy = self.get_target_entropy(state)
                obj_alpha = (self.alpha_log * (target_entropy - log_prob).detach()).mean()
                self.optimizer_update(self.alpha_optimizer, obj_alpha)

                '''objective of actor'''
                alpha = self.alpha_log.exp().detach()
                alphas += alpha.item()
                with torch.no_grad():
                    self.alpha_log[:] = self.alpha_log.clamp(-16, 2)

                q_value_pg = self.cri_target(state, action_pg).mean()
                obj_actor = (q_value_pg - log_prob * alpha).mean()
                obj_actors += obj_actor.item()
                self.optimizer_update(self.act_optimizer, -obj_actor)

        return obj_critics / update_times, obj_actors / update_times, alphas / update_times
