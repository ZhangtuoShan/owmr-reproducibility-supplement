import os
import torch
from copy import deepcopy
from typing import Tuple, Union
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from elegantrl.train import Config, ReplayBuffer
from elegantrl.train.device import resolve_torch_device


class AgentBase:
    """
    The basic agent of ElegantRL

    net_dims: the middle layer dimension of MLP (MultiLayer Perceptron)
    state_dim: the dimension of state (the number of state vector)
    action_dim: the dimension of action (or the number of discrete action)
    gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
    args: the arguments for agent training. `args = Config()`
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.gamma = args.gamma  # discount factor of future rewards
        self.num_envs = args.num_envs  # the number of sub envs in vectorized env. `num_envs=1` in single env.
        self.batch_size = args.batch_size  # num of transitions sampled from replay buffer.
        self.repeat_times = args.repeat_times  # repeatedly update network using ReplayBuffer
        self.reward_scale = args.reward_scale  # an approximate target reward usually be closed to 256
        self.learning_rate = args.learning_rate  # the learning rate for network updating
        self.if_off_policy = args.if_off_policy  # whether off-policy or on-policy of DRL algorithm
        self.clip_grad_norm = args.clip_grad_norm  # clip the gradient after normalization
        self.soft_update_tau = args.soft_update_tau  # the tau of soft target update `net = (1-tau)*net + net1`
        self.state_value_tau = args.state_value_tau  # the tau of normalize for value and state
        self.replay_store_after_period = int(
            getattr(args, 'replay_store_after_period', 0) or 0
        )
        if self.replay_store_after_period < 0:
            raise ValueError(
                "replay_store_after_period must be nonnegative, got "
                f"{self.replay_store_after_period}."
            )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_mask_state_start = getattr(args, 'action_mask_state_start', None)
        self.action_mask_dim = int(getattr(args, 'action_mask_dim', 0) or 0)
        self.inactive_action_value = float(getattr(args, 'inactive_action_value', 0.0))
        if self.action_mask_state_start is not None and self.action_mask_dim != action_dim:
            raise ValueError(
                "Continuous action mask must contain one entry per action dimension; "
                f"got mask_dim={self.action_mask_dim}, action_dim={action_dim}."
            )
        self.last_state = None  # last state of the trajectory for training. last_state.shape == (num_envs, state_dim)
        self.device = resolve_torch_device(getattr(args, "device", None), gpu_id)
        rollout_device_arg = getattr(args, "rollout_device", None)
        self.rollout_device = self.device if rollout_device_arg is None else resolve_torch_device(rollout_device_arg, gpu_id)
        self._rollout_act = None

        '''network'''
        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        actor_net_dims = getattr(args, "actor_net_dims", None) or net_dims
        critic_net_dims = getattr(args, "critic_net_dims", None) or net_dims
        self.act = self.act_target = act_class(actor_net_dims, state_dim, action_dim).to(self.device) \
            if act_class.__name__.split('.')[-1] != 'ActorDiscreteMultiPPO' else None
        self.cri = self.cri_target = cri_class(critic_net_dims, state_dim, action_dim).to(self.device) \
            if cri_class else self.act
        if self.act is not None and hasattr(self.act, 'set_action_mask'):
            self.act.set_action_mask(
                state_start=self.action_mask_state_start,
                mask_dim=self.action_mask_dim,
                inactive_action_value=self.inactive_action_value,
            )

        '''optimizer'''
        self.act_optimizer = torch.optim.AdamW(self.act.parameters(), self.learning_rate) \
            if act_class.__name__.split('.')[-1] != 'ActorDiscreteMultiPPO' else None
        self.cri_optimizer = torch.optim.AdamW(self.cri.parameters(), self.learning_rate) \
            if cri_class else self.act_optimizer
        from types import MethodType  # built-in package of Python3
        if act_class.__name__.split('.')[-1] != 'ActorDiscreteMultiPPO':
            self.act_optimizer.parameters = MethodType(get_optim_param, self.act_optimizer) \
            
        self.cri_optimizer.parameters = MethodType(get_optim_param, self.cri_optimizer)

        """attribute"""
        if self.num_envs == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

        self.if_use_per = getattr(args, 'if_use_per', None)  # use PER (Prioritized Experience Replay)
        if self.if_use_per:
            self.criterion = torch.nn.SmoothL1Loss(reduction="none")
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction="mean")
            self.get_obj_critic = self.get_obj_critic_raw

        """save and load"""
        self.save_attr_names = {'act', 'act_target', 'act_optimizer', 'cri', 'cri_target', 'cri_optimizer'}

    def _get_rollout_actor(self):
        if self.rollout_device == self.device:
            return self.act
        if self._rollout_act is None or self._rollout_act.__class__ is not self.act.__class__:
            self._rollout_act = deepcopy(self.act).to(self.rollout_device)
        rollout_state_dict = {
            key: value.detach().to(self.rollout_device) if isinstance(value, Tensor) else value
            for key, value in self.act.state_dict().items()
        }
        self._rollout_act.load_state_dict(rollout_state_dict)
        self._rollout_act.eval()
        return self._rollout_act

    def _get_rollout_state(self) -> Tensor:
        return self.last_state.to(self.rollout_device)

    def _set_last_rollout_state(self, state: Tensor):
        self.last_state = state.to(self.device).detach()

    def _rollout_items_to_train_device(self, items: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        if self.rollout_device == self.device:
            return items
        return tuple(item.to(self.device) if isinstance(item, Tensor) else item for item in items)

    def get_action_mask(self, states: Tensor):
        if self.action_mask_state_start is None:
            return None
        start = int(self.action_mask_state_start)
        end = start + self.action_mask_dim
        return states[..., start:end].clamp(0.0, 1.0)

    def apply_action_mask(self, states: Tensor, actions: Tensor) -> Tensor:
        mask = self.get_action_mask(states)
        if mask is None:
            return actions
        inactive = torch.full_like(actions, self.inactive_action_value)
        return actions * mask + inactive * (1.0 - mask)

    def explore_one_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        if_random: uses random action for warn-up exploration
        return: `(states, actions, rewards, undones)` for off-policy
            num_envs == 1
            states.shape == (horizon_len, num_envs, state_dim)
            actions.shape == (horizon_len, num_envs, action_dim)
            rewards.shape == (horizon_len, num_envs)
            undones.shape == (horizon_len, num_envs)
        """
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32, device=self.rollout_device)
        actions = torch.zeros((horizon_len, self.num_envs, self.action_dim), dtype=torch.float32, device=self.rollout_device)
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32, device=self.rollout_device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool, device=self.rollout_device)

        state = self._get_rollout_state()  # state.shape == (1, state_dim) for a single env.
        store_after = self.replay_store_after_period
        total_periods = int(getattr(env, 'total_periods', 0) or 0)
        if store_after and total_periods and store_after >= total_periods:
            raise ValueError(
                "replay_store_after_period must be smaller than the training "
                f"episode length, got {store_after} >= {total_periods}."
            )

        get_action = self._get_rollout_actor().get_action
        t = 0
        while t < horizon_len:
            period_before_step = int(getattr(env, 'period', store_after))
            store_transition = period_before_step >= store_after
            # try:
            action = torch.rand(1, self.action_dim, device=self.rollout_device) * 2 - 1.0 if if_random else get_action(state)
            action = self.apply_action_mask(state, action)
            # except ValueError:
            #     print(torch.isnan(state).any())
            #     print(state)
            if store_transition:
                states[t] = state

            ary_action = action[0].detach().cpu().numpy()
            ary_state, reward, done, _ = env.step(ary_action)  # next_state
            ary_state = env.reset() if done else ary_state  # ary_state.shape == (state_dim, )
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.rollout_device).unsqueeze(0)
            if store_transition:
                actions[t] = action
                rewards[t] = reward
                dones[t] = done
                t += 1

        self._set_last_rollout_state(state)  # state.shape == (1, state_dim) for a single env.

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return self._rollout_items_to_train_device((states, actions, rewards, undones))

    def explore_vec_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        if_random: uses random action for warn-up exploration
        return: `(states, actions, rewards, undones)` for off-policy
            states.shape == (horizon_len, num_envs, state_dim)
            actions.shape == (horizon_len, num_envs, action_dim)
            rewards.shape == (horizon_len, num_envs)
            undones.shape == (horizon_len, num_envs)
        """
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32, device=self.rollout_device)
        actions = torch.zeros((horizon_len, self.num_envs, self.action_dim), dtype=torch.float32, device=self.rollout_device)
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32, device=self.rollout_device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool, device=self.rollout_device)

        state = self._get_rollout_state()  # last_state.shape == (num_envs, state_dim)
        get_action = self._get_rollout_actor().get_action
        for t in range(horizon_len):
            action = torch.rand(self.num_envs, self.action_dim, device=self.rollout_device) * 2 - 1.0 if if_random \
                else get_action(state).detach()
            action = self.apply_action_mask(state, action)
            states[t] = state  # state.shape == (num_envs, state_dim)

            state, reward, done, _ = env.step(action)  # next_state
            actions[t] = action
            rewards[t] = reward
            dones[t] = done

        self._set_last_rollout_state(state)

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return self._rollout_items_to_train_device((states, actions, rewards, undones))

    def update_net(self, buffer: Union[ReplayBuffer, tuple]) -> Tuple[float, ...]:
        obj_critic = 0.0  # criterion(q_value, q_label).mean().item()
        obj_actor = 0.0  # q_value.mean().item()
        assert isinstance(buffer, ReplayBuffer) or isinstance(buffer, tuple)
        assert isinstance(self.batch_size, int)
        assert isinstance(self.repeat_times, int)
        assert isinstance(self.reward_scale, float)
        return obj_critic, obj_actor

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)  # next_ss: next states
            next_as = self.act_target(next_ss)  # next actions
            next_qs = self.cri_target(next_ss, next_as)  # next q values
            q_labels = rewards + undones * self.gamma * next_qs

        q_values = self.cri(states, actions)
        obj_critic = self.criterion(q_values, q_labels)
        return obj_critic, states

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            states, actions, rewards, undones, next_ss, is_weights, is_indices = buffer.sample_for_per(batch_size)
            # is_weights, is_indices: important sampling `weights, indices` by Prioritized Experience Replay (PER)

            next_as = self.act_target(next_ss)
            next_qs = self.cri_target(next_ss, next_as)
            q_labels = rewards + undones * self.gamma * next_qs

        q_values = self.cri(states, actions)
        td_errors = self.criterion(q_values, q_labels)
        obj_critic = (td_errors * is_weights).mean()

        buffer.td_error_update_for_per(is_indices.detach(), td_errors.detach())
        return obj_critic, states

    def get_cumulative_rewards(self, rewards: Tensor, undones: Tensor) -> Tensor:
        returns = torch.empty_like(rewards)

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        last_state = self.last_state
        next_action = self.act_target(last_state)
        next_value = self.cri_target(last_state, next_action).detach()
        for t in range(horizon_len - 1, -1, -1):
            returns[t] = next_value = rewards[t] + masks[t] * next_value
        return returns

    def optimizer_update(self, optimizer: torch.optim, objective: Tensor):
        """minimize the optimization objective via update the network parameters

        optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()


    def optimizer_update_constraint(self, optimizer: torch.optim, objective: Tensor, constraints):
        """minimize the optimization objective via update the network parameters

        optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()
        
    

    def optimizer_update_amp(self, optimizer: torch.optim, objective: Tensor):  # automatic mixed precision
        """minimize the optimization objective via update the network parameters

        amp: Automatic Mixed Precision

        optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        amp_scale = torch.cuda.amp.GradScaler()  # write in __init__()

        optimizer.zero_grad()
        amp_scale.scale(objective).backward()  # loss.backward()
        amp_scale.unscale_(optimizer)  # amp

        # from torch.nn.utils import clip_grad_norm_
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        amp_scale.step(optimizer)  # optimizer.step()
        amp_scale.update()  # optimizer.step()

    def update_avg_std_for_normalization(self, states: Tensor, returns: Tensor):
        tau = self.state_value_tau
        if tau == 0:
            return

        state_avg = states.mean(dim=0, keepdim=True)
        state_std = states.std(dim=0, keepdim=True)
        self.act.state_avg[:] = self.act.state_avg * (1 - tau) + state_avg * tau
        self.act.state_std[:] = self.cri.state_std * (1 - tau) + state_std * tau + 1e-4
        self.cri.state_avg[:] = self.act.state_avg
        self.cri.state_std[:] = self.act.state_std

        returns_avg = returns.mean(dim=0)
        returns_std = returns.std(dim=0)
        self.cri.value_avg[:] = self.cri.value_avg * (1 - tau) + returns_avg * tau
        self.cri.value_std[:] = self.cri.value_std * (1 - tau) + returns_std * tau + 1e-4

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        """soft update target network via current network

        target_net: update target network via current network to make training more stable.
        current_net: current network update via an optimizer
        tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd: str, if_save: bool):
        """save or load training files for Agent

        cwd: Current Working Directory. ElegantRL save training files in CWD.
        if_save: True: save files. False: load files.
        """
        assert self.save_attr_names.issuperset({'act', 'act_target', 'act_optimizer'})

        training_state_path = f"{cwd}/agent_training_state.pth"
        attr_names = sorted(self.save_attr_names)
        if if_save:
            os.makedirs(cwd, exist_ok=True)

            # Keep the legacy full-object files because evaluation scripts load
            # act.pth directly as a module. The state-dict bundle below is the
            # authoritative format for resuming training.
            for attr_name in attr_names:
                torch.save(getattr(self, attr_name), f"{cwd}/{attr_name}.pth")

            training_state = {"format_version": 2, "attributes": {}}
            for attr_name in attr_names:
                attr = getattr(self, attr_name)
                if isinstance(attr, torch.nn.Module):
                    record = {"kind": "module", "state_dict": attr.state_dict()}
                elif isinstance(attr, torch.optim.Optimizer):
                    record = {"kind": "optimizer", "state_dict": attr.state_dict()}
                elif isinstance(attr, Tensor):
                    record = {"kind": "tensor", "value": attr.detach()}
                else:
                    record = {"kind": "object", "value": attr}
                training_state["attributes"][attr_name] = record
            torch.save(training_state, training_state_path)
            return

        if os.path.isfile(training_state_path):
            training_state = torch.load(
                training_state_path,
                map_location=self.device,
                weights_only=False,
            )
            records = training_state.get("attributes", {})
            for attr_name in attr_names:
                record = records.get(attr_name)
                if record is not None:
                    self._load_training_attribute(attr_name, record["kind"], record)
            return

        # Backward compatibility for checkpoints written before format_version=2.
        # Load state into the objects created by __init__ instead of replacing
        # them; replacing a network and optimizer from separate pickle files
        # leaves the optimizer bound to different Parameter objects.
        for attr_name in attr_names:
            file_path = f"{cwd}/{attr_name}.pth"
            if not os.path.isfile(file_path):
                continue
            loaded = torch.load(file_path, map_location=self.device, weights_only=False)
            current = getattr(self, attr_name)
            if isinstance(current, torch.nn.Module) and isinstance(loaded, torch.nn.Module):
                current.load_state_dict(loaded.state_dict())
            elif isinstance(current, torch.optim.Optimizer) and isinstance(loaded, torch.optim.Optimizer):
                current.load_state_dict(loaded.state_dict())
            elif isinstance(current, Tensor) and isinstance(loaded, Tensor):
                with torch.no_grad():
                    current.copy_(loaded.to(device=current.device, dtype=current.dtype))
            elif isinstance(current, torch.nn.Module) and isinstance(loaded, dict):
                current.load_state_dict(loaded)
            elif isinstance(current, torch.optim.Optimizer) and isinstance(loaded, dict):
                current.load_state_dict(loaded)
            else:
                setattr(self, attr_name, loaded)

    def _load_training_attribute(self, attr_name: str, kind: str, record: dict):
        current = getattr(self, attr_name)
        if kind in {"module", "optimizer"}:
            current.load_state_dict(record["state_dict"])
        elif kind == "tensor":
            value = record["value"]
            with torch.no_grad():
                current.copy_(value.to(device=current.device, dtype=current.dtype))
        elif kind == "object":
            setattr(self, attr_name, record["value"])
        else:
            raise ValueError(f"Unsupported training-state kind {kind!r} for {attr_name!r}.")


def get_optim_param(optimizer: torch.optim) -> list:  # backup
    params_list = []
    for params_dict in optimizer.state_dict()["state"].values():
        params_list.extend([t for t in params_dict.values() if isinstance(t, torch.Tensor)])
    return params_list
