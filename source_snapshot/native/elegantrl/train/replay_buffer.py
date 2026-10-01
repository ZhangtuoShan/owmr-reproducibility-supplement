import os
import math
import torch
from typing import Tuple
from torch import Tensor

from elegantrl.train.config import Config
from elegantrl.train.device import resolve_torch_device


class ReplayBuffer:  # for off-policy
    def __init__(self,
                 max_size: int,
                 state_dim: int,
                 action_dim: int,
                 gpu_id: int = 0,
                 num_seqs: int = 1,
                 if_use_per: bool = False,
                 args: Config = Config()):
        self.p = 0  # pointer
        self.if_full = False
        self.cur_size = 0
        self.add_size = 0
        self.add_item = None
        self.max_size = max_size
        self.num_seqs = num_seqs
        self.device = resolve_torch_device(getattr(args, "device", None), gpu_id)

        self.replay_sampler = str(getattr(args, "replay_sampler", "uniform")).lower()
        valid_samplers = {"uniform", "paired_uniform", "inventory_stratified"}
        if self.replay_sampler not in valid_samplers:
            raise ValueError(
                f"Unknown replay_sampler={self.replay_sampler!r}; "
                f"choose one of {sorted(valid_samplers)}."
            )
        self.replay_focus_fraction = float(getattr(args, "replay_focus_fraction", 0.25))
        if not 0.0 < self.replay_focus_fraction < 1.0:
            raise ValueError("replay_focus_fraction must be in (0, 1).")
        self.replay_sampler_refresh_updates = int(
            getattr(args, "replay_sampler_refresh_updates", 10)
        )
        if self.replay_sampler_refresh_updates < 1:
            raise ValueError("replay_sampler_refresh_updates must be positive.")
        raw_lead_times = getattr(args, "replay_strata_rdc_lead_times", None)
        self.replay_strata_rdc_lead_times = (
            tuple(int(value) for value in raw_lead_times)
            if raw_lead_times is not None
            else None
        )
        raw_cdc_lead_time = getattr(args, "replay_strata_cdc_lead_time", None)
        self.replay_strata_cdc_lead_time = (
            int(raw_cdc_lead_time) if raw_cdc_lead_time is not None else None
        )
        if self.replay_sampler == "inventory_stratified":
            if not self.replay_strata_rdc_lead_times:
                raise ValueError(
                    "inventory_stratified replay requires replay_strata_rdc_lead_times."
                )
            if self.replay_strata_cdc_lead_time is None:
                raise ValueError(
                    "inventory_stratified replay requires replay_strata_cdc_lead_time."
                )

        replay_seed = int(getattr(args, "random_seed", 0)) + 70_000
        generator_device = self.device.type if self.device.type != "cuda" else self.device
        self._replay_uniform_generator = torch.Generator(device=generator_device)
        self._replay_uniform_generator.manual_seed(replay_seed + 1)
        self._replay_priority_generator = torch.Generator(device=generator_device)
        self._replay_priority_generator.manual_seed(replay_seed + 2)
        self._replay_sparse_generator = torch.Generator(device=generator_device)
        self._replay_sparse_generator.manual_seed(replay_seed + 3)
        self._sampler_update_count = 0
        self._sampler_cache_update = -1
        self._sampler_cache = None
        self.last_sampled_flat_ids = None

        """The struction of ReplayBuffer (for example, num_seqs = num_workers * num_envs == 2*4 = 8
        ReplayBuffer:
        worker0 for env0:   sequence of sub_env0.0  self.states  = Tensor[s, s, ..., s, ..., s]     
                                                    self.actions = Tensor[a, a, ..., a, ..., a]   
                                                    self.rewards = Tensor[r, r, ..., r, ..., r]   
                                                    self.undones = Tensor[d, d, ..., d, ..., d]
                                                                          <-----max_size----->
                                                                          <-cur_size->
                                                                                     ↑ pointer
                            sequence of sub_env0.1  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
                            sequence of sub_env0.2  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
                            sequence of sub_env0.3  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
        worker1 for env1:   sequence of sub_env1.0  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
                            sequence of sub_env1.1  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
                            sequence of sub_env1.2  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
                            sequence of sub_env1.3  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
        
        D: done=True
        d: done=False
        sequence of transition: s-a-r-d, s-a-r-d, s-a-r-D  s-a-r-d, s-a-r-d, s-a-r-d, s-a-r-d, s-a-r-D  s-a-r-d, ...
                                <------trajectory------->  <----------trajectory--------------------->  <-----------
        """
        self.states = torch.empty((max_size, num_seqs, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((max_size, num_seqs, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((max_size, num_seqs), dtype=torch.float32, device=self.device)
        self.undones = torch.empty((max_size, num_seqs), dtype=torch.float32, device=self.device)

        self.if_use_per = if_use_per
        if if_use_per:
            self.sum_trees = [SumTree(buf_len=max_size) for _ in range(num_seqs)]
            self.per_alpha = getattr(args, 'per_alpha', 0.6)  # alpha = (Uniform:0, Greedy:1)
            self.per_beta = getattr(args, 'per_beta', 0.4)  # alpha = (Uniform:0, Greedy:1)
            """PER.  Prioritized Experience Replay. Section 4
            alpha, beta = 0.7, 0.5 for rank-based variant
            alpha, beta = 0.6, 0.4 for proportional variant
            """
        else:
            self.sum_trees = None
            self.per_alpha = None
            self.per_beta = None

    def update(self, items: Tuple[Tensor, ...]):
        self.add_item = items
        states, actions, rewards, undones = items
        # assert states.shape[1:] == (env_num, state_dim)
        # assert actions.shape[1:] == (env_num, action_dim)
        # assert rewards.shape[1:] == (env_num,)
        # assert undones.shape[1:] == (env_num,)
        self.add_size = rewards.shape[0]

        p = self.p + self.add_size  # pointer
        if p > self.max_size:
            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size

            self.states[p0:p1], self.states[0:p] = states[:p2], states[-p:]
            self.actions[p0:p1], self.actions[0:p] = actions[:p2], actions[-p:]
            self.rewards[p0:p1], self.rewards[0:p] = rewards[:p2], rewards[-p:]
            self.undones[p0:p1], self.undones[0:p] = undones[:p2], undones[-p:]
        else:
            self.states[self.p:p] = states
            self.actions[self.p:p] = actions
            self.rewards[self.p:p] = rewards
            self.undones[self.p:p] = undones

        if self.if_use_per:
            '''data_ids for single env'''
            data_ids = torch.arange(self.p, p, dtype=torch.long, device=self.device)
            if p > self.max_size:
                data_ids = torch.fmod(data_ids, self.max_size)

            '''apply data_ids for vectorized env'''
            for sum_tree in self.sum_trees:
                sum_tree.update_ids(data_ids=data_ids.cpu(), prob=10.)

        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p
        self._mark_sampler_data_updated()

    def _mark_sampler_data_updated(self) -> None:
        self._sampler_update_count += 1

    def _rebuild_inventory_sampler_cache(self, sample_len: int) -> None:
        lead_times = self.replay_strata_rdc_lead_times
        cdc_lead_time = self.replay_strata_cdc_lead_time
        if not lead_times or cdc_lead_time is None:
            raise ValueError("Inventory replay strata require RDC and CDC lead times.")

        expected_state_dim = sum(value + 1 for value in lead_times) + cdc_lead_time + 1
        if self.states.shape[-1] != expected_state_dim:
            raise ValueError(
                "Inventory replay sampler state layout mismatch: "
                f"expected {expected_state_dim}, got {self.states.shape[-1]}."
            )

        # Flatten sequence-major to match flat_id = sequence * sample_len + time.
        states = self.states[:sample_len].permute(1, 0, 2).reshape(
            sample_len * self.num_seqs, -1
        )
        source_actions = getattr(self, "raw_actions", self.actions)
        actions = source_actions[:sample_len].permute(1, 0, 2).reshape(
            sample_len * self.num_seqs, -1
        )

        offset = 0
        on_hand = []
        rdc_positions = []
        for lead_time in lead_times:
            width = lead_time + 1
            inventory = states[:, offset: offset + width]
            on_hand.append(inventory[:, 0])
            rdc_positions.append(inventory.sum(dim=1))
            offset += width
        cdc_width = cdc_lead_time + 1
        cdc_local_position = states[:, offset: offset + cdc_width].sum(dim=1)
        on_hand_tensor = torch.stack(on_hand, dim=1)
        backlog = (-on_hand_tensor).clamp_min(0.0).sum(dim=1)
        shortage_nodes = (on_hand_tensor < 0.0).sum(dim=1)
        cdc_echelon_position = cdc_local_position + torch.stack(
            rdc_positions, dim=1
        ).sum(dim=1)

        positive_backlog = backlog[backlog > 1e-8]
        backlog_p90 = (
            float(torch.quantile(positive_backlog, 0.90))
            if positive_backlog.numel()
            else 0.0
        )
        priority_mask = (shortage_nodes >= 2) | (backlog > backlog_p90)
        all_ids = torch.arange(states.shape[0], device=self.device, dtype=torch.long)
        priority_ids = all_ids[priority_mask]

        quantile_levels = torch.tensor(
            [0.25, 0.50, 0.75], device=self.device, dtype=states.dtype
        )
        cdc_edges = torch.quantile(cdc_echelon_position, quantile_levels)
        cdc_ip_bin = torch.bucketize(cdc_echelon_position, cdc_edges, right=True)
        shortage_bin = shortage_nodes.clamp_max(2)
        cdc_action = actions[:, -1]
        cdc_action_bin = torch.ones_like(cdc_ip_bin)
        cdc_action_bin[cdc_action <= -0.95] = 0
        cdc_action_bin[cdc_action >= 0.95] = 2
        any_rdc_upper = (actions[:, :-1] >= 0.95).any(dim=1).long()
        cell_ids = (
            ((cdc_ip_bin * 3 + shortage_bin) * 3 + cdc_action_bin) * 2
            + any_rdc_upper
        ).long()
        unique_cells, inverse, counts = torch.unique(
            cell_ids, sorted=True, return_inverse=True, return_counts=True
        )
        sparse_weights = counts[inverse].float().reciprocal()

        self._sampler_cache = {
            "sample_len": sample_len,
            "priority_ids": priority_ids,
            "sparse_weights": sparse_weights,
            "priority_pool_rate": float(priority_ids.numel() / max(all_ids.numel(), 1)),
            "backlog_p90": backlog_p90,
            "occupied_cells": int(unique_cells.numel()),
        }
        self._sampler_cache_update = self._sampler_update_count

    def _inventory_sampler_cache_is_stale(self, sample_len: int) -> bool:
        if self._sampler_cache is None:
            return True
        if int(self._sampler_cache["sample_len"]) != sample_len:
            return True
        return (
            self._sampler_update_count - self._sampler_cache_update
            >= self.replay_sampler_refresh_updates
        )

    def _sample_flat_ids(self, batch_size: int, sample_len: int) -> Tensor:
        flat_size = sample_len * self.num_seqs
        if self.replay_sampler == "uniform":
            # Preserve historical global-RNG behavior exactly.
            flat_ids = torch.randint(
                flat_size, size=(batch_size,), requires_grad=False
            )
            self.last_sampled_flat_ids = flat_ids
            return flat_ids

        paired_uniform_ids = torch.randint(
            flat_size,
            size=(batch_size,),
            device=self.device,
            generator=self._replay_uniform_generator,
            requires_grad=False,
        )
        if self.replay_sampler == "paired_uniform":
            self.last_sampled_flat_ids = paired_uniform_ids
            return paired_uniform_ids

        if self._inventory_sampler_cache_is_stale(sample_len):
            self._rebuild_inventory_sampler_cache(sample_len)
        cache = self._sampler_cache
        focus_count = int(round(batch_size * self.replay_focus_fraction))
        priority_count = focus_count // 2
        sparse_count = focus_count - priority_count
        uniform_count = batch_size - focus_count

        priority_ids = cache["priority_ids"]
        if priority_ids.numel() == 0:
            priority_samples = paired_uniform_ids[uniform_count: uniform_count + priority_count]
        else:
            positions = torch.randint(
                priority_ids.numel(),
                size=(priority_count,),
                device=self.device,
                generator=self._replay_priority_generator,
            )
            priority_samples = priority_ids[positions]
        sparse_positions = torch.multinomial(
            cache["sparse_weights"],
            sparse_count,
            replacement=True,
            generator=self._replay_sparse_generator,
        )
        flat_ids = torch.cat(
            (paired_uniform_ids[:uniform_count], priority_samples, sparse_positions)
        )
        self.last_sampled_flat_ids = flat_ids
        return flat_ids

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        sample_len = self.cur_size - 1
        ids = self._sample_flat_ids(batch_size, sample_len)
        ids0 = torch.fmod(ids, sample_len)  # ids % sample_len
        ids1 = torch.div(ids, sample_len, rounding_mode='floor')  # ids // sample_len

        return (self.states[ids0, ids1],
                self.actions[ids0, ids1],
                self.rewards[ids0, ids1],
                self.undones[ids0, ids1],
                self.states[ids0 + 1, ids1],)  # next_state

    def _gather_n_step(
            self,
            chronological_ids: Tensor,
            sequence_ids: Tensor,
            n_step: int,
            gamma: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Gather episode-safe n-step transitions from the circular buffer.

        ``chronological_ids`` are offsets from the oldest stored transition.
        The cumulative continuation mask stops rewards and bootstrapping at an
        episode boundary.  Returning ``gamma**n * mask`` separately avoids
        applying an extra discount inside the agent.
        """
        n_step = int(n_step)
        if n_step < 1:
            raise ValueError(f"n_step must be positive, got {n_step}.")
        if self.cur_size <= n_step:
            raise ValueError(
                f"Replay buffer needs more than n_step={n_step} transitions; "
                f"cur_size={self.cur_size}."
            )

        chronological_ids = chronological_ids.to(self.device, dtype=torch.long)
        sequence_ids = sequence_ids.to(self.device, dtype=torch.long)
        oldest = self.p if self.if_full else 0
        first_ids = torch.remainder(oldest + chronological_ids, self.max_size)

        offsets = torch.arange(n_step, device=self.device, dtype=torch.long)
        transition_ids = torch.remainder(first_ids[:, None] + offsets[None, :], self.max_size)
        seq_matrix = sequence_ids[:, None].expand_as(transition_ids)
        rewards = self.rewards[transition_ids, seq_matrix]
        undones = self.undones[transition_ids, seq_matrix]

        alive_before = torch.ones_like(rewards)
        if n_step > 1:
            alive_before[:, 1:] = torch.cumprod(undones[:, :-1], dim=1)
        discounts = torch.pow(
            torch.as_tensor(float(gamma), dtype=rewards.dtype, device=self.device),
            torch.arange(n_step, dtype=rewards.dtype, device=self.device),
        )
        n_step_returns = (rewards * alive_before * discounts[None, :]).sum(dim=1)
        bootstrap_discount = undones.prod(dim=1) * float(gamma) ** n_step
        next_ids = torch.remainder(first_ids + n_step, self.max_size)

        return (
            self.states[first_ids, sequence_ids],
            self.actions[first_ids, sequence_ids],
            n_step_returns,
            bootstrap_discount,
            self.states[next_ids, sequence_ids],
        )

    def _gather_sequences(
            self,
            chronological_ids: Tensor,
            sequence_ids: Tensor,
            horizon: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Gather chronological, episode-marked replay sequences.

        The saved transition at offset ``k`` contains ``(s_k, a_k, r_k,
        undone_k)``.  The returned next-state tensor contains
        ``(s_1, ..., s_horizon)``.  Consumers must apply ``undones`` when
        constructing returns so that reset states are never bootstrapped into
        the preceding episode.
        """
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError(f"horizon must be positive, got {horizon}.")
        if self.cur_size <= horizon:
            raise ValueError(
                f"Replay buffer needs more than horizon={horizon} transitions; "
                f"cur_size={self.cur_size}."
            )

        chronological_ids = chronological_ids.to(self.device, dtype=torch.long)
        sequence_ids = sequence_ids.to(self.device, dtype=torch.long)
        oldest = self.p if self.if_full else 0
        first_ids = torch.remainder(oldest + chronological_ids, self.max_size)

        offsets = torch.arange(horizon, device=self.device, dtype=torch.long)
        transition_ids = torch.remainder(first_ids[:, None] + offsets[None, :], self.max_size)
        next_ids = torch.remainder(transition_ids + 1, self.max_size)
        seq_matrix = sequence_ids[:, None].expand_as(transition_ids)

        return (
            self.states[first_ids, sequence_ids],
            self.actions[first_ids, sequence_ids],
            self.rewards[transition_ids, seq_matrix],
            self.undones[transition_ids, seq_matrix],
            self.states[next_ids, seq_matrix],
        )

    def sample_sequences(
            self,
            batch_size: int,
            horizon: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample fixed-length chronological sequences for multi-step targets."""
        horizon = int(horizon)
        valid_per_sequence = self.cur_size - horizon
        if valid_per_sequence <= 0:
            raise ValueError(
                f"Replay buffer has no valid sequence starts: cur_size={self.cur_size}, "
                f"horizon={horizon}."
            )
        flat_ids = torch.randint(
            valid_per_sequence * self.num_seqs,
            size=(batch_size,),
            device=self.device,
            requires_grad=False,
        )
        chronological_ids = torch.remainder(flat_ids, valid_per_sequence)
        sequence_ids = torch.div(flat_ids, valid_per_sequence, rounding_mode='floor')
        return self._gather_sequences(
            chronological_ids=chronological_ids,
            sequence_ids=sequence_ids,
            horizon=horizon,
        )

    def sample_n_step(
            self,
            batch_size: int,
            n_step: int,
            gamma: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample n-step targets without crossing the newest buffer entry."""
        n_step = int(n_step)
        valid_per_sequence = self.cur_size - n_step
        if valid_per_sequence <= 0:
            raise ValueError(
                f"Replay buffer has no valid n-step starts: cur_size={self.cur_size}, "
                f"n_step={n_step}."
            )
        flat_ids = torch.randint(
            valid_per_sequence * self.num_seqs,
            size=(batch_size,),
            device=self.device,
            requires_grad=False,
        )
        chronological_ids = torch.remainder(flat_ids, valid_per_sequence)
        sequence_ids = torch.div(flat_ids, valid_per_sequence, rounding_mode='floor')
        return self._gather_n_step(
            chronological_ids=chronological_ids,
            sequence_ids=sequence_ids,
            n_step=n_step,
            gamma=gamma,
        )

    def sample_for_per(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        beg = -self.max_size
        end = (self.cur_size - self.max_size) if (self.cur_size < self.max_size) else -1

        '''get is_indices, is_weights'''
        is_indices: list = []
        is_weights: list = []

        assert batch_size % self.num_seqs == 0
        sub_batch_size = batch_size // self.num_seqs
        for env_i in range(self.num_seqs):
            sum_tree = self.sum_trees[env_i]
            _is_indices, _is_weights = sum_tree.important_sampling(batch_size, beg, end, self.per_beta)
            is_indices.append(_is_indices + sub_batch_size * env_i)
            is_weights.append(_is_weights)

        is_indices: Tensor = torch.hstack(is_indices).to(self.device)
        is_weights: Tensor = torch.hstack(is_weights).to(self.device)

        ids0 = torch.fmod(is_indices, self.cur_size)  # is_indices % sample_len
        ids1 = torch.div(is_indices, self.cur_size, rounding_mode='floor')  # is_indices // sample_len
        return (
            self.states[ids0, ids1],
            self.actions[ids0, ids1],
            self.rewards[ids0, ids1],
            self.undones[ids0, ids1],
            self.states[ids0 + 1, ids1],  # next_state
            is_weights,  # important sampling weights
            is_indices,  # important sampling indices
        )

    def td_error_update_for_per(self, is_indices: Tensor, td_error: Tensor):  # td_error = (q-q).detach_().abs()
        prob = td_error.clamp(1e-8, 10).pow(self.per_alpha)

        # self.sum_tree.update_ids(is_indices.cpu(), prob.cpu())
        batch_size = td_error.shape[0]
        sub_batch_size = batch_size // self.num_seqs
        for env_i in range(self.num_seqs):
            sum_tree = self.sum_trees[env_i]
            slice_i = env_i * sub_batch_size
            slice_j = slice_i + sub_batch_size

            sum_tree.update_ids(is_indices[slice_i:slice_j].cpu(), prob[slice_i:slice_j].cpu())

    def save_or_load_history(self, cwd: str, if_save: bool):
        item_names = self.get_history_items()

        if if_save:
            for item, name in item_names:
                if self.cur_size == self.p:
                    buf_item = item[:self.cur_size]
                else:
                    buf_item = torch.vstack((item[self.p:self.cur_size], item[0:self.p]))
                file_path = f"{cwd}/replay_buffer_{name}.pth"
                print(f"| buffer.save_or_load_history(): Save {file_path}")
                torch.save(buf_item, file_path)

        elif all([os.path.isfile(f"{cwd}/replay_buffer_{name}.pth") for item, name in item_names]):
            max_sizes = []
            for item, name in item_names:
                file_path = f"{cwd}/replay_buffer_{name}.pth"
                print(f"| buffer.save_or_load_history(): Load {file_path}")
                buf_item = torch.load(file_path, map_location=self.device, weights_only=False)

                max_size = buf_item.shape[0]
                if max_size > self.max_size:
                    raise ValueError(
                        f"Saved replay buffer has {max_size} items, exceeding configured max_size={self.max_size}."
                    )
                item[:max_size] = buf_item
                max_sizes.append(max_size)
            assert all([max_size == max_sizes[0] for max_size in max_sizes])
            self.cur_size = self.p = max_sizes[0]
            self.if_full = self.cur_size == self.max_size
            self._mark_sampler_data_updated()

    def get_history_items(self):
        return (
            (self.states, "states"),
            (self.actions, "actions"),
            (self.rewards, "rewards"),
            (self.undones, "undones"),
        )


class ReplayBufferRecommendedSAC(ReplayBuffer):
    """Replay buffer for RA-SAC recommended training.

    It stores the actor's raw virtual action for diagnostics and the
    reallocated feasible action used as the critic input.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_actions = torch.empty_like(self.actions)

    def update(self, items: Tuple[Tensor, ...]):
        self.add_item = items
        states, raw_actions, feasible_actions, rewards, undones = items
        self.add_size = rewards.shape[0]

        p = self.p + self.add_size
        if p > self.max_size:
            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size

            self.states[p0:p1], self.states[0:p] = states[:p2], states[-p:]
            self.raw_actions[p0:p1], self.raw_actions[0:p] = raw_actions[:p2], raw_actions[-p:]
            self.actions[p0:p1], self.actions[0:p] = feasible_actions[:p2], feasible_actions[-p:]
            self.rewards[p0:p1], self.rewards[0:p] = rewards[:p2], rewards[-p:]
            self.undones[p0:p1], self.undones[0:p] = undones[:p2], undones[-p:]
        else:
            self.states[self.p:p] = states
            self.raw_actions[self.p:p] = raw_actions
            self.actions[self.p:p] = feasible_actions
            self.rewards[self.p:p] = rewards
            self.undones[self.p:p] = undones

        if self.if_use_per:
            raise NotImplementedError("PER is not implemented for ReplayBufferRecommendedSAC.")

        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p
        self._mark_sampler_data_updated()

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        sample_len = self.cur_size - 1
        ids = self._sample_flat_ids(batch_size, sample_len)
        ids0 = torch.fmod(ids, sample_len)
        ids1 = torch.div(ids, sample_len, rounding_mode='floor')

        return (self.states[ids0, ids1],
                self.raw_actions[ids0, ids1],
                self.actions[ids0, ids1],
                self.rewards[ids0, ids1],
                self.undones[ids0, ids1],
                self.states[ids0 + 1, ids1],)

    def get_history_items(self):
        return super().get_history_items() + ((self.raw_actions, "raw_actions"),)


class ReplayBufferANSAC(ReplayBufferRecommendedSAC):
    """Replay buffer for action-normalized SAC variants.

    It stores raw relaxed actions, executed feasible actions, original rewards,
    and relaxed-action violations. Penalty/Lagrangian terms can then be computed
    online during critic updates without changing the environment reward.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.violations = torch.empty_like(self.rewards)

    def update(self, items: Tuple[Tensor, ...]):
        self.add_item = items
        states, raw_actions, feasible_actions, rewards, violations, undones = items
        self.add_size = rewards.shape[0]
        if violations.ndim == rewards.ndim + 1:
            violations = violations.squeeze(-1)

        p = self.p + self.add_size
        if p > self.max_size:
            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size

            self.states[p0:p1], self.states[0:p] = states[:p2], states[-p:]
            self.raw_actions[p0:p1], self.raw_actions[0:p] = raw_actions[:p2], raw_actions[-p:]
            self.actions[p0:p1], self.actions[0:p] = feasible_actions[:p2], feasible_actions[-p:]
            self.rewards[p0:p1], self.rewards[0:p] = rewards[:p2], rewards[-p:]
            self.violations[p0:p1], self.violations[0:p] = violations[:p2], violations[-p:]
            self.undones[p0:p1], self.undones[0:p] = undones[:p2], undones[-p:]
        else:
            self.states[self.p:p] = states
            self.raw_actions[self.p:p] = raw_actions
            self.actions[self.p:p] = feasible_actions
            self.rewards[self.p:p] = rewards
            self.violations[self.p:p] = violations
            self.undones[self.p:p] = undones

        if self.if_use_per:
            raise NotImplementedError("PER is not implemented for ReplayBufferANSAC.")

        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        sample_len = self.cur_size - 1

        ids = torch.randint(sample_len * self.num_seqs, size=(batch_size,), requires_grad=False)
        ids0 = torch.fmod(ids, sample_len)
        ids1 = torch.div(ids, sample_len, rounding_mode='floor')

        return (self.states[ids0, ids1],
                self.raw_actions[ids0, ids1],
                self.actions[ids0, ids1],
                self.rewards[ids0, ids1],
                self.violations[ids0, ids1],
                self.undones[ids0, ids1],
                self.states[ids0 + 1, ids1],)

    def get_history_items(self):
        return super().get_history_items() + ((self.violations, "violations"),)


ReplayBufferPASAC = ReplayBufferANSAC


class SumTree:
    """ BinarySearchTree for PER (SumTree)
    Contributor: Github GyChou, Github mississippiu
    Reference: https://github.com/kaixindelele/DRLib/tree/main/algos/pytorch/td3_sp
    Reference: https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
    """

    def __init__(self, buf_len: int):
        self.buf_len = buf_len  # replay buffer len
        self.max_len = (buf_len - 1) + buf_len  # parent_nodes_num + leaf_nodes_num
        self.depth = math.ceil(math.log2(self.max_len))

        self.tree = torch.zeros(self.max_len, dtype=torch.float32)

    def update_id(self, data_id: int, prob=10):  # 10 is max_prob
        tree_id = data_id + self.buf_len - 1

        delta = prob - self.tree[tree_id]
        self.tree[tree_id] = prob

        for depth in range(self.depth - 2):  # propagate the change through tree
            tree_id = (tree_id - 1) // 2  # faster than the recursive loop
            self.tree[tree_id] += delta

    def update_ids(self, data_ids: Tensor, prob: Tensor = 10.):  # 10 is max_prob
        l_ids = data_ids + self.buf_len - 1

        self.tree[l_ids] = prob
        for depth in range(self.depth - 2):  # propagate the change through tree
            p_ids = ((l_ids - 1) // 2).unique()  # parent indices
            l_ids = p_ids * 2 + 1  # left children indices
            r_ids = l_ids + 1  # right children indices
            self.tree[p_ids] = self.tree[l_ids] + self.tree[r_ids]

            l_ids = p_ids

    def get_leaf_id_and_value(self, v) -> Tuple[int, float]:
        """Tree structure and array storage:
        Tree index:
              0       -> storing priority sum
            |  |
          1     2
         | |   | |
        3  4  5  6    -> storing priority for transitions
        Array type for storing: [0, 1, 2, 3, 4, 5, 6]
        """
        p_id = 0  # the leaf's parent node

        for depth in range(self.depth - 2):  # propagate the change through tree
            l_id = min(2 * p_id + 1, self.max_len - 1)  # the leaf's left node
            r_id = l_id + 1  # the leaf's right node
            if v <= self.tree[l_id]:
                p_id = l_id
            else:
                v -= self.tree[l_id]
                p_id = r_id
        return p_id, self.tree[p_id]  # leaf_id and leaf_value

    def important_sampling(self, batch_size: int, beg: int, end: int, per_beta: float) -> Tuple[Tensor, Tensor]:
        # get random values for searching indices with proportional prioritization
        values = (torch.arange(batch_size) + torch.rand(batch_size)) * (self.tree[0] / batch_size)

        # get proportional prioritization
        leaf_ids, leaf_values = list(zip(*[self.get_leaf_id_and_value(v) for v in values]))
        leaf_ids = torch.tensor(leaf_ids, dtype=torch.long)
        leaf_values = torch.tensor(leaf_values, dtype=torch.float32)

        indices = leaf_ids - (self.buf_len - 1)
        assert 0 <= indices.min()
        assert indices.max() < self.buf_len

        prob_ary = leaf_values / self.tree[beg:end].min()
        weights = torch.pow(prob_ary, -per_beta)
        return indices, weights
