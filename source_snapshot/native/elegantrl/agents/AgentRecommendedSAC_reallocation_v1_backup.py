"""Backup of the early RecommendedSAC reallocation layer.

This class preserves the first practical reallocation design used before the
implicit-gradient/root-correction and distribution-aware inverse-CDF
refinements. It intentionally keeps the current lambda search interval from
``AgentRecommendedSAC``: roughly ``[-max(holding_cost), max(backorder_cost)]``
with a tiny numerical margin.

Use this only for ablation or reproduction. The production RecommendedSAC class
in ``AgentSAC.py`` remains the preferred implementation.
"""

import math

import torch
from torch import Tensor
from typing import Tuple

from elegantrl.agents.AgentSAC import AgentRecommendedSAC


class AgentRecommendedSACReallocationV1Backup(AgentRecommendedSAC):
    """RecommendedSAC with the pre-implicit-gradient reallocation layer.

    Differences from the current ``AgentRecommendedSAC``:
    - keeps the bisection-based lambda search;
    - does not apply implicit differentiation / root correction to lambda;
    - uses the original normal-approximation demand inverse CDF even when the
      parent class has built distribution-aware quantile tables;
    - still inherits the current narrowed lambda bounds from
      ``AgentRecommendedSAC._init_reallocation_tensors``.

    This is close to the first version of the differentiable reallocation
    training idea: gradients pass through the allocation formula for the final
    bisection lambda value, but the bisection decisions themselves are not made
    differentiable.
    """

    def _normal_demand_position_from_prob(self, prob: Tensor) -> Tensor:
        prob = prob.clamp(self.prob_eps, 1.0 - self.prob_eps)
        normal_quantile = torch.erfinv(prob * 2.0 - 1.0) * math.sqrt(2.0)
        return self.demand_mu + self.demand_sigma * normal_quantile

    def _allocation_given_lambda(self, lambda_value: Tensor, rdc_positions: Tensor, upper_alloc: Tensor) -> Tensor:
        raw_prob = (self.backorder_cost - lambda_value) / (self.backorder_cost + self.holding_cost)
        unconstrained_position = self._normal_demand_position_from_prob(raw_prob)
        lower = rdc_positions
        upper = rdc_positions + upper_alloc
        position = torch.max(torch.min(unconstrained_position, upper), lower)
        return position - rdc_positions

    def soft_lagrangian_reallocation(self, states: Tensor, raw_actions: Tensor) -> Tuple[Tensor, Tensor]:
        rdc_positions, cdc_on_hand, cdc_position = self._parse_inventory_positions(states)
        normalized_action = (raw_actions + 1.0) * 0.5

        target_rdc = self.rdc_min_bsl + normalized_action[:, :self.rdc_size] * (self.rdc_max_bsl - self.rdc_min_bsl)
        target_cdc = self.cdc_min_bsl + normalized_action[:, -1:] * (self.cdc_max_bsl - self.cdc_min_bsl)

        upper_alloc = self._smooth_positive(target_rdc - rdc_positions)
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

        lambda_star = (lambda_low + lambda_high) * 0.5
        lagrangian_alloc = self._allocation_given_lambda(lambda_star, rdc_positions, upper_alloc)
        feasible_rdc = torch.where(total_upper <= capacity + 1e-6, upper_alloc, lagrangian_alloc)
        feasible_cdc = self._smooth_positive(target_cdc - cdc_position)
        feasible_action = torch.cat((feasible_rdc, feasible_cdc), dim=1)
        feasible_action_norm = feasible_action / self.order_scale * 2.0 - 1.0
        return feasible_action_norm, feasible_action
