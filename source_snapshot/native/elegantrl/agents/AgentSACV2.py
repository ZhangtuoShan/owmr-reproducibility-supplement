"""Compatibility entry point for standard SAC with explicit optimizer setup.

AgentSACV2 combines the corrected squashed-Gaussian log probability with the
standard online clipped-double-Q actor objective. Its actor, critic, and
temperature optimizers use Adam without weight decay, and the temperature
learning rate is independently configurable.
"""

from types import MethodType

import torch

from elegantrl.agents.AgentBase import get_optim_param
from elegantrl.agents.AgentSACLogProbFixed import AgentSACLogProbFixedOnlineMinQ
from elegantrl.train.config import Config


class AgentSACV2(AgentSACLogProbFixedOnlineMinQ):
    """SAC with corrected log-probability, online min-Q, and zero-decay Adam."""

    def __init__(
        self,
        net_dims: [int],
        state_dim: int,
        action_dim: int,
        gpu_id: int = 0,
        args: Config = Config(),
    ):
        super().__init__(
            net_dims=net_dims,
            state_dim=state_dim,
            action_dim=action_dim,
            gpu_id=gpu_id,
            args=args,
        )

        alpha_learning_rate = getattr(args, "alpha_learning_rate", None)
        self.alpha_learning_rate = float(
            self.learning_rate if alpha_learning_rate is None else alpha_learning_rate
        )
        self.optimizer_weight_decay = 0.0

        self.act_optimizer = torch.optim.Adam(
            self.act.parameters(),
            lr=self.learning_rate,
            weight_decay=self.optimizer_weight_decay,
        )
        self.cri_optimizer = torch.optim.Adam(
            self.cri.parameters(),
            lr=self.learning_rate,
            weight_decay=self.optimizer_weight_decay,
        )
        self.alpha_optimizer = torch.optim.Adam(
            (self.alpha_log,),
            lr=self.alpha_learning_rate,
            weight_decay=self.optimizer_weight_decay,
        )

        self.act_optimizer.parameters = MethodType(get_optim_param, self.act_optimizer)
        self.cri_optimizer.parameters = MethodType(get_optim_param, self.cri_optimizer)
        self.alpha_optimizer.parameters = MethodType(get_optim_param, self.alpha_optimizer)
