"""Compatibility names for the standardized SAC backbone.

These names remain importable for existing experiment configurations and saved
actors. Their current updates now match AgentSAC. Historical A/B implementations
are preserved in the dated source backup, not selected implicitly by these names.
"""

from elegantrl.agents.AgentSAC import AgentSAC
from elegantrl.agents.net import ActorSAC


class ActorSACLogProbFixed(ActorSAC):
    """Compatibility subclass using the sampled, stable tanh-Gaussian density."""


class AgentSACLogProbFixed(AgentSAC):
    """Compatibility subclass using the standard SAC updates."""

    act_class = ActorSACLogProbFixed


class AgentSACLogProbFixedOnlineMinQ(AgentSACLogProbFixed):
    """Compatibility name; online clipped-double-Q is now the default."""
