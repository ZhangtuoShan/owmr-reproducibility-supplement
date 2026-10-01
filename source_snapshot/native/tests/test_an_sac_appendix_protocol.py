import inspect
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from elegantrl.agents.AgentSAC import AgentANSAC, AgentLagANSAC, AgentPenaltyANSAC


class CriticProbe:
    def __init__(self):
        self.actions = None

    def get_q_min(self, states, actions):
        self.actions = actions
        return torch.full((states.shape[0],), 10.0)

    def get_q1_q2(self, states, actions):
        self.actions = actions
        zeros = torch.zeros(states.shape[0])
        return zeros, zeros


class ActionNormalizationProtocolTest(unittest.TestCase):
    def setUp(self):
        self.states = torch.zeros((2, 3))
        self.next_states = torch.ones((2, 3))
        self.raw = torch.tensor([[-0.2, 0.4], [0.8, -0.3]])
        self.feasible = torch.tensor([[0.1, 0.1], [0.2, 0.2]])
        self.next_raw = torch.tensor([[0.3, -0.4], [-0.5, 0.6]])
        self.next_feasible = torch.tensor([[0.4, 0.4], [0.5, 0.5]])
        self.violations = torch.tensor([0.5, 1.0])
        self.sample = (
            self.states, self.raw, self.feasible,
            torch.tensor([1.0, 2.0]), self.violations,
            torch.tensor([1.0, 0.0]), self.next_states,
        )

    def make_agent(self, agent_class):
        agent = object.__new__(agent_class)
        agent.critic_uses_feasible_action = agent_class is AgentANSAC
        agent.reward_scale = 0.5
        agent.gamma = 0.9
        agent.alpha_log = torch.tensor(0.0)
        agent.act = SimpleNamespace(
            get_action_logprob=lambda states: (self.next_raw, torch.zeros(2))
        )
        agent.cri = CriticProbe()
        agent.cri_target = CriticProbe()
        agent.criterion = lambda q, target: ((q - target) ** 2).mean()
        return agent

    def test_penalty_critic_uses_original_actions_and_scaled_target_penalty(self):
        agent = self.make_agent(AgentPenaltyANSAC)
        agent.pa_penalty_coef = 2.0
        agent.pa_penalty_power = 1.0
        agent.pa_penalty_scale_by_reward = True
        loss, _, sampled_violations = agent.get_obj_critic_raw(
            SimpleNamespace(sample=lambda batch_size: self.sample), 2
        )

        self.assertIs(agent.cri.actions, self.raw)
        self.assertIs(agent.cri_target.actions, self.next_raw)
        self.assertIs(sampled_violations, self.violations)
        self.assertAlmostEqual(loss.item(), 91.25, places=5)

    def test_plain_an_critic_keeps_executed_action_path(self):
        agent = self.make_agent(AgentANSAC)
        agent.pa_penalty_coef = 0.0
        agent.pa_penalty_power = 1.0
        agent.pa_penalty_scale_by_reward = True
        agent.soft_proportional_allocation = (
            lambda states, actions: (self.next_feasible, None, None)
        )
        agent.get_obj_critic_raw(SimpleNamespace(sample=lambda batch_size: self.sample), 2)

        self.assertIs(agent.cri.actions, self.feasible)
        self.assertIs(agent.cri_target.actions, self.next_feasible)

    def test_lagrange_target_penalty_and_projected_update(self):
        agent = self.make_agent(AgentLagANSAC)
        agent.lag_lambda = torch.tensor(3.0)
        agent.lag_penalty_scale_by_reward = True
        agent.lag_delta = 0.25
        agent.lag_lambda_lr = 0.5
        agent.lag_lambda_max = 4.0
        self.assertTrue(torch.allclose(
            agent._violation_penalty(self.violations), torch.tensor([0.75, 1.5])
        ))
        agent.get_obj_critic_raw(SimpleNamespace(sample=lambda batch_size: self.sample), 2)
        self.assertIs(agent.cri.actions, self.raw)
        self.assertIs(agent.cri_target.actions, self.next_raw)
        agent._update_lagrange_multiplier(self.violations)
        self.assertAlmostEqual(agent.lag_lambda.item(), 3.25)

    def test_rollout_stores_unmasked_actor_output(self):
        agent = self.make_agent(AgentPenaltyANSAC)
        agent.num_envs = 1
        agent.state_dim = 3
        agent.action_dim = 2
        agent.device = torch.device("cpu")
        agent.last_state = torch.zeros((1, 3))
        actor_output = torch.tensor([[0.7, -0.2]])
        agent.act = SimpleNamespace(get_action=lambda state: actor_output)
        agent.apply_action_mask = lambda state, action: torch.full_like(action, -1.0)
        agent.soft_proportional_allocation = lambda state, action: (
            action, torch.zeros_like(action), torch.zeros((1, 1))
        )
        env = SimpleNamespace(step_feasible_action=lambda action: (
            np.zeros(3), 1.0, False, {}
        ))

        _, replay_raw, _, rewards, _, _ = agent.explore_one_env(env, 1)
        self.assertTrue(torch.equal(replay_raw[0], actor_output))
        self.assertAlmostEqual(rewards[0, 0].item(), 0.5)

    def test_actor_objectives_do_not_subtract_penalty_twice(self):
        for agent_class in (AgentPenaltyANSAC, AgentLagANSAC):
            source = inspect.getsource(agent_class.update_net)
            self.assertNotIn("penalty_pg", source)
            self.assertIn("obj_actor = (q_value_pg - log_prob * alpha).mean()", source)


if __name__ == "__main__":
    unittest.main()
