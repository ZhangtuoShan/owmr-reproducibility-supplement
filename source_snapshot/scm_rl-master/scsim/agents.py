import numpy as np
from loguru import logger


class BaseStockAgent:
    def __init__(self, S=np.asarray([5])):
        self.v = np.round(np.asarray(S))

    def compute_action(self, current_level: np.ndarray):
        assert (
            current_level.shape == self.v.shape
        ), f"Shape should match {current_level.shape} {self.v.shape}"

        action = (self.v - current_level) * (current_level < self.v)
        assert np.all(action) >= 0
        return action

class EchelonStockHeuristicAgent:
    def __init__(self, base_stock_level):
        self.v = np.round(np.asarray(base_stock_level))

    def compute_action(self, obs: np.ndarray):
        assert len(obs) == len(self.v)
        corrected_level = obs.copy()
        corrected_level[0] += np.sum(corrected_level[1:])

        action = (self.v - corrected_level) * (corrected_level < self.v)
        assert np.all(action >= 0)
        # for a in action:
        #     if int(a) != a:
        #         logger.info('corrected={} v={} action={}'.format(corrected_level, self.v, action))
        return action