import random
from collections import OrderedDict

import gym
import numpy as np
from gym import spaces


class OurOWMRRLLibWrapper(gym.Env):
    """Torch-free minimal Gym adapter for this project's Kaynov K5 dynamics.

    This is intentionally scoped to Kaynov scenario 13 / data_kaynov()[5].
    RLlib/Kaynov actions are [CDC, RDC1, ..., RDC10]. This project's paper-mode
    actions are [RDC1, ..., RDC10, CDC], so the wrapper reorders internally.
    """

    metadata = {"render.modes": []}

    def __init__(self, config=None):
        config = dict(config or {})

        self.number_of_stores = int(config.get("number_of_stores", 10))
        if self.number_of_stores != 10:
            raise ValueError("This minimal wrapper is currently scoped to Kaynov K5 with 10 RDCs.")

        self.delay_to_warehouse = int(config.get("delay_to_warehouse", 2))
        self.delay_to_stores = [int(x) for x in _expand(config.get("delay_to_stores", 2), self.number_of_stores)]
        self.mean_demand_stores = [float(x) for x in _expand(config.get("mean_demand_stores", 5), self.number_of_stores)]
        self.std_demand_stores = [float(x) for x in _expand(config.get("std_demand_stores", 14), self.number_of_stores)]
        self.demand_distribution = [str(x).lower() for x in _expand(config.get("demand_distribution", "normal"), self.number_of_stores)]

        self.warehouse_holding_cost = float(config.get("warehouse_holding_cost", 3))
        self.store_holding_cost = [float(x) for x in _expand(config.get("store_holding_cost", 3), self.number_of_stores)]
        self.shortage_cost = [float(x) for x in _expand(config.get("shortage_cost", 60), self.number_of_stores)]
        self.cost_emergency_shipment = float(config.get("cost_emergency_shipment", 0.0))

        self.episode_length = int(config.get("episode_length", 100))
        self.if_generate_demand = bool(config.get("if_generate_demand", True))
        self.force_zero_first_demand = bool(config.get("force_zero_first_demand", True))
        self.is_emergency_shipment = bool(config.get("is_emergency_shipment", True))
        self.prob_customer_waiting = [
            float(x) for x in _expand(config.get("prob_customer_waiting", config.get("rho", 0.8)), self.number_of_stores)
        ]

        self.production_capacity = float(config.get("production_capacity", 100))
        self.warehouse_capacity = float(config.get("warehouse_capacity", 1000))
        self.store_capacity = [float(x) for x in _expand(config.get("store_capacity", 100), self.number_of_stores)]
        self.cost_plus = float(config.get("cost_plus", 0.0))

        self.rdc_keys = list(range(1, self.number_of_stores + 1))
        self.action_space = spaces.MultiDiscrete([101] + [26] * self.number_of_stores)

        self.initial_rdc_inventory = OrderedDict()
        for k, lead_time, mean in zip(self.rdc_keys, self.delay_to_stores, self.mean_demand_stores):
            self.initial_rdc_inventory[k] = np.hstack(([int(mean * lead_time)], np.zeros(lead_time)))
        self.initial_cdc_inventory = np.hstack((
            [int(self.delay_to_warehouse * sum(self.mean_demand_stores))],
            np.zeros(self.delay_to_warehouse),
        ))

        sample_obs = np.asarray(self.reset(), dtype=np.float32)
        high = np.full(sample_obs.shape, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def reset(self):
        self.period = 0
        self.rdc_inventory = OrderedDict((k, inv.copy().astype(float)) for k, inv in self.initial_rdc_inventory.items())
        self.cdc_inventory = self.initial_cdc_inventory.copy().astype(float)
        self._generate_episode_demand()
        self._intransit_transition()
        return self._get_state()

    def step(self, action):
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        if action.size != self.number_of_stores + 1:
            raise ValueError(f"Expected {self.number_of_stores + 1} action branches, got {action.size}.")

        cdc_order = float(action[0])
        rdc_orders = [float(x) for x in action[1:]]
        self._analyze_paper_action(rdc_orders, cdc_order)

        demand_info = self._satisfy_demand_kaynov_partial_backlogging()

        cost = {"RHC": 0.0, "BC": 0.0, "LC": 0.0, "WHC": 0.0, "ESC": 0.0}
        cost["WHC"] = self.warehouse_holding_cost * max(float(self.cdc_inventory[0]), 0.0)
        for idx, k in enumerate(self.rdc_keys):
            cost["RHC"] += self.store_holding_cost[idx] * max(float(self.rdc_inventory[k][0]), 0.0)
            cost["LC"] += self.shortage_cost[idx] * demand_info[k]["lost"]
            cost["ESC"] += self.cost_emergency_shipment * demand_info[k]["emergency_shipment"]

        reward = -sum(cost.values()) + self.cost_plus

        self._intransit_transition()
        self.period += 1
        done = self.period == self.episode_length
        info = {"cost": cost, "demand_info": demand_info}
        return self._get_state(), float(reward), bool(done), info

    def _generate_episode_demand(self):
        self.demand = {}
        if not self.if_generate_demand:
            raise ValueError("This minimal wrapper currently supports fresh generated demand only.")
        for idx, k in enumerate(self.rdc_keys):
            dist_name = self.demand_distribution[idx]
            if dist_name == "normal":
                samples = np.random.normal(
                    self.mean_demand_stores[idx],
                    self.std_demand_stores[idx],
                    size=(self.episode_length + 1,),
                )
            elif dist_name == "poisson":
                samples = np.random.poisson(
                    self.mean_demand_stores[idx],
                    size=(self.episode_length + 1,),
                )
            elif dist_name == "uniform":
                samples = np.random.uniform(
                    self.mean_demand_stores[idx],
                    self.std_demand_stores[idx],
                    size=(self.episode_length + 1,),
                )
            else:
                raise ValueError(f"Unsupported demand distribution: {dist_name}")
            self.demand[k] = np.clip(np.floor(samples), 0, 500).astype(float)

    def _sample_period_demand(self, k):
        if self.force_zero_first_demand and self.period < 1:
            return 0.0
        return float(self.demand[k][self.period])

    def _get_state(self):
        state_rdc = np.hstack([self.rdc_inventory[k] for k in self.rdc_keys])
        state_cdc = self.cdc_inventory.copy()
        return np.hstack((state_rdc, state_cdc)).astype(np.float32)

    def _intransit_transition(self):
        for k in self.rdc_keys:
            inv = self.rdc_inventory[k]
            if len(inv) > 2:
                self.rdc_inventory[k] = np.hstack((inv[0] + inv[1], inv[2:], [0.0]))
            else:
                self.rdc_inventory[k] = np.hstack((inv[0] + inv[1], [0.0]))
        inv = self.cdc_inventory
        if len(inv) > 2:
            self.cdc_inventory = np.hstack((inv[0] + inv[1], inv[2:], [0.0]))
        else:
            self.cdc_inventory = np.hstack((inv[0] + inv[1], [0.0]))

    def _analyze_paper_action(self, rdc_orders, cdc_order):
        cdc_on_hand = round(float(self.cdc_inventory[0]), 0)
        _, allocated = allocation_random_sequential(cdc_on_hand, rdc_orders)
        self.cdc_inventory[0] = max(float(cdc_on_hand - round(sum(allocated), 0)), 0.0)
        for k, qty in zip(self.rdc_keys, allocated):
            self.rdc_inventory[k][-1] = float(qty)
        self.cdc_inventory[-1] = float(min(max(cdc_order, 0.0), self.production_capacity))

    def _satisfy_demand_kaynov_partial_backlogging(self):
        demand_info = {}
        emergency_available = max(float(self.cdc_inventory[0]), 0.0)
        for idx, k in enumerate(self.rdc_keys):
            demand = self._sample_period_demand(k)
            previous_inventory = max(float(self.rdc_inventory[k][0]), 0.0)
            regular_satisfied = min(previous_inventory, demand)
            shortage = max(demand - previous_inventory, 0.0)

            emergency_requested = False
            emergency_shipment = 0.0
            if shortage > 1e-12:
                emergency_requested = np.random.random() < self.prob_customer_waiting[idx]
                if emergency_requested and self.is_emergency_shipment and emergency_available > 1e-12:
                    emergency_shipment = min(shortage, emergency_available)
                    emergency_available -= emergency_shipment

            lost = max(shortage - emergency_shipment, 0.0)
            self.rdc_inventory[k][0] = max(previous_inventory - demand, 0.0)
            demand_info[k] = {
                "demand": demand,
                "regular_satisfied": regular_satisfied,
                "emergency_requested": emergency_requested,
                "emergency_shipment": emergency_shipment,
                "lost": lost,
            }

        self.cdc_inventory[0] = max(emergency_available, 0.0)
        return demand_info

    def render(self, mode="human"):
        return None

    def close(self):
        return None


def _expand(value, n):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == n:
            return list(value)
        if len(value) == 1:
            return list(value) * n
        raise ValueError(f"Expected scalar or length {n}, got {value!r}.")
    return [value for _ in range(n)]


def allocation_random_sequential(on_hand, to_allocate):
    new_allocate = [0.0 for _ in to_allocate]
    if sum(to_allocate) > on_hand:
        numbers = list(range(len(to_allocate)))
        random.shuffle(numbers)
        for rdc in numbers:
            pre_allocate = float(to_allocate[rdc])
            if pre_allocate > on_hand:
                new_allocate[rdc] = float(on_hand)
                on_hand = 0.0
                break
            new_allocate[rdc] = pre_allocate
            on_hand -= pre_allocate
        return on_hand, new_allocate

    new_allocate = [float(x) for x in to_allocate]
    on_hand -= round(sum(new_allocate), 0)
    return max(float(on_hand), 0.0), new_allocate
