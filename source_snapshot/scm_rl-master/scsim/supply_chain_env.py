import numpy as np
from gym import spaces
from loguru import logger

from scsim.history import History

MAXIMUM_CAPACITY = 1e9


def proportional_allocation(requested, total_available, allocate_leftover=True, **kwargs):
    assert len(requested) > 0, "Requested value list cannot be empty"
    assert total_available >= 0, total_available
    requested = np.asarray(requested, dtype=np.int64)

    total_requested = sum(requested)
    if total_available == 0 or total_requested == 0:
        allocated = [0 for _ in range(len(requested))]
    elif total_requested < total_available:
        allocated = requested
    else:
        allocated = np.asarray(
            [int(total_available * x / total_requested) for x in requested]
            # [0 for x in requested]
        )
        leftover = int(total_available - sum(allocated))

        if allocate_leftover:
            for i in range(leftover):
                idx = np.argmax(requested - allocated)
                # idx = np.random.randint(len(requested))
                allocated[idx] += 1

    # print(requested, total_available, allocated)
    return allocated


def sequence_allocation(requested, total_available, levels=None, random_sequence=True, **kwargs):
    N = len(requested)
    assert N > 0, "Requested value list cannot be empty"
    assert total_available >= 0
    requested = np.asarray(requested, dtype=np.int64)
    total_requested = sum(requested)
    if total_available == 0 or total_requested == 0:
        allocated = [0 for _ in range(len(requested))]
    elif total_requested < total_available:
        allocated = requested
    else:
        allocated = [0 for _ in range(N)]
        if levels is None:
            permut_idxs = np.random.permutation(N) if random_sequence else np.arange(N)
        else:
            # permut_idxs = np.flip(np.argsort(levels))
            permut_idxs = np.argsort(levels)
        for i in permut_idxs:
            if total_available > 0:
                shortage = -min(0, requested[i] - total_available)
                a = min(total_available, requested[i])
                allocated[i] = a
                total_available = max(0, total_available - requested[i])
            else:
                allocated[i] = 0

    # print(requested, total_available, allocated)
    return allocated


def ballanced_allocation(requested, total_available, basestock_levels, demand_variance, **kwargs):
    assert len(requested) > 0, "Requested value list cannot be empty"
    assert total_available >= 0
    requested = np.asarray(requested, dtype=np.int64)

    total_requested = sum(requested)
    if total_available == 0 or total_requested == 0:
        allocated = [0 for _ in range(len(requested))]
    elif total_requested < total_available:
        allocated = requested
    else:
        shortage = total_available - total_requested

        allocated = []
        N = len(requested)
        p = [1 / (2 * N) + V / (2 * (sum(demand_variance))) for V in demand_variance]
        for i in range(N):
            allocated.append(requested[i] - p[i] * requested[i])

    # print(requested, total_available, allocated)
    return allocated


def connected_buckets_allocation(requested, total_available, **kwargs):
    """
    requested: List[retailer_1_order, ...]
    """
    assert len(requested) > 0, "Requested value list cannot be empty"
    assert total_available >= 0
    requested_copy = np.asarray(requested, dtype=np.int64).copy()

    total_requested = sum(requested_copy)
    if total_available == 0 or total_requested == 0:
        allocated = [0 for _ in range(len(requested_copy))]
    elif total_requested < total_available:
        allocated = requested_copy
    else:
        budget = total_available
        allocated = np.zeros_like(requested_copy)
        while budget > 0:
            argmax = np.argmax(requested_copy)
            allocated[argmax] += 1
            requested_copy[argmax] -= 1
            budget -= 1
    return allocated


allocation_fns = {
    "proportional_allocation": proportional_allocation,
    "sequential_allocation": sequence_allocation,
    'connected_buckets_allocation': connected_buckets_allocation
}


def expand_to_list_if_needed(value_or_list, n_elements=7):
    if isinstance(value_or_list, (int, float, str)):
        expanded_values = [value_or_list for __ in range(n_elements)]
    elif hasattr(value_or_list, "__iter__"):
        assert (
                len(value_or_list) == n_elements
        ), "Number of elements in the list should be {}, got {}: {}".format(
            n_elements, len(value_or_list), value_or_list
        )
        expanded_values = value_or_list
    else:
        raise ValueError(
            "The argument should be float, int or iterable, got {}: {}".format(
                type(value_or_list), value_or_list
            )
        )
    return expanded_values


def set_capacity(cap):
    assert cap >= 0, "Capacity should be a at least zero, got: {}".format(cap)
    if cap == 0:
        return MAXIMUM_CAPACITY
    return cap


from gym import Env


class OWMREnvironment(Env):

    def __init__(self, config):
        self.reward_scale = float(config.get("reward_scale", 1.0))
        if self.reward_scale <= 0:
            raise ValueError(
                "reward_scale must be positive, got: {}".format(self.reward_scale)
            )
        self.is_emergency_shipment = config.get("is_emergency_shipment", False)
        self.number_of_stores = config.get("number_of_stores", 3)
        self.delay_to_warehouse = config.get("delay_to_warehouse", 2)
        self.production_capacity = config.get("production_capacity", 0)
        self.production_capacity = set_capacity(self.production_capacity)
        self.warehouse_action_unit = config.get("warehouse_action_unit", 1)
        assert self.warehouse_action_unit > 0, (
            "warehouse_action_unit should be positive, got: {}".format(self.warehouse_action_unit)
        )
        self.warehouse_capacity = config.get("warehouse_capacity", 0)

        self.is_retailer_backlog = config.get("is_retailer_backlog", False)

        self.warehouse_capacity = set_capacity(self.warehouse_capacity)

        self.prob_customer_waiting = expand_to_list_if_needed(
            config.get("prob_customer_waiting", 0.8), n_elements=self.number_of_stores
        )

        self.cost_emergency_shipment = config.get("cost_emergency_shipment", 0)

        self.store_capacity = expand_to_list_if_needed(
            config.get("store_capacity", 0), n_elements=self.number_of_stores
        )
        self.store_capacity = [set_capacity(c) for c in self.store_capacity]
        self.delay_to_stores = expand_to_list_if_needed(
            config.get("delay_to_stores", 1), n_elements=self.number_of_stores
        )
        self.mean_demand_stores = expand_to_list_if_needed(
            config.get("mean_demand_stores", 3), n_elements=self.number_of_stores
        )
        self.std_demand_stores = expand_to_list_if_needed(
            config.get("std_demand_stores", 0), n_elements=self.number_of_stores
        )

        self.warehouse_holding_cost = config.get("warehouse_holding_cost", 1)
        self.store_holding_cost = expand_to_list_if_needed(
            config.get("store_holding_cost", 1), n_elements=self.number_of_stores
        )

        self.shortage_cost = expand_to_list_if_needed(config.get("shortage_cost", 9), n_elements=self.number_of_stores)
        self.demand_distribution = expand_to_list_if_needed(
            config.get("demand_distribution", "poisson"),
            n_elements=self.number_of_stores,
        )  # or poisson or normal

        self.episode_length = config.get("episode_length", 100)
        self.full_state = config.get("full_state", False)
        self.steps = 0

        self.max_allowed_action = np.asarray(
            expand_to_list_if_needed(
                config.get("max_allowed_action", 20),
                n_elements=self.number_of_stores + 1,
            ),
            dtype=np.int64,
        )
        if np.any(self.max_allowed_action <= 0):
            raise ValueError(
                "All max_allowed_action branch counts must be positive, got: {}".format(
                    self.max_allowed_action.tolist()
                )
            )
        self.action_space = spaces.MultiDiscrete(self.max_allowed_action)

        self.retailer_action_units = np.asarray(
            expand_to_list_if_needed(
                config.get("retailer_action_units", 1),
                n_elements=self.number_of_stores,
            ),
            dtype=np.int64,
        )
        if np.any(self.retailer_action_units <= 0):
            raise ValueError(
                "All retailer_action_units must be positive, got: {}".format(
                    self.retailer_action_units.tolist()
                )
            )
        default_retailer_max_orders = (
            (self.max_allowed_action[1:] - 1) * self.retailer_action_units
        )
        self.retailer_max_order_quantities = np.asarray(
            config.get(
                "retailer_max_order_quantities",
                default_retailer_max_orders.tolist(),
            ),
            dtype=np.int64,
        )
        if self.retailer_max_order_quantities.shape != (self.number_of_stores,):
            raise ValueError(
                "Expected {} retailer_max_order_quantities, got {}".format(
                    self.number_of_stores,
                    self.retailer_max_order_quantities.tolist(),
                )
            )
        if np.any(self.retailer_max_order_quantities < 0):
            raise ValueError(
                "retailer_max_order_quantities must be non-negative, got: {}".format(
                    self.retailer_max_order_quantities.tolist()
                )
            )

        self.warehouse_start_level = config.get(
            "warehouse_start_level",
            self.delay_to_warehouse * sum(self.mean_demand_stores),
        )
        self.stores_start_level = expand_to_list_if_needed(
            config.get(
                "stores_start_level",
                [
                    int(self.mean_demand_stores[i] * self.delay_to_stores[i])
                    for i in range(self.number_of_stores)
                ],
            ),
            n_elements=self.number_of_stores,
        )

        self.allocation_type = config.get("allocation_type", "proportional_allocation")
        try:
            self.allocation_fn = allocation_fns[self.allocation_type]
        except KeyError:
            raise KeyError(
                "Not supported allocation type. Expected one of {}, got: {}".format(
                    allocation_fns.keys, self.allocation_type
                )
            )

        sample_obs = self.reset()
        high = np.array([np.inf] * len(sample_obs))
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def reset(self):
        from collections import deque

        self.production_delay = deque(
            [0 for __ in range(self.delay_to_warehouse)], maxlen=self.delay_to_warehouse
        )
        self.warehouse_inventory_level = self.warehouse_start_level

        self.store_delays = [
            deque([0 for __ in range(delay)], maxlen=delay)
            for delay in self.delay_to_stores
        ]
        self.store_inventory_levels = self.stores_start_level.copy()

        self.demand = []
        for store_idx in range(self.number_of_stores):
            if self.demand_distribution[store_idx] == "normal":
                demand_data = np.random.normal(
                    self.mean_demand_stores[store_idx],
                    self.std_demand_stores[store_idx],
                    size=(self.episode_length + 1,),
                )
            elif self.demand_distribution[store_idx] == "poisson":
                demand_data = np.random.poisson(
                    self.mean_demand_stores[store_idx], size=(self.episode_length + 1,)
                )
            elif self.demand_distribution[store_idx] == "truncated_poisson":
                demand_data = np.random.poisson(
                    self.mean_demand_stores[store_idx], size=(self.episode_length + 1,)
                )
                demand_data = np.minimum(demand_data, self.std_demand_stores[store_idx])
            elif self.demand_distribution[store_idx] == 'uniform':
                demand_data = np.random.uniform(
                    self.mean_demand_stores[store_idx], self.std_demand_stores[store_idx], size=(self.episode_length + 1,)
                )
            else:
                raise ValueError(
                    "Unsupported demand distribution {}".format(
                        self.demand_distribution
                    )
                )
            demand_data = np.floor(demand_data)
            demand_data = np.clip(demand_data, 0, 500).astype(np.int64, copy=False)
            self.demand.append(demand_data)

        self.steps = 0
        from collections import defaultdict

        self.history = History()

        return self.get_obs(self.full_state)

    def log(self, var_name, var_value):
        self.history.log(var_name, var_value)
        # logger.debug('{}: {}'.format(var_name, var_value))

    def get_obs(self, full_state):
        if full_state:
            state = list(self.production_delay) + [self.warehouse_inventory_level]
            for store_idx in range(self.number_of_stores):
                state += list(self.store_delays[store_idx])
                state.append(self.store_inventory_levels[store_idx])
        else:
            warehouse_position = (
                    sum(self.production_delay) + self.warehouse_inventory_level
            )
            state = [warehouse_position]
            for store_idx in range(self.number_of_stores):
                store_position = self.store_inventory_levels[store_idx] + sum(
                    self.store_delays[store_idx]
                )
                state.append(store_position)
        return np.asarray(state, dtype=np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        if action.shape != self.action_space.shape:
            raise ValueError(
                "Expected action shape {}, got {}".format(
                    self.action_space.shape, action.shape
                )
            )
        if not self.action_space.contains(action):
            raise ValueError(
                "Action {} is outside MultiDiscrete bounds {}".format(
                    action.tolist(), self.max_allowed_action.tolist()
                )
            )

        def clip_action_by_capacity(a, position, capacity):
            assert (
                    position <= capacity
            ), "Position {} cannot be higher than capacity {}".format(
                position, capacity
            )
            return min(a, capacity - position)

        def position(level, pipeline):
            return level + sum(pipeline)

        warehouse_receive = self.production_delay[-1]
        self.production_delay[-1] = 0
        self.warehouse_inventory_level += warehouse_receive
        production_raw_action = action[0]
        production_action = min(self.production_capacity, production_raw_action * self.warehouse_action_unit)
        warehouse_pre_position = position(
            self.warehouse_inventory_level, self.production_delay
        )
        production_action = clip_action_by_capacity(
            production_action, warehouse_pre_position, self.warehouse_capacity
        )
        assert warehouse_pre_position + production_action <= self.warehouse_capacity
        self.production_delay.appendleft(production_action)

        self.log("warehouse.pre_level", self.warehouse_inventory_level)
        self.log("warehouse.pre_position", warehouse_pre_position)
        self.log("warehouse.raw_action", production_raw_action)
        self.log("warehouse.action_unit", self.warehouse_action_unit)
        self.log("warehouse.action", production_action)

        requested_store_raw_action = action[1:]
        requested_store_action = np.minimum(
            requested_store_raw_action * self.retailer_action_units,
            self.retailer_max_order_quantities,
        )
        allocated_actions = self.allocation_fn(
            requested=requested_store_action,
            total_available=max(0, self.warehouse_inventory_level),
        )
        total_requested = sum(requested_store_action)
        total_allocated = sum(allocated_actions)
        self.log("warehouse.total_allocated", total_allocated)
        self.log("warehouse.total_requested", total_requested)
        self.log("warehouse.total_shortage", total_requested - total_allocated)

        self.warehouse_inventory_level -= total_allocated

        total_stores_holding_cost = 0
        total_stores_shortage_cost = 0
        total_stores_emergency_cost = 0
        total_emergency = 0
        for store_idx in range(self.number_of_stores):
            # Receive products
            store_receive = self.store_delays[store_idx][-1]
            self.store_delays[store_idx][-1] = 0
            self.store_inventory_levels[store_idx] += store_receive
            allocated_store_action = allocated_actions[store_idx]
            store_pre_position = position(
                self.store_inventory_levels[store_idx], self.store_delays[store_idx]
            )
            self.log("retailer{}.pre_position".format(store_idx), store_pre_position)
            store_capacity = self.store_capacity[store_idx]
            allocated_store_action = clip_action_by_capacity(
                allocated_store_action, store_pre_position, store_capacity
            )
            self.log("retailer_{}.allocated_action".format(store_idx), allocated_store_action)
            self.log(
                "retailer_{}.requested_action".format(store_idx),
                requested_store_action[store_idx],
            )
            self.log(
                "retailer_{}.raw_action".format(store_idx),
                requested_store_raw_action[store_idx],
            )
            self.log(
                "retailer_{}.action_unit".format(store_idx),
                self.retailer_action_units[store_idx],
            )
            self.store_delays[store_idx].appendleft(allocated_store_action)

            # Satisfy demand from store
            store_demand = self.demand[store_idx][self.steps]
            store_on_hand = max(0, self.store_inventory_levels[store_idx])
            if self.is_retailer_backlog:
                satisfied_by_store = store_demand
            else:
                satisfied_by_store = min(store_on_hand, store_demand)
            self.store_inventory_levels[store_idx] -= satisfied_by_store

            # compute emergency_shipment
            shortage_on_store = max(0, store_demand - store_on_hand)
            warehouse_on_hand = max(0, self.warehouse_inventory_level)
            if np.random.random() < self.prob_customer_waiting[store_idx] and self.is_emergency_shipment:
                emergency_shipment = min(warehouse_on_hand, shortage_on_store)
            else:
                emergency_shipment = 0
            self.warehouse_inventory_level -= emergency_shipment
            total_shortage = shortage_on_store - emergency_shipment

            # calculate costs
            # Holding costs
            if self.is_retailer_backlog:
                store_backlog = -min(self.store_inventory_levels[store_idx], 0)
                store_shortage_cost = store_backlog * self.shortage_cost[store_idx]
            else:
                store_shortage_cost = total_shortage * self.shortage_cost[store_idx]

            total_stores_shortage_cost += store_shortage_cost
            # Shortage costs
            store_holding_cost = (
                    max(self.store_inventory_levels[store_idx], 0)
                    * self.store_holding_cost[store_idx]
            )
            total_stores_holding_cost += store_holding_cost
            # Emergency shipment costs
            cost_emergency_shipment = emergency_shipment * self.cost_emergency_shipment
            total_stores_emergency_cost += cost_emergency_shipment
            total_emergency += emergency_shipment
            store_position = position(
                self.store_inventory_levels[store_idx], self.store_delays[store_idx]
            )

            # log variables
            self.log(f"retailer_{store_idx}.demand", store_demand)
            self.log(f"retailer_{store_idx}.satisfied_by_store", satisfied_by_store)
            self.log(f"retailer_{store_idx}.shortage", shortage_on_store)
            self.log(f"retailer_{store_idx}.emergency_shipment", emergency_shipment)
            self.log(
                f"retailer_{store_idx}.cost_emergency_shipment", cost_emergency_shipment
            )
            self.log(f"retailer_{store_idx}.total_shortage", total_shortage)
            self.log(f"retailer_{store_idx}.shortage_cost", store_shortage_cost)
            self.log(f"retailer_{store_idx}.holding_cost", store_holding_cost)
            self.log(
                f"retailer_{store_idx}.level", self.store_inventory_levels[store_idx]
            )
            self.log(f"retailer_{store_idx}.position", store_position)

        self.log(f"warehouse.level", self.warehouse_inventory_level)
        warehouse_holding_cost = (
                self.warehouse_inventory_level * self.warehouse_holding_cost
        )
        self.log(f"warehouse.holding_cost", warehouse_holding_cost)
        self.log(f"warehouse.emergency_cost", total_stores_emergency_cost)
        self.log("warehouse.total_emergency", total_emergency)
        self.log("all_retailers.total_level", sum(self.store_inventory_levels))
        self.log('all_retailers.holding_cost', total_stores_holding_cost)
        self.log('all_retailers.shortage_cost', total_stores_shortage_cost)
        self.steps += 1
        raw_reward = -(
                warehouse_holding_cost
                + total_stores_emergency_cost
                + total_stores_shortage_cost
                + total_stores_holding_cost
        )
        reward = raw_reward * self.reward_scale
        self.log("reward.raw", raw_reward)
        self.log("reward.scale", self.reward_scale)
        self.log("reward.scaled", reward)
        done = self.steps >= self.episode_length

        for allocated, requested in zip(allocated_actions, requested_store_action):
            if allocated > requested:
                logger.debug('allocated: {}, requested: {}'.format(allocated_actions, requested_store_action))
                raise AssertionError()
            assert allocated >= 0
            assert requested >= 0

        return self.get_obs(self.full_state), reward, done, {}

    def render(self, mode="human"):
        pass


if __name__ == "__main__":
    import yaml

    from pathlib import Path

    config = yaml.full_load(
        open(Path('../configs/scenario_1/default_config.yaml'), 'r')
    )

    env = OWMREnvironment(config)
    from plotly import express as px
    from scsim.agents import EchelonStockHeuristicAgent
    from scsim.grid_search import run_episode

    run_episode(EchelonStockHeuristicAgent([59, 10, 10, 10]), env)

    fig = px.line(env.history.get_df())
    fig.show()
