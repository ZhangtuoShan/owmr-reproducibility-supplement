import itertools

import numpy as np
import pandas as pd
import ray
from loguru import logger
from tqdm import tqdm

from scsim.supply_chain_env import OWMREnvironment


def run_episode(
    agent,
    env,
    run_seed=None,
    warm_up_steps=0,
    full_fetch=False,
    episode_length=None,
    explore=None,
):
    # if run_seed is not None:
    np.random.seed(run_seed)
    prev_episode_length = env.episode_length = episode_length or env.episode_length
    state = env.reset()
    done = False
    score = 0
    steps = 0
    while not done:
        if full_fetch:
            action, _, infos = agent.compute_action(state, full_fetch=True)
            for k, v in infos.items():
                env.model.log(k, v)
        if explore is None:
            action = agent.compute_action(state)
        else:
            action = agent.compute_action(state, explore=explore)
        state, reward, done, meta = env.step(action)
        score += reward if warm_up_steps <= steps else 0
        steps += 1
        if episode_length:
            done = steps > episode_length
    env.episode_length = prev_episode_length
    return score, env.history


def evaluate(
    agent,
    env,
    num_trials=100,
    experiment_seed=None,
    run_seed=None,
    warmup_steps=0,
    **kwargs,
):
    dff = pd.DataFrame()

    if experiment_seed is not None:
        np.random.seed(experiment_seed)

    for i in tqdm(range(num_trials), disable=True):
        score, history = run_episode(
            agent, env, run_seed=run_seed, warm_up_steps=warmup_steps, **kwargs
        )
        costs = history.get_costs_summary(warmup_steps=warmup_steps)

        dff = pd.concat(
            [
                dff,
                pd.DataFrame(
                    {
                        "index": [i],
                        "score": score,
                        **costs,
                        # 'history': history.get_df()
                    }
                ),
            ],
            ignore_index=True,
        )
    return dff

def resolve_agent_params(means, stds, delays, k):
    # Actual_mean_demand * (1 + store_lead_time) + k * actual_std_demand * SQRT(1+store_leadtime)
    store_base_stock = [
        mean * (1 + delay) + k * std * np.sqrt(1 + delay)
        for mean, std, delay in zip(means, stds, delays)
    ]
    store_base_stock = [int(np.clip(np.round(base_stock), 0, 100000)) for base_stock in store_base_stock]
    return store_base_stock

def evaluate_heuristic(
    env_cls,
    env_config,
    agent_cls,
    agent_params,
    warm_up_steps=0,
    num_trials=100,
    experiment_seed=None,
    use_k_plus_mu=False
):
    env: OWMREnvironment = env_cls(env_config)
    number_of_stores = env.number_of_stores
    if use_k_plus_mu:
        real_demand_means = env_config['real_demand_means']
        real_demand_stds = env_config['real_demand_stds']
        delay_to_stores = env.delay_to_stores
        k = agent_params[-1] / 10
        store_params = resolve_agent_params(real_demand_means, real_demand_stds, delay_to_stores, k)
        agent_params = [agent_params[0]] + store_params
        logger.debug('Used mu plus {}: {}'.format(k, agent_params))
    elif (number_of_stores + 1) != len(agent_params):
        agent_params = (*agent_params, *[agent_params[-1]] * (number_of_stores - len(agent_params) + 1))
    # logger.info('Evaluating params: {}'.format(agent_params))
    agent = agent_cls(agent_params)
    if hasattr(agent, "interface"):
        env_config["interface"] = agent.interface
    df = evaluate(
        agent,
        env,
        num_trials=num_trials,
        experiment_seed=experiment_seed,
        warmup_steps=warm_up_steps,
    )
    stats = df.mean()
    stats = pd.concat(
        [
            stats,
            pd.Series(
                [agent_params, df["score"].std() / np.sqrt(len(df))],
                index=["agent_params", "score_mean_std"],
            ),

        ]
    )
    del df
    return stats


@ray.remote
def worker(
    env_cls,
    env_config,
    agent_cls,
    agent_params_chunk,
    num_trials=20,
    experiment_seed=None,
    tqdm_disable=True,
    use_k_plus_mu=False
):
    results = []
    for agent_params in tqdm(agent_params_chunk, disable=tqdm_disable):
        results.append(
            evaluate_heuristic(
                env_cls,
                env_config,
                agent_cls,
                agent_params,
                num_trials=num_trials,
                experiment_seed=experiment_seed,
                use_k_plus_mu=use_k_plus_mu
            )
        )
    return results


def get_chunks(iterable, chunks=1):
    # This is from http://stackoverflow.com/a/2136090/2073595
    lst = list(iterable)
    return [lst[i::chunks] for i in range(chunks)]


def grid_search(
    env_config, agent_cls, ranges, experiment_seed=None, num_trials=20, num_workers=4, use_k_plus_mu=False,
):

    params = list(itertools.product(*ranges))
    import time

    start = time.time()
    chunked_params = get_chunks(params, chunks=num_workers)

    suppress_progress = [False if x == 0 else True for x in range(num_workers)]
    results = [
        worker.remote(
            env_cls=OWMREnvironment,
            env_config=env_config,
            agent_cls=agent_cls,
            agent_params_chunk=p,
            num_trials=num_trials,
            experiment_seed=experiment_seed,
            tqdm_disable=s,
            use_k_plus_mu=use_k_plus_mu
        )
        for p, s in zip(chunked_params, suppress_progress)
    ]
    evals = ray.get(results)
    print(f"{len(params)} took {time.time() - start}")
    from itertools import chain

    df = pd.DataFrame(chain.from_iterable(evals))
    return df


#

if __name__ == "__main__":
    from scsim.agents import EchelonStockHeuristicAgent

    ray.init()
    result = grid_search(
        {},
        EchelonStockHeuristicAgent,
        ranges=[
            list(range(59, 80)),
            list(range(10, 11)),
            list(range(10, 11)),
            list(range(10, 11)),
        ],
    )
    print('Best score:', result[0])
    print('Best score std:', result[1])
    print('Best params:', result[2])
    print('All results:', result[3])
