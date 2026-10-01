import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

import scsim
from scsim.agents import EchelonStockHeuristicAgent
from scsim.experiment import load_rllib_agent, load_configs
from scsim.grid_search import grid_search, evaluate_heuristic, evaluate
from scsim.supply_chain_env import OWMREnvironment

path_to_dir  = Path(scsim.__file__).parent.parent / 'experiments' / 'scenario_213' / 'ray_results' / 'PPO'
results = {}
folders = [p for p in path_to_dir.glob('*') if p.is_dir()]
path_to_config = '/home/illya/scm_rl/configs/scenario_213'
default_config, env_eval_config, env_train_config = load_configs(path_to_config)
logger.info('Found {} folders in {}'.format(folders, path_to_dir))
num_trials = 1000
import re
pattern = r'std_demand_stores=(\d)'
for folder in folders:
    match = re.search(pattern, folder.name)
    std_demand_stores = int(match.group(1))
    logger.info('Setting std to {}'.format(std_demand_stores))

    default_config['std_demand_stores'] = int(std_demand_stores)
    env_eval_config['std_demand_stores'] = int(std_demand_stores)
    env_train_config['std_demand_stores'] = int(std_demand_stores)

    results_path = path_to_dir / 'std_demand_stores={}.csv'.format(std_demand_stores)
    results_df = pd.read_csv(results_path)
    results_df = results_df.sort_values(by='score', ascending=False)
    logger.info('Scenario: {}. Best score after 50 evaluations: {}'.format(std_demand_stores, results_df.iloc[0]))
    agent_params = results_df.iloc[0]['agent_params']
    agent_params = [int(p) for p in tuple(agent_params.strip('()[]').split(','))]

    evaluation_results_name = 'std_demand_stores={}, agent_params={}, num_trials={}.csv'.format(std_demand_stores, agent_params,
                                                                                         num_trials)
    evaluation_results_path = path_to_dir / evaluation_results_name
    if evaluation_results_path.exists():
        logger.info('Found results evaluation for {}'.format(evaluation_results_name))
        stats = pd.read_csv(evaluation_results_path, index_col=0, header=None).T
    else:
        logger.info('Running evaluation for {}'.format(evaluation_results_name))
        stats = evaluate_heuristic(
            OWMREnvironment,
            default_config,
            EchelonStockHeuristicAgent,
            agent_params=agent_params,
            num_trials=num_trials,
        )
        logger.info(stats)
        stats.T.to_csv(evaluation_results_path)
    logger.info(
        'std_demand_stores: {}, agent_params: {}, score: {}'.format(std_demand_stores, agent_params,
                                                                           stats['score']))

    results[std_demand_stores] = {'Echelon Stock': {
            'Mean Score': float(stats['score']),
            'Std Error': float(stats['score_mean_std'])
        }}
    explore = False
    saved_results_csv = path_to_dir / "PPO_std_demand_stores={}_num_trials={}_explore={}.csv".format(
        std_demand_stores, num_trials, explore
    )
    agent = None
    if saved_results_csv.exists():
        logger.info('Found results for {}'.format(saved_results_csv))
        result = pd.read_csv(saved_results_csv)
    else:
        latest_experiment = folder
        logger.info("Loading ppo: {}".format(latest_experiment))


        if agent is None:
            agent, _ = load_rllib_agent(
                latest_experiment, eval_env_config={"explore": explore}, num_workers=1,
            )

        env = OWMREnvironment(env_eval_config)
        logger.info('Running PPO with {} std...'.format(std_demand_stores))
        result = evaluate(
            agent,
            env,
            num_trials=num_trials,
            experiment_seed=7,
            explore=explore,
        )

    mean_score = result["score"].mean()
    std_error = result["score"].std() / np.sqrt(len(result))

    agent_name = f"PPO"
    results[std_demand_stores][agent_name] =  {"Mean Score": mean_score, "Std Error": std_error}
    result.to_csv(saved_results_csv)
    logger.info(results[std_demand_stores][agent_name])
    with open(path_to_dir / "results.json", "w") as f:
        json.dump(results, f)
