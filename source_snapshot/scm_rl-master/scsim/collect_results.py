from pathlib import Path

import yaml

import scsim
PROJECT_DIR = Path(scsim.__file__).parent.parent
from loguru import logger
import pandas as pd

from scsim.grid_search import evaluate_heuristic

from scsim.supply_chain_env import OWMREnvironment
from scsim.agents import EchelonStockHeuristicAgent
experiments_dir = PROJECT_DIR / 'experiments'
experiment_folders = experiments_dir.glob('*')


results = {}
num_trials = 1000

for experiment_folder in experiment_folders:
    scenario_id = experiment_folder.name.split('_')[-1]
    result_csvs = experiment_folder.glob('*.csv')
    results[scenario_id] = {}
    for results_csv in result_csvs:
        csv_name = results_csv.name
        logger.debug(csv_name)
        if 'agent_params' in csv_name or 'PPO' in csv_name:
            continue

        allocation_type = csv_name.split(',')[0].split('=')[-1]
        logger.info(allocation_type)
        results_df = pd.read_csv(results_csv)
        results_df = results_df.sort_values(by='score', ascending=False)
        # logger.info('Scenario: {}. Best score after 50 evaluations: {}'.format(scenario_id, results_df.iloc[0]))
        agent_params = results_df.iloc[0]['agent_params']
        agent_params = [int(p) for p in tuple(agent_params.strip('()[]').split(','))]
        with open(experiment_folder / 'default_config.yaml') as f:
            config = yaml.full_load(f)
        config['allocation_type'] = allocation_type

        evaluation_results_name = 'allocation={}, agent_params={}, num_trials={}.csv'.format(allocation_type, agent_params, num_trials)
        evaluation_results_path = experiment_folder / evaluation_results_name
        if evaluation_results_path.exists():
            logger.info('Found results evaluation for {}'.format(evaluation_results_name))
            stats = pd.read_csv(evaluation_results_path, index_col=0, header=None).T
        else:
            logger.info('Running evaluation for {}'.format(evaluation_results_name))
            stats = evaluate_heuristic(
                OWMREnvironment,
                config,
                EchelonStockHeuristicAgent,
                agent_params=agent_params,
                num_trials=num_trials,
            )
            stats.T.to_csv(evaluation_results_path)
        results[scenario_id]['ES: {}'.format(allocation_type)] = {
            'Mean Score': float(stats['score']),
            'Std Error': float(stats['score_mean_std'])
        }
        logger.info('Scenario: {}, allocation: {}, agent_params: {}, score: {}'.format(scenario_id, allocation_type, agent_params, stats['score']))
import json
with open(experiments_dir / 'results.json', 'w') as f:
    json.dump(results, f)

