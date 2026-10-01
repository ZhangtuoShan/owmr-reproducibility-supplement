from pathlib import Path

import pandas as pd
import yaml
import ray

from scsim.grid_search import grid_search
from loguru import logger

import scsim

PROJECT_DIR = Path(scsim.__file__).parent.parent


num_trials = 50
num_workers = 32
experiment_seed = 7
configs_dir = PROJECT_DIR / 'configs'
experiments_dir = PROJECT_DIR / 'experiments'
logger.info("Configs dir: {}".format(configs_dir))
scenario_folders = list(configs_dir.glob('scenario*'))
scenario_folders.sort(key=lambda x: int(x.name.split('_')[-1]))
logger.info('Found following folders: {}'.format(scenario_folders))
ray.init(local_mode=False)
for scenario_folder in scenario_folders:
    for allocation_type in ['proportional_allocation', 'connected_buckets_allocation']:
        scenario_id = int(scenario_folder.name.split('_')[-1])
        logger.info('Running scenario {} with allocation {}.'.format(scenario_id, allocation_type))
        configs_path = scenario_folder / 'default_config.yaml'
        with open(configs_path, 'r') as f:
            env_config = yaml.full_load(f)
        env_config['allocation_type'] = allocation_type
        with open(scenario_folder / 'search_spaces.yaml', 'r') as f:
            search_spaces_config = yaml.full_load(f)
            search_spaces = search_spaces_config['search_space']
            use_k_plus_mu = search_spaces_config.get('use_mu_plus_std_by_k', False)

        from scsim.agents import EchelonStockHeuristicAgent

        results_dir = experiments_dir / f'scenario_{scenario_id}'
        results_dir.mkdir(exist_ok=True, parents=True)
        # write env config
        with open(results_dir / 'default_config.yaml', 'w') as f:
            yaml.dump(env_config, f)

        results_path = results_dir / 'allocation={}, search_spaces={}, n_trials={}.csv'.format(allocation_type, search_spaces, num_trials)
        if results_path.exists():
           df = pd.read_csv(results_path)
        else:
            df = grid_search(
                env_config,
                EchelonStockHeuristicAgent,
                ranges=[range(*r) for r in search_spaces],
                num_trials=num_trials,
                num_workers=num_workers,
                experiment_seed=experiment_seed,
                use_k_plus_mu=use_k_plus_mu
            )
            df.to_csv(results_path, index=None)
        df = df.sort_values(by="score", ascending=False)
        best_score, best_score_std, agent_params = (
            df.loc[:, ["score", "score_mean_std", "agent_params"]].iloc[0].to_list()
        )
        logger.info('Scenario: {} |  Allocation function: {} | Best score: {} mean {}, std {}, params: {} | search_space: {}'.format(
            scenario_id, allocation_type, allocation_type, best_score, best_score_std, agent_params, search_spaces))
