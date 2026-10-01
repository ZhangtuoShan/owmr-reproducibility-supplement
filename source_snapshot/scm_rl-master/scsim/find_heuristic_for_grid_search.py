from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

import scsim
from scsim.agents import EchelonStockHeuristicAgent
from scsim.grid_search import grid_search

path_to_dir  = Path(scsim.__file__).parent.parent / 'experiments' / 'scenario_213' / 'ray_results' / 'PPO'

folders = [p for p in path_to_dir.glob('*') if p.is_dir()]
path_to_config = '/home/illya/scm_rl/experiments/scenario_213/default_config.yaml'
logger.info('Found {} folders in {}'.format(folders, path_to_dir))
num_trials = 50
import re
pattern = r'std_demand_stores=(\d)'
for folder in folders:
    match = re.search(pattern, folder.name)
    std_demand_stores = match.group(1)
    logger.info('Setting std to {}'.format(std_demand_stores))

    with open(path_to_config, 'r') as f:
        config = yaml.full_load(f)
    config['std_demand_stores'] = int(std_demand_stores)
    logger.info(config)
    results_path = path_to_dir / 'std_demand_stores={}.csv'.format(std_demand_stores)
    if results_path.exists():
        df = pd.read_csv(results_path)
    else:
        df = grid_search(
            config,
            EchelonStockHeuristicAgent,
            ranges=[range(*r) for r in [[0, 400], [0, 20]]],
            num_trials=num_trials,
            num_workers=32,
            experiment_seed=7,
            use_k_plus_mu=False
        )
        df.to_csv(results_path, index=None)