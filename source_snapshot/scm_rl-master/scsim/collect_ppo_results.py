import json
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import yaml
from loguru import logger

import scsim
from scsim.experiment import load_configs, load_rllib_agent
from scsim.grid_search import evaluate
from scsim.supply_chain_env import OWMREnvironment

PROJECT_ROOT = Path(scsim.__file__).parent.parent

experiment_dir = PROJECT_ROOT / "experiments"

scenario_paths = [
    folder for folder in experiment_dir.glob("*") if "scenario" in folder.name
]
logger.info(scenario_paths)

num_trials = 1000
explore = False
ray.init()


for scenario_path in scenario_paths:

    scenario_id = scenario_path.name.split("_")[-1]
    ppo_results = scenario_path / "ray_results" / "PPO"

    if int(scenario_id) in []:
        continue

    with open(experiment_dir / "results.json", "r") as f:
        results = json.load(f)
    agent = None
    for allocation_type in ['connected_buckets_allocation', 'proportional_allocation']:

        saved_results_csv = scenario_path / "PPO_allocation={}_num_trials={}_explore={}.csv".format(
            allocation_type, num_trials, explore
        )
        if saved_results_csv.exists():
            logger.info('Found results for {}'.format(saved_results_csv))
            result = pd.read_csv(saved_results_csv)
        else:

            ppo_experiments = [
                folder
                for folder in ppo_results.glob("**/*")
                if "OWMREnvironment" in folder.name and 'skip' not in folder.name
            ]
            if len(ppo_experiments) == 0:
                logger.info("No experiments found for {}".format(scenario_path.name))
                continue
            logger.info("Found {} experiments".format(ppo_experiments))
            latest_experiment = sorted(ppo_experiments)[-1]
            logger.info("Loading ppo: {}".format(latest_experiment))

            eval_config = {}
            eval_config["explore"] = False
            eval_config['allocation_type'] = allocation_type

            if agent is None:
                agent, _ = load_rllib_agent(
                    latest_experiment, eval_env_config={"explore": explore}, num_workers=1,
                )

            _, env_eval_config, env_train_config = load_configs(scenario_path)

            env = OWMREnvironment(env_eval_config)
            logger.info('Running PPO with {} allocation...'.format(allocation_type))
            result = evaluate(
                agent,
                env,
                num_trials=num_trials,
                experiment_seed=7,
                explore=explore,
            )

        mean_score = result["score"].mean()
        std_error = result["score"].std() / np.sqrt(len(result))
        if scenario_id not in results:
            results[scenario_id] = {}
        agent_name = f"PPO: {allocation_type}"
        results[scenario_id][agent_name] = {"Mean Score": mean_score, "Std Error": std_error}
        result.to_csv(saved_results_csv)
        logger.info(results[scenario_id][agent_name])
        with open(experiment_dir / "results.json", "w") as f:
            json.dump(results, f)
