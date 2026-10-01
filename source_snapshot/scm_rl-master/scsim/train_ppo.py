from pathlib import Path

import ray
from loguru import logger
from ray import tune

from scsim.experiment import load_configs
from scsim.supply_chain_env import OWMREnvironment

from yaml import load
import yaml

def train(tune_config, env_train_config, env_eval_config, output_dir):
    declared_cpus = max(2, int(tune_config["config"].get("num_workers", 1)) + 1)
    ray.init(object_store_memory=2e9, num_cpus=declared_cpus)
    tune.run(
        tune_config['run'],
        num_samples=1,
        restore=tune_config.get("restore", None),
        resume=tune_config.get("resume", False),
        stop=tune_config.get('stop', None),
        checkpoint_freq=tune_config.get('checkpoint_freq', 20),
        checkpoint_at_end=tune_config.get('checkpoint_at_end', True),
        local_dir=output_dir,
        config={
            **tune_config["config"],
            "env": OWMREnvironment,
            "env_config": env_train_config,
            "evaluation_config": {"env_config": env_eval_config, "explore": False},
        },
        # restore=join(path_to_root,
        #              'ray_results/PPO/PPO_GymWrapper_0_2020-07-28_09-43-20zkhpvap3/checkpoint_1875/checkpoint-1875')
    )



import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "-s", "--scenario", type=str, help="root experiment folder",
)


if __name__ == "__main__":
    parser.add_argument(
        '--resume', type=str,
    )
    parser.add_argument(
        "--restore", type=str
    )
    parser.add_argument(
        "--timesteps-total",
        type=int,
        default=None,
        help="Override stop.timesteps_total for this run without editing the shared PPO config.",
    )
    parser.add_argument("--vf-clip-param", type=float, default=None)
    parser.add_argument("--vf-loss-coeff", type=float, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--sgd-minibatch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-sgd-iter", type=int, default=None)
    parser.add_argument("--clip-param", type=float, default=None)
    parser.add_argument("--kl-coeff", type=float, default=None)
    parser.add_argument("--kl-target", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=None,
        help="Override the Tune checkpoint interval without editing the shared PPO config.",
    )
    parser.add_argument(
        "--simple-optimizer",
        action="store_true",
        help="Use RLlib's single-device SGD path; avoids old TF multi-device dtype failures.",
    )
    args = parser.parse_args()

    scenario = args.scenario
    resume = args.resume
    restore = args.restore
    logger.info('Running Training for {}'.format(scenario))
    import scsim
    PROJECT_ROOT = Path(scsim.__file__).parent.parent
    CONFIGS_DIR = PROJECT_ROOT / 'configs'
    scenario_folder = CONFIGS_DIR / scenario
    scenario_id = scenario_folder.name.split('_')[-1]
    experiment_folder = PROJECT_ROOT / 'experiments'
    tune_config_path = CONFIGS_DIR / 'rllib_baselines' / 'ppo_config.yaml'

    tune_config = load(
        open(tune_config_path), Loader=yaml.FullLoader
    )
    if args.timesteps_total is not None:
        tune_config.setdefault("stop", {})["timesteps_total"] = int(args.timesteps_total)
    trainer_config = tune_config.setdefault("config", {})
    if args.vf_clip_param is not None:
        trainer_config["vf_clip_param"] = float(args.vf_clip_param)
    if args.vf_loss_coeff is not None:
        trainer_config["vf_loss_coeff"] = float(args.vf_loss_coeff)
    if args.train_batch_size is not None:
        if args.train_batch_size <= 0:
            raise ValueError("--train-batch-size must be positive")
        trainer_config["train_batch_size"] = int(args.train_batch_size)
    if args.sgd_minibatch_size is not None:
        if args.sgd_minibatch_size <= 0:
            raise ValueError("--sgd-minibatch-size must be positive")
        trainer_config["sgd_minibatch_size"] = int(args.sgd_minibatch_size)
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            raise ValueError("--learning-rate must be positive")
        trainer_config["lr"] = float(args.learning_rate)
    if args.num_sgd_iter is not None:
        if args.num_sgd_iter <= 0:
            raise ValueError("--num-sgd-iter must be positive")
        trainer_config["num_sgd_iter"] = int(args.num_sgd_iter)
    if args.clip_param is not None:
        if args.clip_param <= 0:
            raise ValueError("--clip-param must be positive")
        trainer_config["clip_param"] = float(args.clip_param)
    if args.kl_coeff is not None:
        if args.kl_coeff < 0:
            raise ValueError("--kl-coeff must be nonnegative")
        trainer_config["kl_coeff"] = float(args.kl_coeff)
    if args.kl_target is not None:
        if args.kl_target <= 0:
            raise ValueError("--kl-target must be positive")
        trainer_config["kl_target"] = float(args.kl_target)
    if args.num_workers is not None:
        trainer_config["num_workers"] = int(args.num_workers)
    if args.seed is not None:
        trainer_config["seed"] = int(args.seed)
    if args.checkpoint_freq is not None:
        if args.checkpoint_freq <= 0:
            raise ValueError("--checkpoint-freq must be positive")
        tune_config["checkpoint_freq"] = int(args.checkpoint_freq)
    if args.simple_optimizer:
        trainer_config["simple_optimizer"] = True

    default_env_config, env_eval_config, env_train_config = load_configs(scenario_folder)

    output_dir = experiment_folder / f'scenario_{scenario_id}'
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / 'default_config.yaml', 'w') as f:
        yaml.dump(default_env_config, f)
    with open(output_dir / 'evaluation_config.yaml', 'w') as f:
        yaml.dump(env_eval_config, f)
    with open(output_dir / 'training_config.yaml', 'w') as f:
        yaml.dump(env_train_config, f)
    tune_config['resume'] = args.resume
    tune_config['restore'] = args.restore
    train(tune_config, env_train_config, env_eval_config, output_dir=output_dir / 'ray_results')
