import pickle

from Benchmark import (
    FiniteHorizonNonstationaryOWMREnv,
    OWMR_input,
    OWMRInventoryEnv,
    auto_hyperparameter,
    env_input_initialize,
)
from dataset import *
from elegantrl.agents import (AgentA2C, AgentConvexSAC, AgentDDPG, AgentDQN,
                              AgentPPO, AgentRecommendedSAC, AgentSAC, AgentTD3)
from elegantrl.train.config import Config, configure_training_evaluation
from elegantrl.train.run import *
from output_paths import output_path
from utils import logger, my_mkdir

DRL_params = {
    'PPO': {
        'lr': 1e-3,
        'gamma': 0.9,
        'rs': 2**0,
        'net': (64,32)
    },
    'A2C': {
        'lr': 1e-4,
        'gamma': 0.96,
        'rs': 2**-6,
        'net': (64,32)
    },

    'DDPG': {
        'lr': 1e-3,
        'gamma': 0.96,
        'rs': 2**-6,
        'net': (64,32)
    },
    'TD3': {
        'lr': 1e-3,  ## 1e-4
        'gamma': 0.9,  ## 0.96
        'rs': 2**-6,
        'net': (128,64)
    },
    # 'SAC': {
    #     'lr': 1e-4,
    #     'gamma': 0.9,
    #     'rs': 2e-6,
    #     'net': (128,64)
    # },
    'SAC': {
        'lr': 1e-4,
        'gamma': 0.96,
        'rs': 2**-6,
        'net': (64,32)
    },
    'RecommendedSAC': {
        'lr': 1e-4,
        'gamma': 0.96,
        'rs': 2**-6,
        'net': (64,32)
    },
    'DiscreteMultiPPO': {
        'lr': 1e-4,
        'net': (64, 64),
        'batch size': 32768,
        'vf loss': 1e-7,
        'entropy': 0,
        'clipping norm': 4,
        'gamma': 0.95,
        'GAE': 0.98,
        'rs': 1
    }
}


NONSTATIONARY_GAMMA_CANDIDATES = (0.99, 0.995, 1.0)


def get_nonstationary_training_defaults(horizon=13):
    """Return finite-horizon off-policy defaults in transition-count units."""
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError(f'horizon must be positive, got {horizon}.')
    return {
        # Forty complete episodes per rollout/update and 800 for warm-up.
        'horizon_len': horizon * 40,
        'replay_warmup_steps': horizon * 800,
        'freeze_state_normalization_from_warmup': True,
        'buffer_size': int(1e5),
        'batch_size': 128,
        'repeat_times': 1.0,
        'break_step': int(5e5),
        'eval_per_step': int(1e4),
        'eval_times': 20,
        # The finite-horizon objective is the undiscounted 13-period cost.
        'gamma': 1.0,
    }


def nonstationary_reward_scale_from_reference_cost(reference_weekly_cost, multiplier=1.0):
    """Scale a reference weekly cost to an order-one per-period reward."""
    reference_weekly_cost = float(reference_weekly_cost)
    multiplier = float(multiplier)
    if not np.isfinite(reference_weekly_cost) or reference_weekly_cost <= 0:
        raise ValueError(
            f'reference_weekly_cost must be finite and positive, got {reference_weekly_cost}.'
        )
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError(f'multiplier must be finite and positive, got {multiplier}.')
    return multiplier / reference_weekly_cost


def _is_off_policy_agent(agent_class):
    name = agent_class.__name__ if agent_class is not None else ''
    on_policy_names = ('SARSA', 'VPG', 'A2C', 'A3C', 'TRPO', 'PPO', 'MPO')
    return all(token not in name for token in on_policy_names)


def get_env_args(info_input, env_class=OWMRInventoryEnv, max_step=None):

    env_args = {'env_name': 'OWMR',
                'max_step': 1000 if max_step is None else int(max_step),
                'state_dim': 0,
                'action_dim': 0,
                'if_discrete': False, }
    for k, v in info_input.items():
        env_args[k] = v
    return env_args


def get_nonstationary_env_args(
        info_input: OWMR_input,
        horizon=13,
        all_demand_list=None,
        if_generate_demand=True,
        allocation_rule='prioritized_greedy',
        train_cost_mode='controllable',
        seed=None,
        **env_overrides,
):
    """Build the explicit environment contract for finite-horizon training."""
    env_kwargs = {
        'all_demand_list': all_demand_list,
        'info_input': info_input,
        'horizon': int(horizon),
        'holding_weight': 1.0,
        'penalty_weight': 0.0,
        'cost_plus': 0.0,
        'seed': seed,
        'if_generate_demand': bool(if_generate_demand),
        'method': 'SAC',
        'allocation_rule': allocation_rule,
        'train_cost_mode': train_cost_mode,
        'include_period_in_state': True,
        'include_action_mask_in_state': True,
    }
    env_kwargs.update(env_overrides)
    return get_env_args(
        env_kwargs,
        env_class=FiniteHorizonNonstationaryOWMREnv,
        max_step=horizon,
    )


def grid_search(agent_class, info_input, instance_name, cwd,
                lr_candidate=None, reward_scale_candidate=None,
                gamma_candidate=None, net_dims=None,
                break_step=None, eval_per_step=None, eval_times=None,
                eval_episode_steps=None, eval_burn_in_steps=None,
                training_eval_profile='short',
                horizon_len=None, buffer_size=None, repeat_times=None,
                batch_size=None, replay_warmup_steps=None,
                freeze_state_normalization_from_warmup=None,
                device='cpu', gpu_id=0, env_class=OWMRInventoryEnv,
                env_args_overrides=None):
    convex_config = None
    if agent_class.__name__.split('.')[-1] == 'AgentConvexSAC':
        convex_config = {'Actor': False, 'Critic': True}
    if agent_class.__name__.split('.')[-1] == 'AgentDiscreteMultiPPO':
        info_input['method'] = 'paper'

    lr_candidate = lr_candidate or [1e-2, 1e-3, 1e-4] # [1e-2, 1e-3, 1e-4, 1e-5]
    reward_scale_candidate = reward_scale_candidate or [2**0, 2**-6, 2**-12]  # [2**0, 2**-6, 2**-12]
    gamma_candidate = gamma_candidate or [0.9, 0.96]
    net_dims = net_dims or [(64, 32)]  # [(64, 32), (32, 64), (128, 64), (64, 128)], (32, 16), (64, 32), (128, 64), (256, 128)

    env_args = get_env_args(info_input, env_class=env_class)
    if env_args_overrides:
        env_args.update(env_args_overrides)
    args = Config(agent_class, env_class, env_args)
    args.device = device
    args.gpu_id = -1 if device == 'cpu' else gpu_id

    configure_training_evaluation(
        args,
        training_eval_profile,
        eval_per_step=1e4 if eval_per_step is None else eval_per_step,
        eval_times=eval_times,
        eval_episode_steps=eval_episode_steps,
        eval_burn_in_steps=eval_burn_in_steps,
    )
    args.break_step = int(1e6) if break_step is None else int(break_step)
    if horizon_len is not None:
        args.horizon_len = int(horizon_len)
    if buffer_size is not None:
        args.buffer_size = int(buffer_size)
    if repeat_times is not None:
        args.repeat_times = float(repeat_times)
    if batch_size is not None:
        args.batch_size = int(batch_size)
    if replay_warmup_steps is not None:
        args.replay_warmup_steps = int(replay_warmup_steps)
    if freeze_state_normalization_from_warmup is not None:
        args.freeze_state_normalization_from_warmup = bool(
            freeze_state_normalization_from_warmup
        )
    for lr in lr_candidate:
        for rs in reward_scale_candidate:
            for gamma in gamma_candidate:
                for net_dim in net_dims:
                    logger.info(f'本轮超参数为: 学习率{lr}, 回报系数{rs}, 折扣因子{gamma}, 隐藏层规模{net_dim}')
                    args.gamma = gamma
                    args.learning_rate = lr
                    args.reward_scale = rs
                    args.net_dims = net_dim

                    # arg_name = f'OWMRInventoryEnv_{agent_class.__name__[5:]}_{instance_name}'
                    # args.cwd = f'./{arg_name}'
                    agent_name = agent_class.__name__[5:]
                    arg_name = f'OWMRInventoryEnv_{instance_name}'
                    args.cwd = str(output_path("train", f'{agent_name}_{cwd}_{lr}_{rs}_{gamma}_{net_dim}', instance_name))
                    my_mkdir(args.cwd)
                    os.makedirs(args.cwd, exist_ok=True)
                    with open(f'{args.cwd}/{arg_name}.pkl', 'wb') as f:
                        pickle.dump(args, f)
                    train_agent(args, convex_config)


def train_(agent_class, info_input, instance_name, cwd, device='cpu', gpu_id=0,
           env_class=OWMRInventoryEnv, env_args_overrides=None,
           break_step=None, horizon_len=None, buffer_size=None,
           eval_per_step=None, eval_times=20, repeat_times=None,
           eval_episode_steps=None, eval_burn_in_steps=None,
           training_eval_profile='short',
           batch_size=None, replay_warmup_steps=None,
           freeze_state_normalization_from_warmup=None,
           actor_learning_rate=None, critic_learning_rate=None,
           actor_update_interval=1, actor_freeze_steps=0,
           actor_net_dims=None, critic_net_dims=None,
           critic_ensemble_size=2, critic_target_min_size=2,
           replay_sampler='uniform',
           learning_rate=None, alpha_learning_rate=None,
           reward_scale=None, gamma=None, net_dims=None,
           init_actor_path=None, resume_cwd=None, resume_recorder=True,
           initial_total_step=0, initial_max_r=-np.inf,
           if_save_best_training_state=False, if_save_buffer=False):
    agent_name = agent_class.__name__[5:]
    if agent_class.__name__.split('.')[-1] == 'AgentDiscreteMultiPPO':
        info_input['method'] = 'paper'
    env_args = get_env_args(info_input, env_class=env_class)
    if env_args_overrides:
        env_args.update(env_args_overrides)

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.device = device
    args.gpu_id = -1 if device == 'cpu' else gpu_id

    

    if args.if_off_policy:
        if env_class is FiniteHorizonNonstationaryOWMREnv:
            defaults = get_nonstationary_training_defaults(args.max_step)
            args.horizon_len = defaults['horizon_len'] if horizon_len is None else int(horizon_len)
            args.buffer_size = defaults['buffer_size'] if buffer_size is None else int(buffer_size)
            args.batch_size = defaults['batch_size'] if batch_size is None else int(batch_size)
            args.replay_warmup_steps = (
                defaults['replay_warmup_steps']
                if replay_warmup_steps is None
                else int(replay_warmup_steps)
            )
            args.freeze_state_normalization_from_warmup = (
                defaults['freeze_state_normalization_from_warmup']
                if freeze_state_normalization_from_warmup is None
                else bool(freeze_state_normalization_from_warmup)
            )
        else:
            args.horizon_len = args.max_step * 2 if horizon_len is None else int(horizon_len)
            args.buffer_size = int(1e4) if buffer_size is None else int(buffer_size)
            if batch_size is not None:
                args.batch_size = int(batch_size)
            if replay_warmup_steps is not None:
                args.replay_warmup_steps = int(replay_warmup_steps)
            if freeze_state_normalization_from_warmup is not None:
                args.freeze_state_normalization_from_warmup = bool(
                    freeze_state_normalization_from_warmup
                )
        args.repeat_times = 1.0 if repeat_times is None else float(repeat_times)
    
    default_break_step = (
        get_nonstationary_training_defaults(args.max_step)['break_step']
        if env_class is FiniteHorizonNonstationaryOWMREnv
        else int(5e7)
    )
    args.break_step = default_break_step if break_step is None else int(break_step)
    args.reward_scale = DRL_params[agent_name]['rs'] if reward_scale is None else float(reward_scale)
    args.learning_rate = DRL_params[agent_name]['lr'] if learning_rate is None else float(learning_rate)
    args.alpha_learning_rate = (
        None if alpha_learning_rate is None else float(alpha_learning_rate)
    )
    args.net_dims = DRL_params[agent_name]['net'] if net_dims is None else tuple(net_dims)
    args.actor_net_dims = None if actor_net_dims is None else tuple(actor_net_dims)
    args.critic_net_dims = None if critic_net_dims is None else tuple(critic_net_dims)
    args.actor_learning_rate = (
        None if actor_learning_rate is None else float(actor_learning_rate)
    )
    args.critic_learning_rate = (
        None if critic_learning_rate is None else float(critic_learning_rate)
    )
    args.actor_update_interval = int(actor_update_interval)
    args.actor_freeze_steps = int(actor_freeze_steps)
    args.critic_ensemble_size = int(critic_ensemble_size)
    args.critic_target_min_size = int(critic_target_min_size)
    args.replay_sampler = str(replay_sampler)
    default_gamma = (
        get_nonstationary_training_defaults(args.max_step)['gamma']
        if env_class is FiniteHorizonNonstationaryOWMREnv and args.if_off_policy
        else DRL_params[agent_name]['gamma']
    )
    args.gamma = default_gamma if gamma is None else float(gamma)
    args.init_actor_path = init_actor_path
    args.resume_cwd = resume_cwd
    args.resume_recorder = bool(resume_recorder)
    args.initial_total_step = int(initial_total_step)
    args.initial_max_r = float(initial_max_r)
    args.if_save_best_training_state = bool(if_save_best_training_state)
    args.if_save_buffer = bool(if_save_buffer)

    if agent_name == 'DiscreteMultiPPO':
        args.batch_size = DRL_params[agent_name]['batch size']
        args.horizon_len = DRL_params[agent_name]['batch size']
        args.lambda_gae_adv = DRL_params[agent_name]['GAE']
        args.lambda_entropy = DRL_params[agent_name]['entropy']
        args.clip_grad_norm = DRL_params[agent_name]['clipping norm']

    # args.reward_scale = 2 ** 0
    # args.learning_rate = 1e-2
    # args.net_dims = (128, 128)  # the middle layer dimension of MultiLayer Perceptron
    # args.gamma = 0.98  # discount factor of future rewards

    default_eval_step = (
        get_nonstationary_training_defaults(args.max_step)['eval_per_step']
        if env_class is FiniteHorizonNonstationaryOWMREnv
        else int(5e5)
    )
    configure_training_evaluation(
        args,
        training_eval_profile,
        eval_per_step=default_eval_step if eval_per_step is None else eval_per_step,
        eval_times=eval_times,
        eval_episode_steps=eval_episode_steps,
        eval_burn_in_steps=eval_burn_in_steps,
    )

    # args.gpu_id = GPU_ID
    args.num_workers = 4
    if_single_process = True

    arg_name = f'OWMRInventoryEnv_{instance_name}'
    args.cwd = str(output_path("train", f'{agent_name}_{cwd}', instance_name))
    my_mkdir(args.cwd)
    with open(f'{args.cwd}/{arg_name}.pkl', 'wb') as f:
        pickle.dump(args, f)

    if if_single_process:
        train_agent(args)
    else:
        train_agent_multiprocessing(args)


def grid_search_nonstationary(
        agent_class,
        input_env: OWMR_input,
        instance_name,
        cwd,
        horizon=13,
        all_demand_list=None,
        if_generate_demand=True,
        allocation_rule='prioritized_greedy',
        train_cost_mode='controllable',
        reference_weekly_cost=None,
        **grid_kwargs,
):
    env_args = get_nonstationary_env_args(
        info_input=input_env,
        horizon=horizon,
        all_demand_list=all_demand_list,
        if_generate_demand=if_generate_demand,
        allocation_rule=allocation_rule,
        train_cost_mode=train_cost_mode,
    )
    if _is_off_policy_agent(agent_class):
        defaults = get_nonstationary_training_defaults(horizon)
        grid_kwargs.setdefault('lr_candidate', [5e-5, 1e-4])
        grid_kwargs.setdefault('gamma_candidate', list(NONSTATIONARY_GAMMA_CANDIDATES))
        if reference_weekly_cost is None:
            grid_kwargs.setdefault('reward_scale_candidate', [2 ** -9, 2 ** -6])
        else:
            base_scale = nonstationary_reward_scale_from_reference_cost(reference_weekly_cost)
            grid_kwargs.setdefault(
                'reward_scale_candidate',
                [base_scale * 0.5, base_scale, base_scale * 2.0],
            )
        for key in (
            'horizon_len',
            'replay_warmup_steps',
            'freeze_state_normalization_from_warmup',
            'buffer_size',
            'batch_size',
            'repeat_times',
            'break_step',
            'eval_per_step',
            'eval_times',
        ):
            grid_kwargs.setdefault(key, defaults[key])
    return grid_search(
        agent_class=agent_class,
        info_input=env_args,
        instance_name=instance_name,
        cwd=cwd,
        env_class=FiniteHorizonNonstationaryOWMREnv,
        **grid_kwargs,
    )


def train_nonstationary(
        agent_class,
        input_env: OWMR_input,
        instance_name,
        cwd,
        horizon=13,
        all_demand_list=None,
        if_generate_demand=True,
        allocation_rule='prioritized_greedy',
        train_cost_mode='controllable',
        reference_weekly_cost=None,
        **train_kwargs,
):
    env_args = get_nonstationary_env_args(
        info_input=input_env,
        horizon=horizon,
        all_demand_list=all_demand_list,
        if_generate_demand=if_generate_demand,
        allocation_rule=allocation_rule,
        train_cost_mode=train_cost_mode,
    )
    if _is_off_policy_agent(agent_class):
        defaults = get_nonstationary_training_defaults(horizon)
        for key in (
            'horizon_len',
            'replay_warmup_steps',
            'freeze_state_normalization_from_warmup',
            'buffer_size',
            'batch_size',
            'repeat_times',
            'break_step',
            'eval_per_step',
            'eval_times',
            'gamma',
        ):
            train_kwargs.setdefault(key, defaults[key])
        if reference_weekly_cost is not None:
            train_kwargs.setdefault(
                'reward_scale',
                nonstationary_reward_scale_from_reference_cost(reference_weekly_cost),
            )
    return train_(
        agent_class=agent_class,
        info_input=env_args,
        instance_name=instance_name,
        cwd=cwd,
        env_class=FiniteHorizonNonstationaryOWMREnv,
        **train_kwargs,
    )



if __name__ == "__main__":
    # datasets = generate_ratio_data()
    # datasets = generate_data()
    datasets = generate_data_with_random_para(rdc_num=20, sample_num=10, random_seed=10)
    instance_name = 'dataset'
    instance_no = 4
    for data in datasets[4:]:
        instance_no += 1
        # if instance_no in [14, 15]:
        rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, RDC_inventory, CDC_inventory = data
        input_env, state_dim, action_dim = env_input_initialize(rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, RDC_inventory, CDC_inventory)
        
        Num_of_Instances_Generated = 100
        learning_rate, net_dims, reward_scale, holding_weight, penalty_weight, cost_plus, all_demand_list = auto_hyperparameter(input_env, Num_of_Instances_Generated)

        input_args = {}
        input_args['all_demand_list'] = all_demand_list
        input_args['holding_weight'] = holding_weight
        input_args['penalty_weight'] = penalty_weight
        input_args['cost_plus'] = cost_plus
        input_args['info_input'] = input_env

        # bom_instance_trained_result_dict = {}
        # 训练模块
        # trainer_test = trainer('', input_args, bom_instance_trained_result_dict)
        # trainer_test.interact_and_train_multi_process()
        agent_classes = [AgentPPO.AgentPPO, AgentTD3.AgentTD3, AgentSAC, AgentRecommendedSAC, AgentPPO.AgentDiscreteMultiPPO]
        agent_class = agent_classes[-1]
        # for agent_class in agent_classes[2:3]:
            # agent_class = [0]
        cwd = '0127_size20_paper_train'
        # grid_search(agent_class, input_args, f'{instance_name}_{instance_no}', cwd)
        print('action * 1:')
        train_(agent_class, input_args, f'{instance_name}_{instance_no}', cwd)
        # DRL_params['DiscreteMultiPPO']['rs'] = 2e-6
        # train_(agent_class, input_args, f'{instance_name}_{instance_no}', cwd)
        # DRL_params['DiscreteMultiPPO']['clipping norm'] = 4
        # DRL_params['DiscreteMultiPPO']['rs'] = 1
        # train_(agent_class, input_args, f'{instance_name}_{instance_no}', cwd)
        # DRL_params['DiscreteMultiPPO']['rs'] = 2e-6
        # train_(agent_class, input_args, f'{instance_name}_{instance_no}', cwd)
