import os
import torch
import numpy as np
from typing import List
from torch import Tensor
from multiprocessing import Pipe, Process

from output_paths import output_path
from elegantrl.train.device import resolve_torch_device


TRAINING_EVAL_PROFILES = {
    # Fast checkpoint screening during training. Publication evaluation is a
    # separate common-demand workflow and is intentionally not represented
    # here.
    "short": {
        "eval_times": 20,
        "eval_episode_steps": 300,
        "eval_burn_in_steps": 150,
    },
    # Historical training-time evaluation protocol, retained only for exact
    # reproduction of old runs.
    "legacy": {
        "eval_times": 100,
        "eval_episode_steps": 1000,
        "eval_burn_in_steps": 500,
    },
}


def configure_training_evaluation(
        args,
        profile="short",
        *,
        eval_per_step=None,
        eval_times=None,
        eval_episode_steps=None,
        eval_burn_in_steps=None,
):
    """Apply one centralized training-time evaluation protocol.

    ``short`` and ``legacy`` are fixed named protocols, so stale values passed
    by historical launchers cannot silently change their meaning. ``custom``
    is the explicit escape hatch for experiments that need different values.
    Evaluation frequency remains independent because changing it would also
    change the existing evaluator-driven checkpoint cadence.
    """
    profile = str(profile).strip().lower()
    if profile not in {*TRAINING_EVAL_PROFILES, "custom"}:
        choices = ", ".join((*TRAINING_EVAL_PROFILES, "custom"))
        raise ValueError(f"Unknown training evaluation profile {profile!r}; choose {choices}.")

    if profile == "custom":
        resolved = {
            "eval_times": getattr(args, "eval_times", 20) if eval_times is None else eval_times,
            "eval_episode_steps": (
                getattr(args, "eval_episode_steps", 300)
                if eval_episode_steps is None
                else eval_episode_steps
            ),
            "eval_burn_in_steps": (
                getattr(args, "eval_burn_in_steps", 150)
                if eval_burn_in_steps is None
                else eval_burn_in_steps
            ),
        }
    else:
        resolved = dict(TRAINING_EVAL_PROFILES[profile])

    # A finite-horizon episode must be evaluated over its actual horizon and
    # has no stationary burn-in segment. Stationary OWMR environments have a
    # much larger max_step and therefore retain the named profile unchanged.
    max_step = int(getattr(args, "max_step", 0) or 0)
    if 0 < max_step < int(resolved["eval_episode_steps"]):
        resolved["eval_episode_steps"] = max_step
        resolved["eval_burn_in_steps"] = 0

    resolved = {key: int(value) for key, value in resolved.items()}
    if resolved["eval_times"] <= 0:
        raise ValueError("eval_times must be positive")
    if resolved["eval_episode_steps"] <= 0:
        raise ValueError("eval_episode_steps must be positive")
    if not 0 <= resolved["eval_burn_in_steps"] < resolved["eval_episode_steps"]:
        raise ValueError(
            "eval_burn_in_steps must satisfy "
            "0 <= eval_burn_in_steps < eval_episode_steps"
        )

    args.training_eval_profile = profile
    args.eval_times = resolved["eval_times"]
    args.eval_episode_steps = resolved["eval_episode_steps"]
    args.eval_burn_in_steps = resolved["eval_burn_in_steps"]
    if eval_per_step is not None:
        args.eval_per_step = int(eval_per_step)
    return resolved


class Config:
    def __init__(self, agent_class=None, env_class=None, env_args=None):
        self.num_envs = None
        self.agent_class = agent_class  # agent = agent_class(...)
        self.if_off_policy = self.get_if_off_policy()  # whether off-policy or on-policy of DRL algorithm

        '''Argument of environment'''
        self.env_class = env_class  # env = env_class(**env_args)
        self.env_args = env_args  # env = env_class(**env_args)
        if env_args is None:  # dummy env_args
            env_args = {'env_name': None,
                        'num_envs': 1,
                        'max_step': 12345,
                        'state_dim': None,
                        'action_dim': None,
                        'if_discrete': None, }
        env_args.setdefault('num_envs', 1)  # `num_envs=1` in default in single env.
        env_args.setdefault('max_step', 12345)  # `max_step=12345` in default, which is a large enough value.
        self.env_name = env_args['env_name']  # the name of environment. Be used to set 'cwd'.
        self.num_envs = env_args['num_envs']  # the number of sub envs in vectorized env. `num_envs=1` in single env.
        self.max_step = env_args['max_step']  # the max step number of an episode. 'set as 12345 in default.
        self.state_dim = env_args['state_dim']  # vector dimension (feature number) of state
        self.action_dim = env_args['action_dim']  # vector dimension (feature number) of action
        self.if_discrete = env_args['if_discrete']  # discrete or continuous action space
        self.action_mask_state_start = env_args.get('action_mask_state_start')
        self.action_mask_dim = int(env_args.get('action_mask_dim', 0) or 0)
        self.inactive_action_value = float(env_args.get('inactive_action_value', 0.0))

        '''Arguments for reward shaping'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256

        '''Arguments for training'''
        self.net_dims = (64, 32)  # the middle layer dimension of MLP (MultiLayer Perceptron)
        self.actor_net_dims = None
        self.critic_net_dims = None
        self.critic_ensemble_size = 2
        self.critic_target_min_size = 2
        self.learning_rate = 6e-5  # the learning rate for network updating
        self.clip_grad_norm = 3.0  # 0.1 ~ 4.0, clip the gradient after normalization
        self.state_value_tau = 0  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
        self.replay_warmup_steps = None  # None preserves the legacy one-horizon random warm-up
        self.freeze_state_normalization_from_warmup = False
        self.replay_store_after_period = 0
        # Optional inventory-aware replay sampling. ``uniform`` preserves the
        # historical global-RNG implementation exactly; the other modes use a
        # dedicated replay RNG so paired online experiments are reproducible.
        self.replay_sampler = "uniform"
        self.replay_focus_fraction = 0.25
        self.replay_sampler_refresh_updates = 10
        self.replay_strata_rdc_lead_times = None
        self.replay_strata_cdc_lead_time = None
        # Optional RB-guided initialization for SAC-family agents. ``replay``
        # fills the replay buffer with RB base-stock actions;
        # ``behavior_clone`` clones the actor and retains the normal random
        # replay warm-up; ``replay_and_clone`` performs both.
        self.rb_bootstrap_mode = "none"
        self.rb_bootstrap_steps = None
        self.rb_bootstrap_method = "RB_distribution_aware_capped"
        self.rb_bootstrap_action_noise_std = 0.0
        self.rb_bc_updates = 1000
        self.rb_bc_batch_size = 256
        self.rb_bc_learning_rate = None
        self.rb_bc_dataset_size = 50000
        # Optional persistent RB guidance. These fields are inert for the
        # standard AgentSAC and are consumed by AgentSACResidualSafe only.
        self.actor_learning_rate = None
        self.critic_learning_rate = None
        self.actor_freeze_steps = 0
        self.actor_update_interval = 1
        self.rb_anchor_coefficient_initial = 0.0
        self.rb_anchor_coefficient_final = 0.0
        self.rb_anchor_decay_steps = 0
        self.rb_anchor_batch_size = 256
        self.rb_teacher_replay_fraction = 0.0
        # ``None`` preserves the historical fixed teacher-replay fraction.
        # Residual-safe experiments may instead decay the fraction linearly.
        self.rb_teacher_replay_fraction_final = None
        self.rb_teacher_replay_decay_steps = 0
        self.rb_guidance_dataset_size = 50000
        self.alpha_learning_rate = None  # None preserves AgentSAC's legacy 4 * learning_rate
        # Opt-in critic target horizon for AgentSACLeadTimeNstep. Standard SAC
        # ignores this setting and retains its historical one-step target.
        self.critic_n_step = 1
        # Optional post-load overrides for complete-state continuation. These
        # are applied after optimizer state dicts restore their saved rates.
        self.resume_learning_rate_override = None
        self.resume_alpha_learning_rate_override = None
        # Optional actor-only warm restart for historical checkpoints that did
        # not preserve critic, optimizer, and replay-buffer state.
        self.init_actor_path = None
        self.initial_total_step = 0
        self.initial_max_r = -np.inf
        self.soft_update_tau = 5e-3  # 2 ** -8 ~= 5e-3. the tau of soft target update `net = (1-tau)*net + tau*net1`
        if self.if_off_policy:  # off-policy
            self.batch_size = int(64)  # num of transitions sampled from replay buffer.
            self.horizon_len = int(512)  # collect horizon_len step while exploring, then update networks
            self.buffer_size = int(1e5)  # ReplayBuffer size. First in first out for off-policy.
            self.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
            self.if_use_per = False  # use PER (Prioritized Experience Replay) for sparse reward
        else:  # on-policy
            self.batch_size = int(128)  # num of transitions sampled from replay buffer.
            self.horizon_len = int(2048)  # collect horizon_len step while exploring, then update network
            self.buffer_size = None  # ReplayBuffer size. Empty the ReplayBuffer for on-policy.
            self.repeat_times = 8.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
            self.if_use_vtrace = False  # use V-trace + GAE (Generalized Advantage Estimation) for sparse reward

        '''Arguments for device'''
        self.device = None  # None keeps legacy gpu_id behavior; use "cpu", "auto", "cuda", or "mps".
        self.rollout_device = None  # None keeps rollout on training device; set "cpu" for CPU env interaction.
        self.gpu_id = int(0)  # `int` means the ID of single GPU, -1 means CPU
        self.num_workers = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.num_threads = 8  # cpu_num for pytorch, `torch.set_num_threads(self.num_threads)`
        self.random_seed = 0  # initialize random seed in self.init_before_training()
        self.learner_gpus = 0  # `int` means the ID of single GPU, -1 means CPU

        '''Arguments for evaluate'''
        self.cwd = None  # current working directory to save model. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = np.inf  # break training if 'total_step > break_step'
        self.break_score = np.inf  # break training if `cumulative_rewards > break_score`
        self.if_keep_save = True  # keeping save the checkpoint. False means save until stop training.
        self.if_over_write = False  # overwrite the best policy network. `self.cwd/actor.pth`
        self.if_save_buffer = False  # if save the replay buffer for continuous training after stop training
        # Optional resumable snapshot written whenever evaluation reaches a new
        # best. It includes the complete agent optimization state and replay
        # buffer under cwd/best_training_state.
        self.if_save_best_training_state = False

        # Optional two-tier checkpoint selection. Frequent short evaluations
        # remain the screening layer; shortlisted checkpoints are promoted on
        # a separate, reproducibly reseeded medium validation environment.
        self.checkpoint_validation_enabled = False
        self.checkpoint_validation_interval = 200000
        self.checkpoint_validation_initial_step = 10000
        self.checkpoint_validation_top_k = 3
        self.checkpoint_validation_times = 100
        self.checkpoint_validation_episode_steps = 500
        self.checkpoint_validation_burn_in_steps = 250
        self.checkpoint_validation_seed = 700001

        self.save_gap = int(8)  # save actor f"{cwd}/actor_*.pth" for learning curve.
        # Default training-time checkpoint screening protocol. Publication
        # evaluation remains a separate 1000 x 1000, burn-in-500 workflow.
        self.training_eval_profile = "short"
        self.eval_times = int(20)
        self.eval_per_step = int(2e4)  # evaluate the agent per training steps
        # Stationary-environment evaluation protocol. Finite-horizon
        # environments use their own horizon and no burn-in.
        self.eval_episode_steps = int(300)
        self.eval_burn_in_steps = int(150)
        self.eval_env_class = None  # eval_env = eval_env_class(*eval_env_args)
        self.eval_env_args = None  # eval_env = eval_env_class(*eval_env_args)

    def init_before_training(self):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        '''set cwd (current working directory) for saving model'''
        if self.cwd is None:  # set cwd (current working directory) for saving model
            self.cwd = str(output_path("train", f'{self.env_name}_{self.agent_class.__name__[5:]}_{self.random_seed}'))

        '''remove history'''
        # if self.if_remove is None:
        #     self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
        # if self.if_remove:
        #     import shutil
        #     shutil.rmtree(self.cwd, ignore_errors=True)
        #     print(f"| Arguments Remove cwd: {self.cwd}")
        # else:
        #     print(f"| Arguments Keep cwd: {self.cwd}")
        # os.makedirs(self.cwd, exist_ok=True)

    def get_if_off_policy(self) -> bool:
        agent_name = self.agent_class.__name__ if self.agent_class else ''
        on_policy_names = ('SARSA', 'VPG', 'A2C', 'A3C', 'TRPO', 'PPO', 'MPO')
        return all([agent_name.find(s) == -1 for s in on_policy_names])

    def print(self):
        from pprint import pprint
        pprint(vars(self))  # prints out args in a neat, readable format


def build_env(env_class=None, env_args: dict = None, gpu_id: int = -1):
    env_args['gpu_id'] = gpu_id  # set gpu_id for vectorized env before build it

    if env_args.get('if_build_vec_env'):
        num_envs = env_args['num_envs']
        env = VecEnv(env_class=env_class, env_args=env_args, num_envs=num_envs, gpu_id=gpu_id)
    elif env_class.__module__ == 'gym.envs.registration':
        import gym
        assert '0.18.0' <= gym.__version__ <= '0.25.2'  # pip3 install gym==0.24.0
        gym.logger.set_level(40)  # Block warning
        env = env_class(id=env_args['env_name'])
    else:
        env = env_class(**kwargs_filter(env_class.__init__, env_args.copy()))

    # env_args.setdefault('num_envs', 1)
    # env_args.setdefault('max_step', 12345)

    # for attr_str in ('env_name', 'num_envs', 'max_step', 'state_dim', 'action_dim', 'if_discrete'):
    #     setattr(env, attr_str, env_args[attr_str])
    return env


def kwargs_filter(function, kwargs: dict) -> dict:
    import inspect
    parameters = inspect.signature(function).parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
        return kwargs
    sign = {val.name for val in parameters}
    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}  # filtered kwargs


def get_gym_env_args(env, if_print: bool) -> dict:
    """get a dict about a standard OpenAI gym env information.
    assert 0.18.0 <= gym.__version__ <= 0.25.3

    env: a standard OpenAI gym env
    if_print: [bool] print the dict about env information.
    return: env_args [dict]

    env_args = {
        'env_name': env_name,       # [str] the environment name, such as XxxXxx-v0
        'num_envs': num_envs.       # [int] the number of sub envs in vectorized env. `num_envs=1` in single env.
        'max_step': max_step,       # [int] the max step number of an episode.
        'state_dim': state_dim,     # [int] the dimension of state
        'action_dim': action_dim,   # [int] the dimension of action or the number of discrete action
        'if_discrete': if_discrete, # [bool] action space is discrete or continuous
    }
    """
    import gym

    if_gym_standard_env = {'unwrapped', 'observation_space', 'action_space', 'spec'}.issubset(dir(env))

    if if_gym_standard_env and (not hasattr(env, 'num_envs')):  # isinstance(env, gym.Env):
        assert '0.18.0' <= gym.__version__ <= '0.25.2'  # pip3 install gym==0.24.0
        env_name = env.unwrapped.spec.id
        num_envs = getattr(env, 'num_envs', 1)
        max_step = getattr(env, '_max_episode_steps', 12345)

        state_shape = env.observation_space.shape
        state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

        if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if if_discrete:  # make sure it is discrete action space
            action_dim = getattr(env.action_space, 'n')
        elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
            action_dim = env.action_space.shape[0]
            if any(env.action_space.high - 1):
                print('WARNING: env.action_space.high', env.action_space.high)
            if any(env.action_space.low + 1):
                print('WARNING: env.action_space.low', env.action_space.low)
        else:
            raise RuntimeError('\n| Error in get_gym_env_info(). Please set these value manually:'
                               '\n  `state_dim=int; action_dim=int; if_discrete=bool;`'
                               '\n  And keep action_space in range (-1, 1).')
    else:
        env_name = getattr(env, 'env_name', 'env')
        num_envs = getattr(env, 'num_envs', 1)
        max_step = getattr(env, 'max_step', 12345)
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete

    env_args = {'env_name': env_name,
                'num_envs': num_envs,
                'max_step': max_step,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'if_discrete': if_discrete, }
    if if_print:
        env_args_str = repr(env_args).replace(',', f",\n{'':11}")
        print(f"env_args = {env_args_str}")
    return env_args


"""vectorized env"""


class SubEnv(Process):
    def __init__(self, sub_pipe0: Pipe, vec_pipe1: Pipe,
                 env_class, env_args: dict, env_id: int = 0):
        super().__init__()
        self.sub_pipe0 = sub_pipe0
        self.vec_pipe1 = vec_pipe1

        self.env_class = env_class
        self.env_args = env_args
        self.env_id = env_id

    def run(self):
        torch.set_grad_enabled(False)

        '''build env'''
        if self.env_class.__module__ == 'gym.envs.registration':  # is standard OpenAI Gym env
            env = self.env_class(id=self.env_args['env_name'])
        else:
            env = self.env_class(**kwargs_filter(self.env_class.__init__, self.env_args.copy()))

        '''set env random seed'''
        random_seed = self.env_id
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        while True:
            action = self.sub_pipe0.recv()
            if action is None:
                state = env.reset()
                self.vec_pipe1.send((self.env_id, state))
            else:
                state, reward, done, info_dict = env.step(action)
                state = env.reset() if done else state
                self.vec_pipe1.send((self.env_id, state, reward, done, info_dict))


class VecEnv:
    def __init__(self, env_class: object, env_args: dict, num_envs: int, gpu_id: int = -1):
        self.device = resolve_torch_device(env_args.get("device"), gpu_id)
        self.num_envs = num_envs  # the number of sub env in vectorized env.

        '''the necessary env information when you design a custom env'''
        self.env_name = env_args['env_name']  # the name of this env.
        self.max_step = env_args['max_step']  # the max step number in an episode for evaluation
        self.state_dim = env_args['state_dim']  # feature number of state
        self.action_dim = env_args['action_dim']  # feature number of action
        self.if_discrete = env_args['if_discrete']  # discrete action or continuous action

        '''speed up with multiprocessing: Process, Pipe'''
        assert self.num_envs <= 64
        self.res_list = [[] for _ in range(self.num_envs)]

        sub_pipe0s, sub_pipe1s = list(zip(*[Pipe(duplex=False) for _ in range(self.num_envs)]))
        self.sub_pipe1s = sub_pipe1s

        vec_pipe0, vec_pipe1 = Pipe(duplex=False)  # recv, send
        self.vec_pipe0 = vec_pipe0

        self.sub_envs = [
            SubEnv(sub_pipe0=sub_pipe0, vec_pipe1=vec_pipe1,
                   env_class=env_class, env_args=env_args, env_id=env_id)
            for env_id, sub_pipe0 in enumerate(sub_pipe0s)
        ]

        [setattr(p, 'daemon', True) for p in self.sub_envs]  # set before process start to exit safely
        [p.start() for p in self.sub_envs]

    def reset(self) -> Tensor:  # reset the agent in env
        torch.set_grad_enabled(False)

        for pipe in self.sub_pipe1s:
            pipe.send(None)
        states, = self.get_orderly_zip_list_return()
        states = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
        return states

    def step(self, action: Tensor) -> (Tensor, Tensor, Tensor, List[dict]):  # agent interacts in env
        action = action.detach().cpu().numpy()
        if self.if_discrete:
            action = action.squeeze(1)
        for pipe, a in zip(self.sub_pipe1s, action):
            pipe.send(a)

        states, rewards, dones, info_dicts = self.get_orderly_zip_list_return()
        states = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        return states, rewards, dones, info_dicts

    def close(self):
        [process.terminate() for process in self.sub_envs]

    def get_orderly_zip_list_return(self):
        for _ in range(self.num_envs):
            res = self.vec_pipe0.recv()
            self.res_list[res[0]] = res[1:]
        return list(zip(*self.res_list))
