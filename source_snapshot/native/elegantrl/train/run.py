import os
import sys
import time
import json
import shutil
import torch
import numpy as np
import torch.multiprocessing as mp  # torch.multiprocessing extends multiprocessing of Python
from copy import deepcopy
from multiprocessing import Process, Pipe

from elegantrl.train.config import Config, build_env, configure_training_evaluation
from elegantrl.train.device import resolve_torch_device
from elegantrl.train.replay_buffer import ReplayBuffer
from elegantrl.train.rb_bootstrap import initialize_sac_from_rb
from elegantrl.train.checkpoint_validation import CheckpointValidationManager
from elegantrl.train.evaluator import Evaluator, get_cumulative_rewards_and_steps
from my_log import logger

if os.name == 'nt':  # if is WindowOS (Windows NT)
    """Fix bug about Anaconda in WindowOS
    OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

'''train'''


def ensure_env_random_seeds(env_args, base_seed):
    """Give demand and allocation reproducible but separate RNG streams."""
    if env_args.get("seed") is None:
        env_args["seed"] = int(base_seed)
    if env_args.get("allocation_seed") is None:
        env_args["allocation_seed"] = (
            int(env_args["seed"]) + 1_000_003
        ) % (2 ** 32)
    return env_args


def warm_up_replay_buffer(agent, env, buffer, horizon_len, warmup_steps=None,
                          freeze_state_normalization=False):
    """Fill an off-policy replay buffer and optionally freeze warm-up state statistics."""
    target_steps = horizon_len if warmup_steps is None else int(warmup_steps)
    if target_steps <= 0:
        raise ValueError(f"replay_warmup_steps must be positive, got {target_steps}")

    state_sum = None
    state_square_sum = None
    state_count = 0
    remaining = target_steps
    while remaining > 0:
        collect_steps = min(horizon_len, remaining)
        buffer_items = agent.explore_env(env, collect_steps, if_random=True)
        buffer.update(buffer_items)
        if freeze_state_normalization:
            states = buffer_items[0].reshape((-1, agent.state_dim)).to(
                device=agent.device, dtype=torch.float64
            )
            chunk_sum = states.sum(dim=0)
            chunk_square_sum = states.square().sum(dim=0)
            state_sum = chunk_sum if state_sum is None else state_sum + chunk_sum
            state_square_sum = (
                chunk_square_sum
                if state_square_sum is None
                else state_square_sum + chunk_square_sum
            )
            state_count += states.shape[0]
        remaining -= collect_steps

    if freeze_state_normalization:
        state_avg = state_sum / state_count
        state_var = (state_square_sum / state_count - state_avg.square()).clamp_min(0.0)
        state_std = state_var.sqrt().clamp_min(1.0)
        state_avg = state_avg.to(device=agent.device, dtype=torch.float32)
        state_std = state_std.to(device=agent.device, dtype=torch.float32)

        visited = set()
        for network_name in ('act', 'act_target', 'cri', 'cri_target'):
            network = getattr(agent, network_name, None)
            if network is None or id(network) in visited:
                continue
            visited.add(id(network))
            if hasattr(network, 'state_avg') and hasattr(network, 'state_std'):
                network.state_avg.copy_(state_avg)
                network.state_std.copy_(state_std)
        logger.info(
            f"| Frozen state normalization from {state_count} random warm-up states: "
            f"mean_abs={state_avg.abs().mean().item():.3f}, "
            f"std_mean={state_std.mean().item():.3f}"
        )

    logger.info(f"| Replay buffer random warm-up: steps={target_steps}")


def save_training_snapshot(
        agent,
        buffer,
        evaluator,
        cwd,
        target_name="best_training_state",
        snapshot_kind="best_complete_optimization_state_with_replay_buffer",
        metrics=None,
):
    """Atomically replace a resumable agent, optimizer, and replay snapshot."""
    target = os.path.join(cwd, target_name)
    temporary = os.path.join(cwd, f".{target_name}_tmp_{os.getpid()}")
    previous = os.path.join(cwd, f".{target_name}_previous")
    shutil.rmtree(temporary, ignore_errors=True)
    os.makedirs(temporary, exist_ok=False)

    agent.save_or_load_agent(temporary, if_save=True)
    if hasattr(buffer, "save_or_load_history"):
        buffer.save_or_load_history(temporary, if_save=True)
    np.save(os.path.join(temporary, "recorder.npy"), np.asarray(evaluator.recorder))
    metadata = {
        "format_version": 1,
        "source_cwd": os.path.abspath(cwd),
        "snapshot_kind": snapshot_kind,
        **((metrics if metrics is not None else evaluator.last_eval_metrics) or {}),
    }
    with open(os.path.join(temporary, "best_checkpoint.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
        file.write("\n")

    shutil.rmtree(previous, ignore_errors=True)
    if os.path.exists(target):
        os.replace(target, previous)
    os.replace(temporary, target)
    shutil.rmtree(previous, ignore_errors=True)
    logger.info(
        f"| Saved resumable best training state: step={metadata.get('step')}, "
        f"avgR={metadata.get('avg_r', metadata.get('validation_avg_r')):.6f}, path={target}"
    )


def save_best_training_snapshot(agent, buffer, evaluator, cwd):
    save_training_snapshot(agent, buffer, evaluator, cwd)


def promote_candidate_training_snapshot(cwd, step, validation):
    """Atomically promote an already-saved Top-K complete state."""
    source = os.path.join(cwd, f"checkpoint_candidate_state_{int(step):012}")
    target = os.path.join(cwd, "best_validation_state")
    temporary = os.path.join(cwd, f".best_validation_state_copy_tmp_{os.getpid()}")
    previous = os.path.join(cwd, ".best_validation_state_copy_previous")
    if not os.path.isdir(source):
        raise FileNotFoundError(f"Missing complete candidate state: {source}")
    shutil.rmtree(temporary, ignore_errors=True)
    shutil.copytree(source, temporary)
    metadata_path = os.path.join(temporary, "best_checkpoint.json")
    metadata = {
        "format_version": 1,
        "source_cwd": os.path.abspath(cwd),
        "snapshot_kind": (
            "medium_validation_promoted_complete_optimization_state_"
            "with_replay_buffer"
        ),
        "step": int(step),
        "avg_r": float(validation["validation_avg_r"]),
        "se_r": float(validation["validation_se_r"]),
        "median_r": float(validation["validation_median_r"]),
        "short_avg_r": float(validation["short_avg_r"]),
        "short_rank": int(validation["short_rank"]),
    }
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
        file.write("\n")
    shutil.rmtree(previous, ignore_errors=True)
    if os.path.exists(target):
        os.replace(target, previous)
    os.replace(temporary, target)
    shutil.rmtree(previous, ignore_errors=True)
    logger.info(
        f"| Promoted queued complete state: step={step}, "
        f"avgR={validation['validation_avg_r']:.6f}, path={target}"
    )


def process_checkpoint_validation_event(event, agent, buffer, evaluator, cwd):
    if event is None:
        return
    if event.get("candidate_added"):
        step = int(event["candidate_step"])
        save_training_snapshot(
            agent,
            buffer,
            evaluator,
            cwd,
            target_name=f"checkpoint_candidate_state_{step:012}",
            snapshot_kind="short_top_k_candidate_complete_state",
        )
    for step in event.get("evicted_steps", []):
        shutil.rmtree(
            os.path.join(cwd, f"checkpoint_candidate_state_{int(step):012}"),
            ignore_errors=True,
        )
        actor_path = os.path.join(cwd, f"checkpoint_candidate_actor__{int(step):012}.pt")
        if os.path.exists(actor_path):
            os.remove(actor_path)
    improved = [item for item in event.get("validations", []) if item["improved"]]
    if improved:
        winner = improved[-1]
        promote_candidate_training_snapshot(cwd, winner["step"], winner)


def apply_resume_optimizer_overrides(
        agent,
        learning_rate=None,
        alpha_learning_rate=None,
):
    """Override step sizes after complete optimizer states have been loaded."""
    applied = {}
    if learning_rate is not None:
        learning_rate = float(learning_rate)
        if learning_rate < 0:
            raise ValueError("resume learning-rate override must be nonnegative")
        optimizer_names = []
        for optimizer_name in ("act_optimizer", "cri_optimizer"):
            optimizer = getattr(agent, optimizer_name, None)
            if optimizer is None:
                continue
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate
            optimizer_names.append(optimizer_name)
        applied["learning_rate"] = learning_rate
        applied["optimizers"] = optimizer_names

    if alpha_learning_rate is not None:
        alpha_learning_rate = float(alpha_learning_rate)
        if alpha_learning_rate < 0:
            raise ValueError("resume alpha learning-rate override must be nonnegative")
        optimizer = getattr(agent, "alpha_optimizer", None)
        if optimizer is None:
            raise AttributeError("agent has no alpha_optimizer for the requested override")
        for param_group in optimizer.param_groups:
            param_group["lr"] = alpha_learning_rate
        applied["alpha_learning_rate"] = alpha_learning_rate
    return applied


def load_initial_actor(agent, actor_path):
    """Warm-start an actor while deliberately rebuilding SAC optimization state."""
    try:
        source_actor = torch.load(
            actor_path, map_location=agent.device, weights_only=False
        )
    except TypeError:
        source_actor = torch.load(actor_path, map_location=agent.device)
    state_dict = (
        source_actor.state_dict()
        if isinstance(source_actor, torch.nn.Module)
        else source_actor
    )
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Expected an actor module or state dict in {actor_path}, "
            f"got {type(source_actor).__name__}."
        )
    agent.act.load_state_dict(state_dict, strict=True)
    agent.act_target.load_state_dict(agent.act.state_dict(), strict=True)

    # Historical actor checkpoints carry the normalization used by the policy.
    # Give fresh critics the same state coordinates before optimization starts.
    for network_name in ("cri", "cri_target"):
        network = getattr(agent, network_name, None)
        if network is None:
            continue
        for stat_name in ("state_avg", "state_std"):
            source = getattr(agent.act, stat_name, None)
            target = getattr(network, stat_name, None)
            if source is not None and target is not None:
                target.copy_(source)
    logger.info(
        f"| Loaded actor-only warm restart from {actor_path}; "
        "critic, optimizers, temperature, and replay buffer are newly initialized"
    )


def train_agent(args: Config, convex_config=None):
    configure_training_evaluation(
        args,
        getattr(args, "training_eval_profile", "short"),
        eval_per_step=getattr(args, "eval_per_step", 20_000),
        eval_times=getattr(args, "eval_times", 20),
        eval_episode_steps=getattr(args, "eval_episode_steps", 300),
        eval_burn_in_steps=getattr(args, "eval_burn_in_steps", 150),
    )
    logger.info(
        "| Training evaluation "
        f"profile={args.training_eval_profile}, interval={args.eval_per_step}, "
        f"episodes={args.eval_times}, steps={args.eval_episode_steps}, "
        f"burn_in={args.eval_burn_in_steps}"
    )
    args.init_before_training()
    os.makedirs(args.cwd, exist_ok=True)
    torch.set_grad_enabled(False)
    env_device = getattr(args, "rollout_device", None) or args.device
    ensure_env_random_seeds(args.env_args, args.random_seed)
    args.env_args["device"] = env_device
    resume_cwd = getattr(args, "resume_cwd", None)
    resume_recorder = getattr(args, "resume_recorder", True)
    init_actor_path = getattr(args, "init_actor_path", None)
    if resume_cwd and init_actor_path:
        raise ValueError("resume_cwd and init_actor_path are mutually exclusive")

    '''init environment'''
    env = build_env(args.env_class, args.env_args, args.gpu_id)
    # RB bootstrap needs the original demand/cost specification once, while
    # keeping the legacy environment constructor and serialization unchanged.
    if "info_input" in args.env_args:
        env._rb_info_input = args.env_args["info_input"]
    args.state_dim = env.state_dim
    args.action_dim = env.action_dim
    args.action_mask_state_start = getattr(env, 'action_mask_state_start', None)
    args.action_mask_dim = getattr(env, 'action_mask_dim', 0)
    args.inactive_action_value = getattr(env, 'inactive_action_value', 0.0)
    # args.method = env.method

    '''init agent'''
    if convex_config:
        agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args, convex_config=convex_config)
    elif env.method == 'paper':
        agent = args.agent_class(args.net_dims, args.state_dim, env.length, gpu_id=args.gpu_id, args=args)
    else:
        agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    if resume_cwd:
        agent.save_or_load_agent(resume_cwd, if_save=False)
        resume_overrides = apply_resume_optimizer_overrides(
            agent,
            learning_rate=getattr(args, "resume_learning_rate_override", None),
            alpha_learning_rate=getattr(args, "resume_alpha_learning_rate_override", None),
        )
        if resume_overrides:
            logger.info(f"| Applied post-load optimizer overrides: {resume_overrides}")

    '''init agent.last_state'''
    state = env.reset()
    if args.num_envs == 1:
        assert state.shape == (args.state_dim,)
        assert isinstance(state, np.ndarray)
        state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    else:
        assert state.shape == (args.num_envs, args.state_dim)
        assert isinstance(state, torch.Tensor)
        state = state.to(agent.device)
    assert state.shape == (args.num_envs, args.state_dim)
    assert isinstance(state, torch.Tensor)
    agent.last_state = state.detach()

    '''init buffer'''
    if args.if_off_policy:
        buffer_class = getattr(agent, 'buffer_class', ReplayBuffer)
        buffer = buffer_class(
            gpu_id=args.gpu_id,
            num_seqs=args.num_envs,
            max_size=args.buffer_size,
            state_dim=args.state_dim,
            action_dim=1 if args.if_discrete else args.action_dim,
            if_use_per=args.if_use_per,
            args=args,
        )
        rb_bootstrap_mode = getattr(args, "rb_bootstrap_mode", "none")
        if resume_cwd and rb_bootstrap_mode != "none":
            raise ValueError(
                "RB bootstrap is an initialization protocol and cannot be combined "
                "with --resume-from-cwd."
            )
        replay_loaded = False
        if resume_cwd and hasattr(buffer, 'save_or_load_history'):
            history_names = tuple(name for _, name in buffer.get_history_items())
            history_exists = all(
                os.path.isfile(f"{resume_cwd}/replay_buffer_{name}.pth")
                for name in history_names
            )
            if history_exists:
                buffer.save_or_load_history(resume_cwd, if_save=False)
                replay_loaded = buffer.cur_size > 1
                logger.info(
                    f"| Resume replay buffer from {resume_cwd}: cur_size={buffer.cur_size}"
                )
        if not replay_loaded:
            if rb_bootstrap_mode != "none":
                initialize_sac_from_rb(agent=agent, env=env, buffer=buffer, args=args)
                # BC-only keeps the conventional random replay initialization;
                # its RB data are supervised labels rather than off-policy data.
                if rb_bootstrap_mode == "behavior_clone":
                    warm_up_replay_buffer(
                        agent=agent,
                        env=env,
                        buffer=buffer,
                        horizon_len=args.horizon_len,
                        warmup_steps=getattr(args, "replay_warmup_steps", None),
                        freeze_state_normalization=False,
                    )
            else:
                warm_up_replay_buffer(
                    agent=agent,
                    env=env,
                    buffer=buffer,
                    horizon_len=args.horizon_len,
                    warmup_steps=getattr(args, 'replay_warmup_steps', None),
                    # A resumed agent already carries the frozen normalization
                    # statistics used by its actor and critics. Re-estimating them
                    # from a fresh replay warm-up would silently change the input
                    # semantics of every restored network.
                    freeze_state_normalization=(
                        getattr(args, 'freeze_state_normalization_from_warmup', False)
                        and not resume_cwd
                    ),
                )
    else:
        buffer = []

    if init_actor_path:
        load_initial_actor(agent, init_actor_path)

    '''init evaluator'''
    eval_env_class = args.eval_env_class if args.eval_env_class else args.env_class
    eval_env_args = args.eval_env_args if args.eval_env_args else args.env_args
    ensure_env_random_seeds(eval_env_args, args.random_seed)
    eval_env_args["device"] = env_device
    eval_env = build_env(eval_env_class, eval_env_args, args.gpu_id)
    evaluator = Evaluator(cwd=args.cwd, env=eval_env, args=args, if_tensorboard=False)
    checkpoint_validator = None
    checkpoint_validation_env = None
    if getattr(args, "checkpoint_validation_enabled", False):
        validation_env_args = deepcopy(eval_env_args)
        validation_seed = int(getattr(args, "checkpoint_validation_seed", 700001))
        validation_env_args["seed"] = validation_seed
        validation_env_args["allocation_seed"] = validation_seed + 1_000_003
        checkpoint_validation_env = build_env(
            eval_env_class, validation_env_args, args.gpu_id
        )
        checkpoint_validator = CheckpointValidationManager(
            cwd=args.cwd,
            env=checkpoint_validation_env,
            args=args,
        )
    if resume_cwd and resume_recorder:
        recorder_path = f"{resume_cwd}/recorder.npy"
        if os.path.exists(recorder_path):
            recorder = np.load(recorder_path)
            evaluator.recorder = [tuple(item) for item in recorder]
            if evaluator.recorder:
                evaluator.total_step = int(evaluator.recorder[-1][0])
                evaluator.eval_step_counter = evaluator.total_step
                evaluator.max_r = max(float(item[1]) for item in evaluator.recorder)
                logger.info(
                    f"| Resume training from {resume_cwd}: "
                    f"total_step={evaluator.total_step}, max_r={evaluator.max_r:.3f}"
                )
        else:
            logger.warning(f"| Resume recorder not found: {recorder_path}")
    elif init_actor_path:
        evaluator.total_step = int(getattr(args, "initial_total_step", 0))
        evaluator.eval_step_counter = evaluator.total_step
        evaluator.max_r = float(getattr(args, "initial_max_r", -np.inf))
        logger.info(
            f"| Actor warm restart counters: total_step={evaluator.total_step}, "
            f"max_r={evaluator.max_r:.6f}"
        )

    '''train loop'''
    cwd = args.cwd
    break_step = args.break_step
    horizon_len = args.horizon_len
    if_off_policy = args.if_off_policy
    if_save_buffer = args.if_save_buffer
    if_save_best_training_state = getattr(args, "if_save_best_training_state", False)
    del args

    if_train = True
    reallocation_agent = None
    while if_train:
        buffer_items = agent.explore_env(env, horizon_len)

        exp_r = buffer_items[3].mean().item() if len(buffer_items) in (5, 6) else buffer_items[2].mean().item()
        if if_off_policy:
            online_add_size = int(buffer_items[0].shape[0])
            if hasattr(agent, "mix_rb_replay_items"):
                buffer_items = agent.mix_rb_replay_items(buffer_items)
            buffer.update(buffer_items)
            buffer.online_add_size = online_add_size
        else:
            buffer[:] = buffer_items

        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        torch.set_grad_enabled(False)

        reallocation_agent = agent if hasattr(agent, 'soft_lagrangian_reallocation') else None
        if_new_best = evaluator.evaluate_and_save(
            actor=agent.act,
            steps=horizon_len,
            exp_r=exp_r,
            logging_tuple=logging_tuple,
            reallocation_agent=reallocation_agent,
        )
        if if_new_best and if_save_best_training_state:
            save_best_training_snapshot(agent, buffer, evaluator, cwd)
        if checkpoint_validator is not None:
            validation_event = checkpoint_validator.consider(
                actor=agent.act,
                short_metrics=evaluator.last_eval_metrics,
                reallocation_agent=reallocation_agent,
            )
            process_checkpoint_validation_event(
                validation_event, agent, buffer, evaluator, cwd
            )
        if_train = (evaluator.total_step <= break_step) and (not os.path.exists(f"{cwd}/stop"))

    logger.info(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')

    if checkpoint_validator is not None:
        process_checkpoint_validation_event(
            checkpoint_validator.finalize(reallocation_agent),
            agent,
            buffer,
            evaluator,
            cwd,
        )

    env.close() if hasattr(env, 'close') else None
    if checkpoint_validation_env is not None and hasattr(checkpoint_validation_env, "close"):
        checkpoint_validation_env.close()
    evaluator.save_training_curve_jpg()

    # with open(f'{args.cwd}/{arg_name}.pkl', 'wb') as f:
    #     pickle.dump(args, f)


    agent.save_or_load_agent(cwd, if_save=True)
    if if_save_buffer and hasattr(buffer, 'save_or_load_history'):
        buffer.save_or_load_history(cwd, if_save=True)


def train_agent_multiprocessing(args: Config):
    configure_training_evaluation(
        args,
        getattr(args, "training_eval_profile", "short"),
        eval_per_step=getattr(args, "eval_per_step", 20_000),
        eval_times=getattr(args, "eval_times", 20),
        eval_episode_steps=getattr(args, "eval_episode_steps", 300),
        eval_burn_in_steps=getattr(args, "eval_burn_in_steps", 150),
    )
    logger.info(
        "| Training evaluation "
        f"profile={args.training_eval_profile}, interval={args.eval_per_step}, "
        f"episodes={args.eval_times}, steps={args.eval_episode_steps}, "
        f"burn_in={args.eval_burn_in_steps}"
    )
    args.init_before_training()
    ensure_env_random_seeds(args.env_args, args.random_seed)
    if args.eval_env_args:
        ensure_env_random_seeds(args.eval_env_args, args.random_seed)

    """Don't set method='fork' when send tensor in GPU"""
    # method = 'spawn' if os.name == 'nt' else 'forkserver'  # os.name == 'nt' means Windows NT operating system (WinOS)
    # method = 'forkserver'
    method = 'spawn'
    mp.set_start_method(method=method, force=True)

    '''build the Pipe'''
    '''
    每个Pipe()返回管道的两端, [conn1, conn2]
    duplex=False 表示单向传信息, 只能从conn2向conn1发信息, conn1只能收, conn2只能发
    duplex=True 表示双向传递信息

    worker_pipe连接各worker与learner, 由learner向worker传actor
    learner_pipe连接learner与各worker, 由worker向learner传buffer_items, last_state
    evaluator_pipe连接evaluator与learner, 由learner向evaluator传训练进度actor and training log, 由evaluator向learner传训练中止信号
    '''
    worker_pipes = [Pipe(duplex=False) for _ in range(args.num_workers)]  # receive, send
    learner_pipe = Pipe(duplex=False)
    evaluator_pipe = Pipe(duplex=True)

    '''build Process'''
    learner = Learner(learner_pipe=learner_pipe, worker_pipes=worker_pipes, evaluator_pipe=evaluator_pipe, args=args)
    workers = [Worker(worker_pipe=worker_pipe, learner_pipe=learner_pipe, worker_id=worker_id, args=args)
               for worker_id, worker_pipe in enumerate(worker_pipes)]
    evaluator = EvaluatorProc(evaluator_pipe=evaluator_pipe, args=args)

    '''start Process'''
    process_list = [learner, *workers, evaluator]
    [process.start() for process in process_list]
    for p in process_list:
        p.join(timeout=5)  # 设置超时时间
        if p.is_alive():
            logger.error(f"Process {p.name} did not terminate in time, terminating.")
            p.terminate()  # 强制终止子进程
            p.join()  # 等待子进程终止
    # [process.join() for process in process_list]
    # print('Process finished')


class Learner(Process):
    def __init__(self, learner_pipe: Pipe, worker_pipes: [Pipe], evaluator_pipe: Pipe, args: Config):
        super().__init__()
        self.recv_pipe = learner_pipe[0]
        self.send_pipes = [worker_pipe[1] for worker_pipe in worker_pipes]
        self.eval_pipe = evaluator_pipe[1]
        self.args = args

    def run(self):
        args = self.args
        torch.set_grad_enabled(False)

        '''init agent'''
        agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
        agent.save_or_load_agent(args.cwd, if_save=False)

        '''init buffer'''
        if args.if_off_policy:
            buffer = ReplayBuffer(
                gpu_id=args.gpu_id,
                num_seqs=args.num_envs * args.num_workers,
                max_size=args.buffer_size,
                state_dim=args.state_dim,
                action_dim=1 if args.if_discrete else args.action_dim,
                if_use_per=args.if_use_per,
                args=args,
            )
        else:
            buffer = []

        '''loop'''
        if_off_policy = args.if_off_policy
        if_save_buffer = args.if_save_buffer
        num_workers = args.num_workers
        num_envs = args.num_envs
        state_dim = args.state_dim
        action_dim = args.action_dim
        horizon_len = args.horizon_len
        num_seqs = args.num_envs * args.num_workers
        num_steps = args.horizon_len * args.num_workers
        cwd = args.cwd
        del args

        agent.last_state = torch.empty((num_seqs, state_dim), dtype=torch.float32, device=agent.device)

        states = torch.empty((horizon_len, num_seqs, state_dim), dtype=torch.float32, device=agent.device)
        actions = torch.empty((horizon_len, num_seqs, action_dim), dtype=torch.float32, device=agent.device)
        rewards = torch.empty((horizon_len, num_seqs), dtype=torch.float32, device=agent.device)
        undones = torch.empty((horizon_len, num_seqs), dtype=torch.bool, device=agent.device)
        if if_off_policy:
            buffer_items_tensor = (states, actions, rewards, undones)
        else:
            logprobs = torch.empty((horizon_len, num_seqs), dtype=torch.float32, device=agent.device)
            buffer_items_tensor = (states, actions, logprobs, rewards, undones)

        if_train = True
        while if_train:
            '''Learner send actor to Workers'''
            for send_pipe in self.send_pipes:
                send_pipe.send(agent.act)

            '''Learner receive (buffer_items, last_state) from Workers'''
            for _ in range(num_workers):
                worker_id, buffer_items, last_state = self.recv_pipe.recv()

                buf_i = worker_id * num_envs
                buf_j = worker_id * num_envs + num_envs
                for buffer_item, buffer_tensor in zip(buffer_items, buffer_items_tensor):
                    buffer_tensor[:, buf_i:buf_j] = buffer_item
                agent.last_state[buf_i:buf_j] = last_state

            '''Learner update training data to (buffer, agent)'''
            if if_off_policy:
                buffer.update(buffer_items_tensor)
            else:
                buffer[:] = buffer_items_tensor

            '''agent update network using training data'''
            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)

            '''Learner receive training signal from Evaluator'''
            if self.eval_pipe.poll():  # whether there is any data available to be read of this pipe
                if_train = self.eval_pipe.recv()  # True means evaluator in idle moments.
                actor = agent.act  # so Leaner send an actor to evaluator for evaluation.
            else:
                actor = None

            '''Learner send actor and training log to Evaluator'''
            exp_r = buffer_items_tensor[2].mean().item()  # the average rewards of exploration
            self.eval_pipe.send((actor, num_steps, exp_r, logging_tuple))

        '''Learner send the terminal signal to workers after break the loop'''
        for send_pipe in self.send_pipes:
            send_pipe.send(None)

        '''save'''
        agent.save_or_load_agent(cwd, if_save=True)
        if if_save_buffer and hasattr(buffer, 'save_or_load_history'):
            logger.info(f"| LearnerPipe.run: ReplayBuffer saving in {cwd}")
            buffer.save_or_load_history(cwd, if_save=True)
            logger.info(f"| LearnerPipe.run: ReplayBuffer saved  in {cwd}")
        logger.info('Learner finished')


class Worker(Process):
    def __init__(self, worker_pipe: Pipe, learner_pipe: Pipe, worker_id: int, args: Config):
        super().__init__()
        self.recv_pipe = worker_pipe[0]
        self.send_pipe = learner_pipe[1]
        self.worker_id = worker_id
        self.args = args

    def run(self):
        args = self.args
        worker_id = self.worker_id
        torch.set_grad_enabled(False)

        '''init environment'''
        env = build_env(args.env_class, args.env_args, args.gpu_id)

        '''init agent'''
        agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
        agent.save_or_load_agent(args.cwd, if_save=False)

        '''init agent.last_state'''
        state = env.reset()
        if args.num_envs == 1:
            assert state.shape == (args.state_dim,)
            assert isinstance(state, np.ndarray)
            state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        else:
            assert state.shape == (args.num_envs, args.state_dim)
            assert isinstance(state, torch.Tensor)
            state = state.to(agent.device)
        assert state.shape == (args.num_envs, args.state_dim)
        assert isinstance(state, torch.Tensor)
        agent.last_state = state.detach()

        '''init buffer'''
        horizon_len = args.horizon_len
        if args.if_off_policy:
            buffer_items = agent.explore_env(env, args.horizon_len, if_random=True)
            self.send_pipe.send((worker_id, buffer_items, agent.last_state))

        '''loop'''
        del args

        while True:
            '''Worker receive actor from Learner'''
            actor = self.recv_pipe.recv()
            if actor is None:
                break

            '''Worker send the training data to Learner'''
            agent.act = actor
            buffer_items = agent.explore_env(env, horizon_len)
            self.send_pipe.send((worker_id, buffer_items, agent.last_state))

        env.close() if hasattr(env, 'close') else None
        logger.info(f'Worker {self.worker_id} finished')


class EvaluatorProc(Process):
    def __init__(self, evaluator_pipe: Pipe, args: Config):
        super().__init__()
        self.pipe = evaluator_pipe[0]
        self.args = args

    def run(self):
        args = self.args
        torch.set_grad_enabled(False)

        '''wandb(weights & biases): Track and visualize all the pieces of your machine learning pipeline.'''
        wandb = None
        if getattr(args, 'if_use_wandb', False):
            import wandb
            wandb_project_name = "train"
            wandb.init(project=wandb_project_name)

        '''init evaluator'''
        eval_env_class = args.eval_env_class if args.eval_env_class else args.env_class
        eval_env_args = args.eval_env_args if args.eval_env_args else args.env_args
        eval_env = build_env(eval_env_class, eval_env_args, args.gpu_id)
        evaluator = Evaluator(cwd=args.cwd, env=eval_env, args=args, if_tensorboard=False)

        '''loop'''
        cwd = args.cwd
        break_step = args.break_step
        device = resolve_torch_device(getattr(args, "device", None), args.gpu_id)
        del args

        if_train = True
        while if_train:
            '''Evaluator receive training log from Learner'''
            actor, steps, exp_r, logging_tuple = self.pipe.recv()
            wandb.log({"obj_cri": logging_tuple[0], "obj_act": logging_tuple[1]}) if wandb else None

            '''Evaluator evaluate the actor and save the training log'''
            if actor is None:
                evaluator.total_step += steps  # update total_step but don't update recorder
            else:
                actor = actor.to(device)
                evaluator.evaluate_and_save(actor, steps, exp_r, logging_tuple)

            '''Evaluator send the training signal to Learner'''
            if_train = (evaluator.total_step <= break_step) and (not os.path.exists(f"{cwd}/stop"))
            self.pipe.send(if_train)

        '''Evaluator save the training log and draw the learning curve'''
        evaluator.save_training_curve_jpg()
        logger.info(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')

        eval_env.close() if hasattr(eval_env, 'close') else None
        logger.info('Evaluator finished')


'''render'''


def render_agent(env_class, env_args: dict, net_dims: [int], agent_class, actor_path: str, render_times: int = 8):
    env = build_env(env_class, env_args)

    state_dim = env_args['state_dim']
    action_dim = env_args['action_dim']
    agent = agent_class(net_dims, state_dim, action_dim, gpu_id=-1)
    actor = agent.act
    del agent

    logger.info(f"| render and load actor from: {actor_path}")
    actor.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))
    for i in range(render_times):
        cumulative_reward, episode_step = get_cumulative_rewards_and_steps(env, actor, if_render=True)
        logger.info(f"|{i:4}  cumulative_reward {cumulative_reward:9.3f}  episode_step {episode_step:5.0f}")
