import os
from collections import defaultdict
from os.path import join, exists
import yaml
import json

from copy import deepcopy

import pandas as pd
from loguru import logger
from tqdm import tqdm

from scsim.agents import *

import numpy as np

from os.path import join

from scsim.visualization import plot_history, plot_comparison, plot_series

from scsim.supply_chain_env import OWMREnvironment

from scsim.grid_search import run_episode, evaluate, evaluate_heuristic, grid_search

try:
    import ray
except ModuleNotFoundError:
    print("Ray is not installed.")
import itertools

DEFAULT_CONFIG = "default_config.yaml"
EVALUATION_CONFIG = "evaluation_config.yaml"
TRAINING_CONFIG = "training_config.yaml"

def std_error(x):
    return np.std(x, ddof=1) / np.sqrt(np.size(x))

def load_configs(path_to_root):
    from scsim.utils import merge_dicts
    from yaml import load, FullLoader
    from copy import deepcopy

    default_config = load(
        open(join(path_to_root, DEFAULT_CONFIG)), Loader=yaml.FullLoader
    )
    eval_config = {}
    try:
        eval_config = (
            load(open(join(path_to_root, EVALUATION_CONFIG)), Loader=yaml.FullLoader)
            or {}
        )
    except FileNotFoundError:
        ...
    train_config = {}
    try:
        train_config = (
            load(open(join(path_to_root, TRAINING_CONFIG)), Loader=yaml.FullLoader)
            or {}
        )
    except FileNotFoundError:
        ...

    eval_config = merge_dicts(deepcopy(default_config), eval_config)
    train_config = merge_dicts(deepcopy(eval_config), train_config)

    return default_config, eval_config, train_config


class Experiment:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.default_env_config, self.eval_env_config, _ = load_configs(self.root_dir)
        self.agents = {}
        self.agent_data = {}

    def load_rllib_agent(self, run_path, name="rl_agent", force_eval=True, **kwargs):
        agent, env = load_rllib_agent(
            run_path=join(self.root_dir, run_path),
            eval_env_config=self.eval_env_config if force_eval else None,
            **kwargs,
        )
        self.agents[name] = (agent, env)
        self.agent_data[name] = {}

        self.agent_data[name]["run_path"] = run_path

        df = pd.read_csv(join(self.root_dir, run_path, "progress.csv"))
        self.agent_data[name]["training_data"] = df
        return agent, env

    def get_agents_data(self, attr):
        return [(agent_name, d[attr]) for agent_name, d in self.agent_data.items()]

    # TODO finish
    def plot_training_curves(self, fig=None):
        max_len = 0
        for agent_name, df in self.get_agents_data("training_data"):
            fig = plot_series(
                df,
                x="timesteps_total",
                y="episode_reward_mean",
                fig=fig,
                name=agent_name,
                fig_config=dict(line=dict(width=2, dash="solid")),
            )
            df_size = df["timesteps_total"].max()
            if df_size > max_len:
                max_len = df_size

        for agent_name, df in self.get_agents_data("evaluation_results"):
            mean = df["mean_score"]
            x = np.linspace(0, max_len, 100)
            y = np.repeat([mean], 100)
            fig = plot_series(
                df=None,
                x=x,
                y=y,
                fig=fig,
                name="eval_mean_{}".format(agent_name),
                fig_config=dict(line=dict(width=2, dash="solid")),
            )

        fig.show()
        return fig

    def add_agent(self, agent_name, agent):
        env_config = self.eval_env_config.copy()
        env = OWMREnvironment(env_config)
        self.agents.update({agent_name: (agent, env)})

    def add_heuristic_agent(
        self, agent_cls, v_params, name="heuristic_agent", interface=None
    ):

        if isinstance(agent_cls, str):
            agent_cls = eval(agent_cls)

        agent = agent_cls(v_params)
        env_config = deepcopy(self.default_env_config)

        env = OWMREnvironment(env_config)

        self.agents.update({name: (agent, env)})
        self.agent_data[name] = {}
        return agent, env

    def get_agent_model_summary(self, agent):
        if isinstance(agent, str):
            agent, _ = self.agents[agent]

        result = []
        for k, v in agent.get_weights().items():
            for k_, v_ in v.items():
                result.append((k_, v_.shape))
        return result

    def compare_agents(
        self,
        num_trials=100,
        warmup_steps=0,
        experiment_seed=None,
        run_seed=None,
        tag=None,
        show=True,
    ):

        df = pd.DataFrame()

        for i, (agent_name, (agent, env)) in enumerate(self.agents.items(), start=1):
            if self.agent_data[agent_name].get("run_path", None):
                agent_hash = "{}_{}".format(
                    self.agent_data[agent_name]["run_path"], agent_name
                )
            else:
                agent_hash = type(agent).__name__ + str(agent.v)
            if tag is not None:
                agent_hash = "{}({})".format(agent_hash, tag)
            agent_hash = "{}({})".format(agent_hash, num_trials)
            try:
                # raise FileNotFoundError
                dff = pd.read_csv(join(self.root_dir, agent_hash + ".csv"), index_col=0)

            except FileNotFoundError:
                dff = evaluate(
                    agent,
                    env,
                    warmup_steps=warmup_steps,
                    num_trials=num_trials,
                    experiment_seed=experiment_seed,
                    run_seed=run_seed,
                )

                dff["agent_name"] = agent_name
                dff.to_csv(join(self.root_dir, agent_hash + ".csv"))
            min_score, mean_score, max_score = dff["score"].agg(["min", "mean", "max"])
            current_std_error = dff["score"].std() / np.sqrt(len(dff))
            self.agent_data[agent_name]["evaluation_results"] = {
                "min_score": min_score,
                "mean_score": mean_score,
                "max_score": max_score,
                "std_error": current_std_error,
            }
            df = df.append(dff, ignore_index=False)

        # Remove all columns that contains only zeros
        # del df['history']
        df = df.loc[:, (df != 0).any(axis=0)]
        if show:
            plot_comparison(df)

        return (
            df.groupby("agent_name").agg(
                [
                    "mean",
                    std_error,
                    "min",
                    "max",
                ]
            ),
            df,
        )

    def plot_training_data(self, x="index", y=("episode_reward_mean",)):
        import plotly.graph_objects as go

        fig = go.Figure()
        for agent_name, df in self.training_info.items():
            if not hasattr(y, "__iter__"):
                y = (y,)
            for elem in y:
                # Skip values that are not in the data
                if elem in df.columns:
                    continue

                fig = plot_series(
                    df=df, x=x, y=elem, name="{}-{}".format(agent_name, elem), fig=fig
                )

        return fig

    def plot_bars(self, y, **kwargs):
        import plotly.graph_objects as go

        if not isinstance(y, (list, tuple)):
            y = [y]

        agent_data = defaultdict(dict)
        for agent_name in self.agents.keys():
            for y_elem in y:
                score, history = self.run_episode(
                    agent_name=agent_name, show=False, **kwargs
                )
                df = history.get_df()
                # logger.info(list(df[y_elem]))
                agent_data[agent_name][y_elem] = list(df[y_elem])
        fig = go.Figure()
        for agent_name, data_to_plot in agent_data.items():
            fig.add_trace(
                go.Bar(
                    x=list(data_to_plot.keys()),
                    y=[np.mean(arr) for arr in data_to_plot.values()],
                    name=agent_name,
                    error_y=dict(type='data', array=[np.std(arr) / np.sqrt(len(arr)) for arr in data_to_plot.values()]))
            )
        fig.update_layout(
            yaxis_title="",
            xaxis_title="",
            font=dict(
                family="Courier New, monospace",
                size=16,
                #         color="RebeccaPurple"
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        return fig

    def show_histogram(self, y, colors=None, **kwargs):
        import plotly.figure_factory as ff
        if not isinstance(y, (list, tuple)):
            y = [y]
        data = []
        names = []

        for agent_name in self.agents.keys():
            for y_elem in y:
                score, history = self.run_episode(
                    agent_name=agent_name, show=False, **kwargs
                )
                df = history.get_df()
                data.append(df[y_elem])
                name = '{}.{}'.format(agent_name, y_elem)
                names.append(name)

        # if names is None:
        #     names = list(self.agents.keys())
        # Create distplot with custom bin_size
        fig = ff.create_distplot(
            data, names, curve_type="normal", bin_size=1, colors=colors
        )
        fig.update_layout(
            yaxis_title="Probability",
            xaxis_title="Order",
            font=dict(
                family="Courier New, monospace",
                size=16,
                #         color="RebeccaPurple"
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        # fig.show()
        for d, n  in zip(data, names):
            logger.info(
                '{}: {} ({})'.format(n, np.mean(d), np.std(d))
            )
        return fig

    def compute_averages(self, y, **kwargs):
        import pandas as pd

        d = {}
        for agent_name in self.agents.keys():
            score, history = self.run_episode(
                agent_name=agent_name, show=False, **kwargs
            )
            dff = history.get_df()[y].mean()
            d[agent_name] = dff
        df = pd.DataFrame(d)
        return df

    def run_episode(self, agent_name, y=None, show=True, **kwargs):
        agent, env = self.agents[agent_name]
        score, history = run_episode(agent, env, **kwargs)

        if show:
            fig = plot_history(history.get_df(), y=y)
            fig.show()
        return score, history

    def run_episode_compare(
        self, y, agent_names=None, fig=None, episode_length=None, seed=None, **kwargs
    ):
        if agent_names is None:
            agent_names = list(self.agents.keys())

        dfs = []
        if seed is None:
            seed = np.random.randint(0, 10e6)

        print("Seed", seed)

        for agent_name in agent_names:
            score, history = self.run_episode(
                agent_name, show=False, run_seed=seed, episode_length=episode_length
            )
            print(agent_name, score)
            dfs.append(history.get_df())
        return compare(dfs, agent_names, y, fig=fig, **kwargs)


def load_rllib_agent(
    run_path,
    env_cls=OWMREnvironment,
    eval_env_config=None,
    checkpoint_name=None,
    **kwargs,
):
    import pickle
    from ray.rllib.agents import ppo

    # trainer_config = pickle.load(open(join(run_path, "params.pkl"), "rb"))
    trainer_config = json.load(open(join(run_path, "params.json"), "rb"))

    if checkpoint_name is None:
        checkpoints = [
            x for x in os.listdir(run_path) if os.path.isdir(join(run_path, x))
        ]
        checkpoints.sort(key=lambda x: int(x.split("_")[-1]), reverse=False)
        checkpoint_name = checkpoints[-1]

    checkpoint_path = join(
        run_path, checkpoint_name, 'checkpoint-' + str(int(checkpoint_name.split('_')[-1]))
    )

    if eval_env_config is None:
        eval_env_config = trainer_config["evaluation_config"].get(
            "env_config", trainer_config["env_config"]
        )
    trainer_config = {**trainer_config, "env": env_cls, **kwargs}
    if trainer_config.get('framework') == 'tf1':
        trainer_config['framework'] = 'tf'
        # trainer_config['use_critic'] = False
        # trainer_config['use_gae'] = False
    # trainer_config['simple_optimizer'] = True
    env = env_cls(eval_env_config)
    trainer = ppo.PPOTrainer(config=trainer_config)
    logger.info('Loading {}'.format(checkpoint_path))
    trainer.restore(checkpoint_path)
    print("loaded", checkpoint_name)
    return trainer, env


import plotly.graph_objects as go


def compare(
    dfs, agent_names, y=None, cumulative=False, fig=None, filter_cols=None, **kwargs
):
    if y is None:
        y = list(dfs[0].columns)
    # y = [v for v in y if filter_cols]

    if fig is None:
        fig = go.Figure()

    for col in y:
        for df, agent_name in zip(dfs, agent_names):
            fig = plot_series(
                df,
                y=col,
                fig=fig,
                cumulative=cumulative,
                name="{} - {}".format(agent_name, col),
                **kwargs,
            )
    return fig


if __name__ == "__main__":
    # root_folder = (
    #     "/workspace/supplychainsimulation/Experiments/distribution_problem_multiagent"
    # )
    # experiment = Experiment(root_folder)
    #
    # ray.init(webui_host="0.0.0.0")
    # # experiment.add_heuristic_agent("BaseStockAgent", [58, 9, 12, 16], name="basestock")
    # experiment.load_rllib_agent(
    #     "ray_results/PPO/PPO_MultiAgentWrapper_0_2020-07-22_16-43-46nh1y09n0",
    #     env_cls=MultiAgentWrapper,
    #     name="ppo",
    #     num_workers=0,
    #     explore=False,
    # )
    # df = experiment.compare_agents(num_trials=10, warmup_steps=0)
    # print(df)

    ranges = [
        list(range(59, 80)),
        list(range(10, 11)),
        list(range(10, 11)),
        list(range(10, 11)),
        list(range(10, 11)),
        list(range(10, 11)),
    ]

    grid_search(
        None,
        None,
        n_agents=6,
        ranges=[
            list(range(59, 80)),
            list(range(10, 11)),
            # list(range(10, 11)),
            # list(range(10, 11)),
            # list(range(10, 11)),
            # list(range(10, 11)),
        ],
    )
