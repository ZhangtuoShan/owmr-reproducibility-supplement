import matplotlib.pyplot as plt
# from scsim.simulation import SimModel
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from os.path import join
import pandas as pd

from scsim.exponential_moving_average import ewma_vectorized

def plot_training_curve(run_path, fig=None):
    import plotly.graph_objects as go
    from scsim.visualization import plot_series
    df = pd.read_csv(join(run_path, 'progress.csv'))
    fig = plot_series(df, x='timesteps_total', y='episode_reward_mean', fig=fig,
                fig_config=dict(line=dict(width=2, dash='solid')))
    # fig = plot_series(df, x='timesteps_total', y='episode_reward_min', fig=fig,
    #             fig_config=dict(line=dict(width=2, dash='dot')))
    # fig = plot_series(df, x='timesteps_total', y='episode_reward_max', fig=fig,
    #             fig_config=dict(line=dict(width=2, dash='dot')))
    fig.show()
    return fig

def plot_comparison(df):
    agent_names = df['agent_name'].unique()

    violin_plot = make_subplots(rows=1, cols=len(agent_names), shared_yaxes=True)
    bar_plot = go.Figure()

    for i, agent_name in enumerate(agent_names, start=1):
        dff = df[df['agent_name'] == agent_name]
        violin_plot.append_trace(
            go.Violin(y=dff["score"], name=agent_name), row=1, col=i
        )
        mean_cost = dff.drop(columns=["score"]).mean()
        bar_plot.add_trace(
                go.Bar(y=mean_cost, x=mean_cost.index, name=agent_name)
            )

    violin_plot.update_traces(
        box_visible=True, meanline_visible=True, points="all", jitter=0.05
    )
    violin_plot.show()

    bar_plot.update_layout(
        xaxis_title="Costs", yaxis_title="Value",
    )
    bar_plot.show()


def plot_series(df, y,
                x=None,
                x_label=None,
                name=None,
                fig=None,
                fig_config={},
                type='line',
                cumulative=False,
                smoothing=None):
    if fig is None:
        fig = go.Figure()

    if isinstance(y, str):
        assert y in df.columns, f'value "{y}" is not in DataFrame'
        name = y if name is None else name
        y = df[y]

    if isinstance(x, str):
        assert x in df.columns or x == 'index', f'value "{x}" is not in DataFrame'
        x = df.index if x == 'index' else df[x]

    if x is None:
        x = np.arange(len(y))

    if cumulative:
        y = np.cumsum(y)

    if smoothing:
        y = ewma_vectorized(y, alpha=smoothing)

    if type == 'line':
        fig_config = {**dict(x=x, y=y, mode='lines', name=name, opacity=0.5), **fig_config}
        fig.add_trace(go.Scatter(**fig_config))
        fig.update_layout(xaxis_title=x_label, yaxis_title='value',)
    elif type == 'bar':
        fig_config = {**dict(x=x, y=y, name=name, opacity=0.5), **fig_config}
        fig.add_trace(go.Bar(**fig_config))
    elif type == 'histogram':
        fig_config = {**dict(x=y, histnorm='probability', name=name), **fig_config}
        fig.add_trace(go.Histogram(**fig_config))
        fig.update_layout(xaxis_title=f"value",
                      yaxis_title="Pr(value)", )

    return fig
    # fig.show()


def plot_averages(df, values, fig=None):
    if fig is None:
        fig = go.Figure()
    for value in values:
        assert value in df, f'value "{value}" is not in DataFrame'
        avg = df[value].mean()
        fig.add_trace(go.Bar(x=[value], y=[avg], name=value))
    return fig

# def plot_history(model, names):
#     fig, axs = plt.subplots(len(names), 1, figsize=(20, len(names) * 3), squeeze=False)
#
#     for i in range(len(names)):
#         axs[i, 0].stem(model.history.get('time'), model.history.get(names[i]), use_line_collection=True)
#         axs[i, 0].set_ylabel(names[i])


def plot_aggregate_costs(df, fig=None, show=False, name=None):
    if fig is None:
        fig = go.Figure()

    unique_attr_names = {c.split('.')[1] for c in df.columns}
    attrs = {c: [] for c in unique_attr_names}
    for i, cost_name in enumerate(unique_attr_names):
        current_costs = [c for c in df.columns if cost_name in c]
        cost_df = df[current_costs]
        s = cost_df.sum(axis=1).sum()
        attrs[cost_name] = s
    attrs = {k: v for k, v in sorted(attrs.items(), key=lambda item: item[1])}
    fig.add_trace(
        go.Bar(x=list(attrs.keys()),
               y=list(attrs.values()),
               name=name),

    )
    if show:
        fig.show()

    return fig


def plot_history(df, fig=None, y=None, fig_config={}, type='line', ):
    if fig is None:
        fig = go.Figure()
    if y is None:
        y = df.columns
    for c in y:
        if df[c].mean == 0:
            continue
        fig = plot_series(df, y=c, fig=fig, fig_config=fig_config, type=type)

    return fig

def plot_costs_per_location(df, new_names=None, col_names=None, normalize=None, fig=None):
    if fig is None:
        fig = go.Figure()
    df.sort_index(axis=1, inplace=True)
    df = df[[
        'retailer_0.shortage_cost',
        'retailer_0.holding_cost',
        'retailer_1.shortage_cost',
        'retailer_1.holding_cost',
        'retailer_2.shortage_cost',
        'retailer_2.holding_cost',
        # 'warehouse.holding_cost',
        'agent_name'
    ]]
    # except KeyError:
    #     df = df[[
    #         'retailer_1.backlog_cost',
    #         'retailer_1.holding_cost',
    #         'retailer_2.backlog_cost',
    #         'retailer_2.holding_cost',
    #         'retailer_3.backlog_cost',
    #         'retailer_3.holding_cost',
    #         # 'warehouse.holding_cost',
    #         'agent_name'
    #     ]
    #     ]
    agent_names = list(df['agent_name'].unique())
    if new_names is None:
        new_names = {k: k for k in agent_names}

    for i, agent_name in enumerate(agent_names, start=1):
        dff = df[df['agent_name'] == agent_name]
        mean_cost = dff.mean()

        if normalize:
            mean_norm = df[df['agent_name'] == normalize].mean()
            mean_cost /= mean_norm
            mean_cost -= 1
            mean_cost *= 100
        #         std_error /= mean_norm
        #         std_error *= 100

        fig.add_trace(
            go.Bar(y=mean_cost, x=mean_cost.index, name=new_names[agent_name])
        )

    fig.update_layout(
        xaxis_title="Costs", yaxis_title="Value",
        font=dict(
            family="Courier New, monospace",
            size=16,
            #         color="RebeccaPurple"
        ),
        xaxis = dict(
            #         tickmode = 'linear',
            #         tick0 = 1,
            #         dtick = 1,
            tickvals=list(range(0, 7)),
            ticktext=col_names,
            #         tickangle=20
        ),         legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,)
    )
    fig
    return fig

def plot_agg_costs_per_location(summary, new_names=None, normalize=None, fig=None):
    if fig is None:
        fig = go.Figure()

    agent_names = list(summary.index)
    if new_names is None:
        new_names = {k: k for k in agent_names}

    d = {}
    for a in ['warehouse', 'retailer_0', 'retailer_1', 'retailer_2']:
        if a == 'warehouse':
            mean_cost = summary['{}.holding_cost'.format(a)]['mean']
            std_error = summary['{}.holding_cost'.format(a)]['std_error']

        else:

            mean_holding_cost = summary['{}.holding_cost'.format(a)]['mean']
            std_error_holding_cost = summary['{}.holding_cost'.format(a)]['std_error']

            try:
                mean_shortage_cost = summary['{}.backlog_cost'.format(a)]['mean']
                std_error_backlog_cost = summary['{}.backlog_cost'.format(a)]['std_error']
            except KeyError:
                mean_shortage_cost = summary['{}.shortage_cost'.format(a)]['mean']
                std_error_backlog_cost = summary['{}.shortage_cost'.format(a)]['std_error']
            mean_cost = mean_holding_cost + mean_shortage_cost
            std_error = std_error_holding_cost + std_error_backlog_cost
        for agent_name in agent_names:
            if agent_name not in d.keys():
                d[agent_name] = [[], []]
            d[agent_name][0].append(mean_cost[agent_name])
            d[agent_name][1].append(std_error[agent_name])

    #     print(a, summary['{}.backlog_cost'.format(a)]['mean'] + summary['{}.holding_cost'.format(a)]['mean'],
    #          )
    #     fig.add_trace(go.Bar(x=s.index, y=s, name=a))
    if normalize:
        mean_norm = d[normalize][0]

    for agent_name, v in d.items():
        mean = np.asarray(v[0])
        std_error = np.asarray(v[1])

        if normalize:
            mean /= mean_norm
            mean -= 1
            mean *= 100
            std_error /= mean_norm
            std_error *= 100

        fig.add_trace(go.Bar(name=new_names[agent_name], y=mean,
                             # error_y=dict(type='data', array=std_error)
                             ))
    fig.update_layout(
        #     title="Plot Title",
        xaxis_title="Location",
        yaxis_title="Total Cost",
        # legend_title="Algorithms",
        font=dict(
            family="Courier New, monospace",
            size=16,
            #         color="RebeccaPurple"
        ),
        xaxis=dict(
            #         tickmode = 'linear',
            #         tick0 = 1,
            #         dtick = 1,
            tickvals=list(range(0, 4)),
            ticktext=[
                'Warehouse',
                'Retailer 1',
                'Retailer 2',
                'Retailer 3',

            ],
            #         tickangle=20
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,)
    )
    return fig

if __name__ == '__main__':
    import pandas as pd

    # df = pd.read_csv('../Experiments/PPO_GymWrapper_0_2020-04-07_17-23-50coyguqjt/progress.csv')
    df = pd.read_csv('../Experiments/newsvendor_backorder/ray_results/PPO/PPO_GymWrapper_0_2020-06-06_12-38-51oo65a3ql/progress.csv')
    # df = pd.read_csv('../Experiments/progress.csv')
    # df_1 = pd.read_csv('../Experiments/progress (1).csv')
    fig = go.Figure()
    # df['(s,Q) baseline'] = -224.0
    # df['Base stock policy'] = 1733.19
    # df['Zero order policy'] = -1172.0
    # df_1['Base stock policy random seed'] = 1686.7306
    # df_1['PPO random seed'] = 1658.322
    # df['Random Policy'] = -4465.0
    plot_series(df, x='timesteps_total', y='episode_reward_mean', fig=fig,
                fig_config=dict(line=dict(width=2, dash='solid')))
    plot_series(df, x='timesteps_total', y='episode_reward_min', fig=fig,
                fig_config=dict(line=dict(width=2, dash='dot')))
    plot_series(df, x='timesteps_total', y='episode_reward_max', fig=fig,
                fig_config=dict(line=dict(width=2, dash='dot')))
    # plot_series(df, x='timesteps_total', y='Base stock policy', fig=fig,
    #             fig_config=dict(line=dict(width=2, dash='solid')))
    # plot_series(df, x='timesteps_total', y='Random Policy', fig=fig, fig_config=dict(line=dict(width=2, dash='solid')))
    # plot_series(df_1, x='timesteps_total', y='episode_reward_mean', fig=fig,
    #             fig_config=dict(line=dict(width=2, dash='solid')))
    # plot_series(df_1, x='timesteps_total', y='Base stock policy random seed', fig=fig,
    #             fig_config=dict(line=dict(width=2, dash='solid')))
    # plot_series(df, x='timesteps_total', y='(s,S) baseline', fig=fig, fig_config=dict(line=dict(width=4, dash='solid')))
    fig.update_layout(xaxis_title="Training steps (in millions)",
                      yaxis_title="Score", )
    fig.show()