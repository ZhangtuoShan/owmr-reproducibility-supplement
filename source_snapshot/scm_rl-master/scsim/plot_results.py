from collections import defaultdict

import loguru
from plotly import graph_objects as go
import numpy as np

default_colors = [
    '#1f77b4',  # muted blue
    '#ef553b',  # safety orange
    '#d62728',  # brick red
    '#2ca02c',  # cooked asparagus green
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]


def preprocess_results(results):
    all_agents = dict()

    for scenario_id, agent_result_for_scenario in results.items():
        for agent_name, values in agent_result_for_scenario.items():
            all_agents[agent_name] = None
    new = {}
    for agent_name in all_agents.keys():
        for scenario_id, results_for_scenario in results.items():
            agent_result_for_scenario = results_for_scenario.get(agent_name, {'Mean Score': 0, 'Std Error': 0})
            if agent_name not in new.keys():
                new[agent_name] = [[], []]
            new[agent_name][0].append(agent_result_for_scenario['Mean Score'])
            new[agent_name][1].append(agent_result_for_scenario['Std Error'])
    return new


def normalize(means, stds, values_to_normalize):
    assert len(means) == len(stds) == len(values_to_normalize)
    norm_mean, norm_stds = [], []
    for mean, std, norm_by_value in zip(means, stds, values_to_normalize):
        if norm_by_value == 0 or mean == 0:
            mean = 0
            std = 0
        else:
            mean /= norm_by_value
            mean *= 100
            mean -= 100
            std /= norm_by_value
            std *= 100
        norm_mean.append(mean)
        norm_stds.append(std)

    return norm_mean, norm_stds



def generate_table(
        scenario_results,
        normalize_by=None,
        plot_agents=None,
        width=1000,
        tick0=None,
        xaxis_title="Setting",
        yaxis_title="% cost difference",
        legend_title='Algorithms', color_shift=0, color_step=1
):
    agent_results = preprocess_results(scenario_results)
    scenario_ids = list(range(len(scenario_results.keys())))

    if tick0 is None:
        tick0 = 0

    if plot_agents is not None:
        agent_results = {k: v for k, v in agent_results.items() if k in plot_agents}


    from texttable import Texttable
    import latextable

    table = Texttable()
    table.set_cols_align(["c"] * (len(agent_results) * 3 + 1))

    rows = dict()

    for i, (agent_name, d) in enumerate(list(agent_results.items())):
        mean = d[0]
        std_error = d[1]
        rows[agent_name] = {}
        if normalize_by is not None:
            norm_by_results = agent_results[normalize_by][0]
            norm_mean, norm_std_error = normalize(mean, std_error, norm_by_results)
            rows[agent_name].update({'norm_mean': norm_mean, "norm_std_error": norm_std_error, })
        rows[agent_name].update({'mean': mean, 'std_error': std_error})

    # table.add_row(agent_results.keys())

    for scenario_id in scenario_ids:
        row = []
        for agent_name in rows.keys():
            mean = rows[agent_name]['mean'][scenario_id - 1]
            std_error = rows[agent_name]['std_error'][scenario_id - 1]
            norm_mean = rows[agent_name]['norm_mean'][scenario_id - 1]
            norm_std_error = rows[agent_name]['norm_std_error'][scenario_id - 1]
            mean = np.round(mean, decimals=2)
            std_error = np.round(std_error, decimals=2)
            norm_mean = np.round(norm_mean, decimals=2)
            norm_std_error = np.round(norm_std_error, decimals=2)
            for e in [mean, std_error, norm_mean]:
                # print()
                row.append("{}~".format(e))
            # elem = f'{mean}|\u00B1{std_error}|({norm_mean}\%)'
            # row.extend(elem.split('|'))

        table.add_row(row)
    print(latextable.draw_latex(table).replace('~', ''))




def plot_bar_chart(
        scenario_results,
        normalize_by=None,
        plot_agents=None,
        width=1000,
        tick0=None,
        xaxis_title="Setting",
        yaxis_title="% cost difference",
        legend_title='Algorithms', color_shift=0, color_step=1
):
    agent_results = preprocess_results(scenario_results)
    scenario_ids = list(scenario_results.keys())

    if tick0 is None:
        tick0 = 0

    if plot_agents is not None:
        agent_results = {k: v for k, v in agent_results.items() if k in plot_agents}

    fig = go.Figure()
    for i, (agent_name, d) in enumerate(list(agent_results.items())):
        mean = d[0]
        std_error = d[1]
        if normalize_by is not None:
            norm_by_results = agent_results[normalize_by][0]
            mean, std_error = normalize(mean, std_error, norm_by_results)
        fig.add_trace(
            go.Bar(
                name=agent_name,
                x=list([str(i) for i in scenario_ids]),
                y=mean,
                error_y=dict(type='data', array=std_error),
                marker_color=default_colors[color_shift + i * color_step]
            )
        )

    if normalize_by is not None:
        normalized_trace_id = list(agent_results.keys()).index(normalize_by)
        norm_line_color = default_colors[color_shift + normalized_trace_id * color_step]
        fig.add_shape(
            # Line Horizontal
            type="line",
            x0=tick0 - 0.5,
            y0=0,
            x1=tick0 + len(scenario_results) - .5,
            y1=0,
            line=dict(
                color=norm_line_color,
                width=2,
                #                 dash="dashdot",
            ), )

    fig.update_layout(barmode='group')

    fig.update_layout(
        #     title="Plot Title",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend_title=legend_title,
        font=dict(
            family="Courier New, monospace",
            size=16,
            #         color="RebeccaPurple"
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=scenario_ids,
            ticktext=[str(i) for i in scenario_ids],
        )
    )

    return fig


def drop_scenarios(scenario_results, scenario_ids):
    scenario_ids = [int(i) for i in scenario_ids]
    return {
        k: v for k, v in scenario_results.items() if int(k) not in scenario_ids
    }


def filter_scenarios(scenario_results, scenario_ids):
    scenario_ids = [int(i) for i in scenario_ids]
    return {
        k: v for k, v in scenario_results.items() if int(k) in scenario_ids
    }


def sort_agents(scenario_results, key=lambda item: item[0], reverse=False):
    from copy import deepcopy
    scenario_results_copy = deepcopy(scenario_results)
    for scenario_id, agent_results in scenario_results.items():
        scenario_results_copy[scenario_id] = {k: v for k, v in sorted(agent_results.items(), key=key, reverse=reverse)}

    return scenario_results_copy


def filter_agents(scenario_results, agent_names):
    agent_names = set(agent_names)
    filtered_scenario_results = defaultdict(dict)
    for scenario_id, agent_results in scenario_results.items():
        for agent_name, agent_result in agent_results.items():
            if agent_name in agent_names:
                filtered_scenario_results[scenario_id][agent_name] = agent_result

    return filtered_scenario_results


def rename_agent(scenario_results, old_to_new_map):
    renamed_scenario_results = defaultdict(dict)
    for scenario_id, agent_results in scenario_results.items():
        for agent_name, agent_result in agent_results.items():
            # loguru.logger.debug(old_to_new_map.get(agent_name, agent_name))
            new_name = old_to_new_map.get(agent_name, agent_name)
            renamed_scenario_results[scenario_id][new_name] = agent_result
    return renamed_scenario_results
