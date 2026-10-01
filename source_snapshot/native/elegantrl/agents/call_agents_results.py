import torch
import numpy as np
import pandas as pd
from scipy import integrate, stats

from Benchmark import (OWMR_input, OWMRInventoryEnv, auto_hyperparameter,
                       env_input_initialize)
from dataset import *
from elegantrl.agents import (AgentA2C, AgentConvexSAC, AgentDDPG, AgentDQN,
                              AgentPPO, AgentSAC, AgentTD3)
from elegantrl.train.config import Config
from elegantrl.train.run import *
from evaluation_statistics import mean_se_median
from utils import logger, my_mkdir

from main_DRL_agents import get_env_args

MAX_WEEK = 1000
OUTPUT_PATH = './output_OWMR'
my_mkdir(OUTPUT_PATH)

def create_env_and_actor(env_class_name, all_demand_list, input_env, eval_times):
    agent_class = AgentSAC
    env_class = env_class_name
    env_args = {'env_name': 'test',
                'num_envs': 1,
                'max_step': 13,
                'state_dim': input_env.state_dim,
                'action_dim': input_env.action_dim,
                'if_discrete': False,
                'all_demand_list': all_demand_list,
                'item_dict': input_env.item_dict,
                'rdc_dict': input_env.rdc_dict,
                'selling_item': input_env.selling_item,
                'demand_feature': input_env.demand_feature,
                'initial_inventory_cdc': input_env.CDC_inventory,
                'initial_inventory_rdc': input_env.RDC_inventory,
                }
    args = Config(agent_class, env_class, env_args)
    gamma = 0.9
    # args.net_dims = net_dims  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = gamma  # discount factor of future rewards
    args.horizon_len = args.max_step
    args.random_seed = 0
    args.eval_times = eval_times
    args.gpu_id = GPU_ID = 0
    '''init environment'''
    env = build_env(args.env_class, args.env_args, args.gpu_id)

    return args, env


def evaluate_single_agent(all_demand_list, info_input, agent_class, actor_path: str, eval_times=100):
    env_args = get_env_args(info_input)  
    env_class = OWMRInventoryEnv

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation


    env = build_env(args.env_class, args.env_args, args.gpu_id)
    agent = agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    logger.info(f"| render and load actor from: {actor_path}")
    state_dict = torch.load(actor_path, map_location=torch.device('cpu'))
    actor = agent.act
    actor.load_state_dict(state_dict)

    rdc_dict = info_input.rdc_dict

    calc_step = MAX_WEEK / 2
    # 初始化决策
    policy_order_rdc, policy_order_cdc = {}, {}
    
    policy_order_rdc = {rdc: np.zeros((eval_times, calc_step)) for rdc in rdc_dict.keys()}
    policy_order_cdc = np.zeros((eval_times, calc_step))

    inv_on_hand_rdc = {rdc: np.zeros((eval_times, calc_step)) for rdc in rdc_dict.keys()}
    inv_on_hand_cdc = np.zeros((eval_times, calc_step))

    backorder_info = {}
    order_up_to_level_rdc = {r: np.zeros((eval_times, calc_step)) for r in rdc_dict.keys()}
    order_up_to_level_cdc = np.zeros((eval_times, calc_step))

    echelon_inventory_rdc = {r: np.zeros((eval_times, calc_step)) for r in rdc_dict.keys()}
    echelon_inventory_cdc = np.zeros((eval_times, calc_step))

    rewards_step = np.zeros((eval_times, calc_step))
    rewards = np.zeros(eval_times)

    for i in range(eval_times):
        logger.info(f'Episode {i}:')
        returns, steps, rws, por, poc, ior, ioc, oulr, oulc, eir, eic = get_cumulative_rewards_and_steps(env, actor)
        for rdc in rdc_dict.keys():
            policy_order_rdc[rdc][i] = por[rdc]
            inv_on_hand_rdc[rdc][i] = ior[rdc]
            order_up_to_level_rdc[rdc][i] = oulr[rdc]
            echelon_inventory_rdc[rdc][i] = eir[rdc]
        policy_order_cdc[i] = poc
        inv_on_hand_cdc[i] = ioc
        order_up_to_level_cdc[i] = oulc
        echelon_inventory_cdc[i] = eic
        rewards[i] = returns
        rewards_step[i] = rws

    record_results_dataframe(rewards, rewards_step, policy_order_rdc, policy_order_cdc, inv_on_hand_rdc, inv_on_hand_cdc, order_up_to_level_rdc, order_up_to_level_cdc, echelon_inventory_rdc, echelon_inventory_cdc, agent_name=agent_class)


def record_results_dataframe(rewards, rewards_step, policy_order_rdc, policy_order_cdc, inv_on_hand_rdc, inv_on_hand_cdc, order_up_to_level_rdc, order_up_to_level_cdc, echelon_inventory_rdc, echelon_inventory_cdc, agent_name):
    rdcs = list(policy_order_rdc.keys())
    eval_times = np.shape(policy_order_cdc)[0]
    calc_steps = np.shape(policy_order_cdc)[1]
    df_agent_result = pd.DataFrame(columns=['Episode', 'Step', 'Location', 'On hand inventory', 'Echelon inventory', 'Order quantity', 'Order-up-to level'])
    datas = []
    for i in range(eval_times):
        for j in range(calc_steps):
            for x in rdcs:
                datas.append([i, j, 'Retailer %d'%x, inv_on_hand_rdc[x][i, j], echelon_inventory_rdc[x][i, j], policy_order_rdc[x][i, j], order_up_to_level_rdc[x][i, j]])
            datas.append([i, j, 'Warehouse', inv_on_hand_cdc[i, j], echelon_inventory_cdc[i, j], policy_order_cdc[i, j], order_up_to_level_cdc[i, j]])

    df_agent_result = pd.DataFrame([pd.Series(row, index=df_agent_result.columns) for row in datas])

    df_result_total = pd.DataFrame(columns=['Item', 'Value'])
    datas = []
    avg_step_cost, se_step_cost, _ = mean_se_median(np.mean(rewards_step, axis=1))
    avg_step_cost = round(avg_step_cost, 3)
    se_step_cost = round(se_step_cost, 3)
    datas.append(['Average cost per period', avg_step_cost])
    datas.append(['SE of average cost per period', se_step_cost])
    for x in rdcs:
        avg_on_hand_inv = round(np.mean(inv_on_hand_rdc[x]), 3)
        datas.append(['Average on hand inventory of retaielr %d'%x, avg_on_hand_inv])
        avg_echelon_inv = round(np.mean(echelon_inventory_rdc[x]), 3)
        datas.append(['Average echelon inventory of retailer %d'%x, avg_echelon_inv])
        avg_order_quantity = round(np.mean(policy_order_rdc[x]), 3)
        datas.append(['Average order quantity of retailer %d'%x, avg_order_quantity])
        avg_order_up_to_level = round(np.mean(order_up_to_level_rdc[x]), 3)
        datas.append(['Average order-up-to level of retailer %d'%x, avg_order_up_to_level])
    avg_on_hand_inv = round(np.mean(inv_on_hand_cdc), 3)
    datas.append(['Average on hand inventory of warehouse', avg_on_hand_inv])
    avg_echelon_inv = round(np.mean(echelon_inventory_cdc), 3)
    datas.append(['Average echelon inventory of warehouse', avg_echelon_inv])
    avg_order_quantity = round(np.mean(policy_order_cdc), 3)
    datas.append(['Average order quantity of warehouse', avg_order_quantity])
    avg_order_up_to_level = round(np.mean(order_up_to_level_cdc), 3)
    datas.append(['Average order-up-to level of warehouse', avg_order_up_to_level])

    df_result_total = pd.DataFrame([pd.Series(row, index=df_result_total.columns) for row in datas])

    with pd.ExcelWriter(f'{OUTPUT_PATH}results_{agent_name}.xlsx') as writer:
        df_agent_result.to_excel(writer, sheet_name='Deteiled results', index=False)
        df_result_total.to_excel(writer, sheet_name='Summary', index=False)


def get_cumulative_rewards_and_steps(env, actor, if_render: bool = False):
    """Usage
    eval_times = 4
    net_dim = 2 ** 7
    actor_path = './LunarLanderContinuous-v2_PPO_1/actor.pt'

    env = build_env(env_class=env_class, env_args=env_args)
    act = agent(net_dim, env.state_dim, env.action_dim, gpu_id=gpu_id).act
    act.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))

    r_s_ary = [get_episode_return_and_step(env, act) for _ in range(eval_times)]
    r_s_ary = np.array(r_s_ary, dtype=np.float32)
    r_avg, s_avg = r_s_ary.mean(axis=0)  # average of episode return and episode step
    """
    env.total_periods = 1000
    max_step = env.total_periods
    calc_step = max_step / 2
    if_discrete = env.if_discrete
    device = next(actor.parameters()).device  # net.parameters() is a Python generator.

    policy_order_rdc = {rdc: np.zeros(calc_step) for rdc in env.rdcs.keys()}
    policy_order_cdc = np.zeros(calc_step)

    inv_on_hand_rdc = {rdc: np.zeros(calc_step) for rdc in env.rdcs.keys()}
    inv_on_hand_cdc = np.zeros(calc_step)

    order_up_to_level_rdc = {r: np.zeros(calc_step) for r in env.rdcs.keys()}
    order_up_to_level_cdc = np.zeros(calc_step)

    echelon_inventory_rdc = {r: np.zeros(calc_step) for r in env.rdcs.keys()}
    echelon_inventory_cdc = np.zeros(calc_step)

    state = env.reset()
    steps = None
    returns = 0.0  # sum of rewards in an episode
    rewards = []
    for steps in range(max_step):
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        tensor_action = actor(tensor_state)
        if if_discrete:
            tensor_action = tensor_action.argmax(dim=1)
        action = tensor_action.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        state, reward, done, _ = env.step(action)

        # record results

        if steps >= calc_step:
            returns += reward
            rewards.append(reward)
            for rdc in env.rdcs.keys():
                policy_order_rdc[rdc][steps - calc_step] = env.RDC_inventory[rdc][-1]
                inv_on_hand_rdc[rdc][steps - calc_step] = env.RDC_inventory[rdc][0]
                order_up_to_level_rdc[rdc][steps - calc_step] = env.target_inventory_position[rdc]
                echelon_inventory_rdc[rdc][steps - calc_step] = env.state_echelon_rdc[rdc][-1]
            policy_order_cdc[steps - calc_step] = env.CDC_inventory[-1]
            inv_on_hand_cdc[steps - calc_step] = env.CDC_inventory[0]
            order_up_to_level_cdc[steps - calc_step] = env.target_inventory_position[0]
            echelon_inventory_cdc[steps - calc_step] = env.state_echelon_cdc[-1]

        if done:
            break
    else:
        print("| get_rewards_and_step: WARNING. max_step > 12345")
    # returns = getattr(env, 'cumulative_returns', returns)
    steps += 1
    return returns, steps, rewards, policy_order_rdc, policy_order_cdc, inv_on_hand_rdc, inv_on_hand_cdc, order_up_to_level_rdc, order_up_to_level_cdc, echelon_inventory_rdc, echelon_inventory_cdc


if __name__ == '__main__':
    # data = generate_certain_data()
    datasets = generate_ratio_data()
    instance_name = 'dataset'
    instance_no = 0
    # arg_name = f'OWMRInventoryEnv_{instance_name}'

    for data in datasets:
        instance_n = f'{instance_name}_{instance_no}'
        cwd = f'./PPO_0625_ratio/{instance_n}'

        rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, RDC_inventory, CDC_inventory = data
        input_env, state_dim, action_dim = env_input_initialize(rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, RDC_inventory, CDC_inventory)
        
        Num_of_Instances_Generated = 100
        learning_rate, net_dims, reward_scale, holding_weight, penalty_weight, cost_plus, all_demand_list = auto_hyperparameter(input_env, Num_of_Instances_Generated)
        input_args = {}
        # input_args['initial_inventory_cdc'] = input_args.pop('CDC_inventory', None)
        # input_args['initial_inventory_rdc'] = input_args.pop('RDC_inventory', None)
        input_args['all_demand_list'] = all_demand_list
        input_args['holding_weight'] = holding_weight
        input_args['penalty_weight'] = penalty_weight
        input_args['cost_plus'] = cost_plus
        input_args['info_input'] = input_env
        input_args['if_generate_demand'] = False

        # generate certain demand
        demand_feature = input_env.demand_feature
        all_demand_list = {k: stats.poisson.rvs(demand_feature[k], size=(Num_of_Instances_Generated, 1000)) for k in demand_feature.keys()}

        agent_class = [AgentPPO.AgentPPO, AgentA2C.AgentA2C, AgentDDPG.AgentDDPG, AgentTD3.AgentTD3, AgentSAC, AgentConvexSAC.AgentConvexSAC][4]
        agent_name = agent_class.__name__[5:]
        # instance_name = 'dataset_2'
        actor_path = f'{cwd}/act.pth'

        evaluate_single_agent(all_demand_list, input_args, agent_class, actor_path, eval_times=100)
