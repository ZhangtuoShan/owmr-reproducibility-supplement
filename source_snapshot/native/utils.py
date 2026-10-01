# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 版本所有 (c) 华为技术有限公司 2024-2024
"""
基础通用函数
"""
import os
from collections import Counter, defaultdict
from copy import deepcopy
from functools import reduce

import networkx as nx
import numpy as np
import numpy.random as rd
import pandas as pd

# from configuration import merge_min_length, merge_max_length, repeted_ratio_high, repeted_ratio_low
from configuration import *
from my_log import logger


def my_mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def remove_cycle_in_bom(bom):
    graph = nx.from_pandas_edgelist(bom, 'partnumber', 'childpartnumber', create_using=nx.DiGraph())  # 构建Bom图
    have_cycle = identify_cycle_in_graph(graph)
    if not have_cycle:
        return bom

    cycle_assembly_list = []
    while True:
        # 找环
        cycle_list = []
        for node in graph.nodes:
            try:
                cycle = nx.find_cycle(G=graph, source=node)
                cycle_list.append(tuple(sorted(cycle)))
            except (Exception,):
                pass
            finally:
                pass
        if not cycle_list:
            break
        unique_cycle_list = []
        for c in set(cycle_list):  # 对环进行去重
            unique_cycle_list.append(list(c))

        # 断bom，基于item排序，选择排序后的第一个节点断bom
        for cycle in unique_cycle_list:
            assembly = cycle[-1][0]
            cycle_assembly_list.append(assembly)
        graph.remove_nodes_from(cycle_assembly_list)
    bom = bom.query('assembly not in @cycle_assembly_list').reset_index(drop=True)
    return bom


def identify_cycle_in_graph(graph):
    have_cycle = True  # 判断是否存在bom环
    # 将图中和环不相关的item删去
    # 前一次遍历图中节点数量
    forward_node_num = -1
    while True:
        # 当前图中节点个数
        current_node_num = len(graph.nodes)
        if current_node_num == forward_node_num:
            # 环以外的节点已经全部删去，退出循环
            break
        else:
            forward_node_num = current_node_num
        remove_nodes = []
        for node in graph.nodes:
            # 剪枝，将环以外的所有节点删去（环内节点的出度、入度>=1）
            if graph.in_degree(node) == 0 or graph.out_degree(node) == 0:
                remove_nodes.append(node)
        graph.remove_nodes_from(remove_nodes)
        if len(graph.nodes) == 0:
            have_cycle = False
            break

    return have_cycle


def adding_lead_qty(bom_df, all_lead_time, step=None):
    '''
    对顶层父项或中间父项采用深度优先的递归方式，找到每个父项到最底层子项的路径(all_path)，用{父项1:[[父项1,父项2,子项],[...]]}的格式存储
    对all_path里的路径取首尾，累乘累加得到累积配比和累积提前期，存储为all_count = {父项1:{(父项1,父项2,子项,):(lt, qty),(...):(lt, qty)}}
    同样的方法得到每个子项到最顶层父项的路径,得到all_path_reverse，存储格式同all_path
    对all_path[item]的路径进行分解去重，得到all_class_count[item]，例如path=['a','b','c'],分解为[['a'],['a','b'],['a','b','c']]
    同样的方法计算all_class_count的累积lt和配比
    其中，对于item对自身有需求，即路径为all_path[item]=[[item,item]],累积提前期和配比为0和1，即all_count[item]={('a','a',):(0,1)}
    '''

    bom = bom_df[['partnumber', 'childpartnumber', 'count']]
    tree_f, tree_s, all_path, all_count = defaultdict(), defaultdict(), defaultdict(), defaultdict()
    all_count, all_class_count, all_class_path = defaultdict(), defaultdict(), defaultdict()
    logger.info(f'bom.shape: %s', bom.shape)

    all_bom_item = set(bom_df['childpartnumber']) | set(bom_df['partnumber'])  # 所有编码

    pure_child = set(bom_df['childpartnumber']) - set(bom_df['partnumber'])  # 纯子项
    items_son = all_bom_item - pure_child
    q_dict = dict(zip(zip(bom_df['partnumber'], bom_df['childpartnumber']), bom_df['count']))  # {(父项, 子项): 配比}
    df = bom.groupby(['partnumber'])
    # for i in df:
    #     tree_f[i[0][0]] = set(i[1]['childpartnumber'])
    for i in df:
        tree_f[i[0]] = set(i[1]['childpartnumber'])

    df1 = bom.groupby(['childpartnumber'])
    # for i in df1:
    #     tree_s[i[0][0]] = set(i[1]['partnumber'])
    for i in df1:
        tree_s[i[0]] = set(i[1]['partnumber'])

    for item in items_son:
        paths = search_path(item, path=[], tree=tree_f)
        all_path[item] = paths
        all_count[item] = {tuple(i): reduce_add(i, q_dict, all_lead_time) for i in paths}

    for item, path_lt_q in all_count.items():
        paths = list(path_lt_q.keys())
        m = []
        for path in paths:
            path = list(path)
            for i in range(len(path)):
                m = m + [path[:i + 1]] if list(path[:i + 1]) not in m else m
            all_class_path[item] = m
        all_class_count[item] = {tuple(n): reduce_add(n, q_dict, all_lead_time) for n in m}

    for item in pure_child:
        all_count[item] = {tuple([item, item]): (0, 1)}
        all_class_count[item] = {tuple([item]): (0, 1), tuple([item, item]): (0, 1)}

    if step:
        return pure_child, tree_s, tree_f, all_class_count, all_bom_item
    else:
        return all_class_count


def search_path(target, path, tree, ends=None):
    t = path.copy()
    f = tree.get(target, None)
    path.append(target)
    if target == ends:
        return [path]
    if f is None:
        return [path]
    paths = []
    for item in f:
        if item not in path:
            pn = search_path(item, path, tree, ends)
            for p in pn:
                paths.append(p)
            path = t.copy()
            path.append(target)
    return paths


def reduce_add(input_list, q_dict, all_lead_time):
    c = ()
    if len(input_list) == 2:
        try:
            qty = q_dict[(input_list[0], input_list[1])]
            lt = all_lead_time[input_list[0]]
            c = (lt, qty)
        except KeyError:
            logger.info('(%s, %s)没有对应bom 或 %s 没有对应all_lead_time', input_list[0], input_list[1], input_list[0])
    else:
        if len(input_list) == 1:
            qty = 1
            lt = 0
            c = (lt, qty)
        else:
            input_list1 = [q_dict.get((input_list[i], input_list[i + 1]), 0) for i in range(len(input_list) - 1)]
            input_list2 = [all_lead_time.get(input_list[i], 0) for i in range(len(input_list) - 1)]
            lt = reduce(lambda x, y: x + y, input_list2)
            qty = reduce(lambda x, y: x * y, input_list1)
            c = (lt, qty)
    return c


def full_gross_demand_data(df_main_demand, all_items, df_main_demand_ny, period):
    net_demand_df = df_main_demand[
        (df_main_demand['item_code'].isin(all_items)) & (df_main_demand['period'] == period)
        & (df_main_demand['demand_type'].isin([8, 2]))].copy()
    net_demand_ny_df = df_main_demand_ny[
        (df_main_demand_ny['item_code'].isin(all_items)) & (df_main_demand_ny['period'] == period)
        & (df_main_demand_ny['demand_type'].isin([8, 2]))].copy()
    net_demand_df = net_demand_df.sort_values(by='demand_type').drop_duplicates(
        ['item_code', 'period', 'bucket_id'], keep='last').copy()
    net_demand_ny_df = net_demand_ny_df.sort_values(by='demand_type').drop_duplicates(
        ['item_code', 'period', 'bucket_id'], keep='last').copy()
    net_demand_df = pd.concat([net_demand_df, net_demand_ny_df])
    demand_df = net_demand_df.groupby(['item_code', 'period', 'bucket_id'])['demand_quantity'].sum().reset_index()
    demand_items = set(demand_df['item_code'])
    return demand_df, demand_items


def bom_cut(st_items, tree_f, decompose_class_demand_ratio_dict):
    """
    得到更小的BOM的分解策略：
    记X为（分部门）全量【有需求】的编码池；
        - 每次从X中取出一个未审视过的的编码m出发，将其加入到待审视编码池Y和当前BOM编码池Z中，重复以下操作，直至Y为空；
            - 同时取出Y中所有编码。找到它们的所有【直接子项】，由【Z中编码与这些子项编码】组成临时BOM，分别对各子项进行判断：若当前
            BOM（所有直接或间接父项）的主需求分解到该编码的毛需求占比超过5%，且该编码自己的主需求对毛需求的占比低于95%，则将其加入到Y和Z中。
        - 此时得到从m出发的分解掉部分子项的BOM。若Z中有编码在X中，则将这些编码也从X中取出。记录该BOM中编码集合，并清空Z。
    - 当X中所有编码审视完后（X为空），最终得到各分解后独立的BOM。
    """
    cut_bom = []
    all_used_items = list(st_items)
    while all_used_items:
        i = all_used_items[0]
        z, re = [i], [i]
        while re:
            son = set(sum([list(tree_f.get(i, set())) for i in re], []))
            bom_items = set(z + list(son))
            re = []
            for item in bom_items:
                a = decompose_class_demand_ratio_dict.get(item, dict()).get(item, 0)
                b = decompose_class_demand_ratio_dict.get(item, dict())
                c = sum([v for k, v in b.items() if (k in bom_items) and (k != item)])
                if (a < repeted_ratio_high) and (c > repeted_ratio_low) and (item not in z):
                    re.append(item)
            z = list(set(z + re))
        all_used_items = [item_list for item_list in all_used_items if item_list not in z]

        if len(cut_bom) == 0:
            cut_bom.append(z)
        else:
            t = True  # T：是否添加z
            for i in cut_bom:
                if set(z).issubset(i):
                    t = False
                elif set(i).issubset(z):
                    cut_bom.remove(i)
                else:
                    pass
            if t:
                cut_bom.append(z)
    return cut_bom


def duplicated_bom(cut_bom):
    all_bom, remove_bom, add_dept_bom = [], [], defaultdict(list)
    # 全量去重以及包含关系，创建all_bom：一个全部门子bom的list存储已经审视的子bom
    for dept, bom in cut_bom.items():
        for z in bom:
            if len(all_bom) == 0:
                all_bom.append(z)
            else:
                t = True  # t：是否添加z
                for i in all_bom:
                    if set(z) < set(i):  # z是i的子集，不添加z进入all_bom,且添加z进入删除bom列表remove_bom
                        t = False
                        remove_bom.append(z)
                    elif set(i) < set(z):  # i是z的子集，添加z进入all_bom,在all_bom中删除i,且添加i进入删除bom列表remove_bom
                        all_bom.remove(i)
                        remove_bom.append(i)
                    elif set(i) == set(z):  # 若i==z，则只保留一个
                        t = False
                        if i in remove_bom:
                            pass
                        else:
                            remove_bom.append(i)
                            add_dept_bom[dept].append(z)
                    else:
                        pass
                if t:
                    all_bom.append(z)
    for dept, bom in cut_bom.items():
        for i in remove_bom:
            for j in bom:
                if set(i) == set(j):
                    cut_bom[dept].remove(j)

    for dept, item_list in add_dept_bom.items():
        cut_bom[dept].append(item_list)
    return cut_bom


def find_one_contact(bom_contact, none_contact_bom, short_contact_bom):
    bom_qq, bom_lap = high_contact_ratio(bom_contact)  # 有关联的子bom数量 重合率（>0)
    while len(bom_contact) > 1:
        bom_q = bom_qq.copy()
        bom_q = zip(bom_q.values(), bom_q.keys())  # 将子bom按照有关联子bom数量，从小到大排列
        bom_q = sorted(bom_q)
        none_contact = [list(i[1]) for i in bom_q if i[0] == 0]  # 与其他子bom毫无关联的部分
        none_contact_bom = none_contact_bom + none_contact
        bom_contact = [list(i[1]) for i in bom_q if i[0] > 0]
        bom_q = [i for i in bom_q if i[0] > 0]
        bom_qq = {bom: n for bom, n in bom_qq.items() if list(bom) in bom_contact}
        if bom_q:
            min_ctt = min([i[0] for i in bom_q])
            # 和其他子bom关联最少的那部分子bom
            min_contact = [i[1] for i in bom_q if i[0] == min_ctt]
            bom = list(min_contact[0])
            # 删掉该子bom，并找出与该子bom重合最多的子bom
            bom_contact.remove(bom)
            max_bom = [list(i[0]) for i in sorted(bom_lap[tuple(bom)].items(), key=lambda item: item[1], reverse=True)]
            con_bom = [list(i) for i in bom_lap[tuple(bom)].keys()]
            # 该子bom如果和其他所有子bom两两合并后，长度大于50，则不考虑该子bom,放入short_contact_bom
            if all(len(list(set(bom).union(set(bom1)))) >= merge_max_length for bom1 in max_bom):
                short_contact_bom.append(bom)
                # 删除bom信息
                bom_lap.pop(tuple(bom))
                bom_qq.pop(tuple(bom))
                # 删除与bom有关联的bom1之间的bom_lap重复度信息，更新和bom1有关联的数量
                for bom1 in con_bom:
                    del bom_lap[tuple(bom1)][tuple(bom)]
                    bom_qq[tuple(bom1)] = len([n for n in bom_lap[tuple(bom1)].keys()])
            else:
                # 若存在合并后长度符合条件的bom1
                for bom1 in max_bom:
                    new_bom = list(set(bom).union(set(bom1)))
                    if len(new_bom) < merge_max_length:
                        # 删除和bom1,bom有关的bom_lap,bom_q信息，将new_bom的信息更新进入bom_lap和bom_q
                        bom_qq, bom_lap = remove_lap_contact(bom_qq, bom_lap, bom_contact, con_bom, new_bom, bom, bom1)
                        break
                    else:
                        continue
    return none_contact_bom, short_contact_bom, bom_contact


def high_contact_ratio(bom):
    # ratio表示对于每个子bom，与其有交集的其他子bom数量(>=0)
    # lap表示对于每个子bom,与其有交集的其他子bom重合度(只录入>0部分)
    ratio, lap = dict(), defaultdict(dict)
    if len(bom) > 1:
        for i in bom:
            bom1 = bom.copy()
            bom1.remove(i)
            re = []
            for j in bom1:
                a = contact_ratio(i, j)  # j对i的重合率，例如i=[a,b],j=[b,c,d],此时j对i的重合率是0.5，但i对j的重合率是1/3
                if a > 0:
                    lap[(tuple(i))][tuple(j)] = a
                re.append(a)
            re1 = [i for i in re if i > 0]
            ratio[tuple(i)] = len(re1)
    return ratio, lap


def contact_ratio(list1, list2):
    counter1 = Counter(list1)
    counter2 = Counter(list2)
    intersection = counter1 & counter2
    overlap = sum(intersection.values())
    total = sum(counter1.values())
    similarity = overlap / total
    return similarity


def one_contact_ratio(bom, new_bom):
    bom1 = bom.copy()
    bom1.remove(new_bom)
    re, lap = [], dict()
    for i in bom1:
        a = contact_ratio(i, new_bom)
        re.append(a)
        if a > 0:
            lap[tuple(i)] = a
    return max(re, default=0), lap


def remove_lap_contact(bom_qq, bom_lap, bom_contact, con_bom, new_bom, bom, bom1):
    # 获取和bom, bom1的关联子bom
    con_bom1 = [list(i) for i in bom_lap[tuple(bom1)].keys()]
    con_bom.remove(bom1)
    con_bom1.remove(bom)
    bom_contact.remove(bom1)
    bom_contact.append(new_bom)
    # 去掉旧的
    bom_lap.pop(tuple(bom))
    bom_lap.pop(tuple(bom1))
    bom_qq.pop(tuple(bom))
    bom_qq.pop(tuple(bom1))

    # 增改新的，添加new_bom的关联信息
    m, q = one_contact_ratio(bom_contact, new_bom)
    if q:
        bom_lap[tuple(new_bom)] = q
    bom_qq[tuple(new_bom)] = m

    # 将原bom和bom1关系删除，添加其他子bom和new_bom的关联.
    bom_qq, bom_lap = drop_contact(con_bom, bom_lap, bom_qq, bom, new_bom)
    bom_qq, bom_lap = drop_contact(con_bom1, bom_lap, bom_qq, bom1, new_bom)

    return bom_qq, bom_lap


def merge_repetitive_bom(cut_bom):
    cut_union_bom = dict()
    for dept, bom in cut_bom.items():
        # 只合并长度在某个范围内的子bom，小于最低范围的不合并，长度大于最高的不合并
        bom1, cut_bom_shorter, cut_bom_longer = [], [], []
        for i in bom:
            if len(i) <= merge_min_length:
                cut_bom_shorter.append(i)
            elif len(i) >= merge_max_length:
                cut_bom_longer.append(i)
            else:
                bom1.append(i)
        none_contact_bom, short_contact_bom, bom_contact = find_one_contact(bom1, [], [])
        bom_merged = none_contact_bom + short_contact_bom + bom_contact
        all_bom = bom_merged + cut_bom_shorter + cut_bom_longer
        cut_union_bom[dept] = all_bom
    return cut_union_bom


def drop_contact(contact_bom, bom_lap, bom_qq, old_bom, new_bom):
    for i in contact_bom:
        del bom_lap[tuple(i)][tuple(old_bom)]
        if contact_ratio(i, new_bom) > 0:
            bom_lap[tuple(i)][tuple(new_bom)] = contact_ratio(i, new_bom)
        bom_qq[tuple(i)] = len(bom_lap[tuple(i)])
    return bom_qq, bom_lap


# reshape demand_dicts
def round_list(data: list):
    return [round(x, 3) for x in data]


def process_feature(data: dict):
    for i in range(len(data['0.1'])):
        for j in range(1, 9):
            if data['0.%d' % j][i] > data['0.%d' % (j + 1)][i]:
                data['0.%d' % j][i] = data['0.%d' % (j + 1)][i]
    # 补齐13周后的预测
    if len(data['0.1']) < MAX_WEEK:
        for i in range(1, 10):
            data['0.%d' % i].extend([data['0.%d' % i][-1]] * (MAX_WEEK - len(data['0.%d' % i])))
    return data


def generate_demand(rdc, demand_feature, period):
    if not demand_feature[rdc]:
        return 0.
    c = rd.rand()
    cr = int(c * 10 // 1)
    if c > 0.9:
        return demand_feature[rdc]['0.9'][period]
    if c < 0.1:
        return c * 10 * demand_feature[rdc]['0.1'][period]

    return demand_feature[rdc]['0.%d' % cr][period] + (c - cr / 10) * 10 * (
            demand_feature[rdc]['0.%d' % (cr + 1)][period] -
            demand_feature[rdc]['0.%d' % cr][period])


class inventory_env_input_args:
    def __init__(self, key, selling_item, rdc_dict, item_dict, RDC_inventory, CDC_inventory, demand_feature, state_dim,
                 action_dim, demand_flag=True) -> None:
        self.key = key
        self.selling_item = selling_item
        self.rdc_dict = rdc_dict
        self.item_dict = item_dict
        self.RDC_inventory = RDC_inventory
        self.CDC_inventory = CDC_inventory
        self.demand_feature = demand_feature
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.demand_flag = demand_flag


def env_input_renew(input_args: inventory_env_input_args, all_RDC_inventory, all_CDC_inventory, demand_agg):
    items = list(input_args.CDC_inventory.keys())

    RDC_inventory = {k: deepcopy(v) for k, v in all_RDC_inventory.items() if k in input_args.selling_item}
    CDC_inventory = {k: deepcopy(v) for k, v in all_CDC_inventory.items() if k in items}
    RDC_inventory = process_on_hand_inventory(RDC_inventory)
    CDC_inventory = process_on_hand_inventory(CDC_inventory)
    for k, v in CDC_inventory.items():
        if len(v) < 2:
            CDC_inventory[k].append(0.)

    non_exist_selling_item = list(set(input_args.selling_item) - set(list(RDC_inventory.keys())))
    non_exist_item = list(set(items) - set(list(CDC_inventory.keys())))
    if non_exist_selling_item:
        logger.info(f'Attention: items {non_exist_selling_item} on sale have no initial RDC inventory')
        for x in non_exist_selling_item:
            RDC_inventory[x] = {}
            for r in rdc_dict:
                RDC_inventory[x][r] = np.zeros(input_args.rdc_dict[r] + 1)
    if non_exist_item:
        logger.info(f'Attention: items {non_exist_item} in current BOM have no initial CDC inventory')
        for x in non_exist_item:
            CDC_inventory[x] = np.zeros(int(input_args.item_dict[x]['lt']) + 1)

    reshaped_demand = {x: {y: {} for y in rdc_dict.keys()} for x in input_args.selling_item}
    for x in demand_agg:
        if x['item_code'] in input_args.selling_item:
            demand_fea = {'0.%s' % i: round_list(x['0.%s' % i]) for i in range(1, 10)}
            demand_fea = process_feature(demand_fea)
            if x['rdc'] == "深圳/东莞":
                x['rdc'] = '深圳'
            reshaped_demand[x['item_code']][x['rdc']] = demand_fea
    demand_feature = reshaped_demand.copy()

    input_args.CDC_inventory = deepcopy(CDC_inventory)
    input_args.RDC_inventory = deepcopy(RDC_inventory)
    input_args.demand_feature = deepcopy(demand_feature)
    return input_args


def env_input_initialize(key, item_dict, item_df, all_demand_items, all_RDC_inventory, all_CDC_inventory, demand_agg):
    items = list(item_df['item_code'].unique())
    selling_item = [x for x in items if x in all_demand_items]
    demand_flag = True
    if not selling_item:
        demand_flag = False

    # 删去无预测需求的顶层编码
    remaining_items = list(item_dict.keys())
    if_revise = True
    while if_revise:
        if_revise = False
        for k in remaining_items:
            if_root = all([k not in v1['children'].keys() for k1, v1 in item_dict.items() if k1 in remaining_items])
            if if_root and k not in selling_item:
                remaining_items.remove(k)
                if_revise = True
    item_dict = {k: v for k, v in item_dict.items() if k in remaining_items}
    for k, v in item_dict.items():
        v['lt'] = 1 if v['lt'] < 1 else v['lt']
    items = remaining_items

    RDC_inventory = {k: deepcopy(v) for k, v in all_RDC_inventory.items() if k in selling_item}
    CDC_inventory = {k: deepcopy(v) for k, v in all_CDC_inventory.items() if k in items}
    RDC_inventory = process_on_hand_inventory(RDC_inventory)
    CDC_inventory = process_on_hand_inventory(CDC_inventory)
    for k, v in CDC_inventory.items():
        if len(v) < 2:
            CDC_inventory[k].append(0.)

    non_exist_selling_item = list(set(selling_item) - set(list(RDC_inventory.keys())))
    non_exist_item = list(set(items) - set(list(CDC_inventory.keys())))
    if non_exist_selling_item:
        logger.info(f'Attention: items {non_exist_selling_item} on sale have no initial RDC inventory')
        for x in non_exist_selling_item:
            RDC_inventory[x] = {}
            for r in rdc_dict:
                RDC_inventory[x][r] = np.zeros(rdc_dict[r] + 1)
    if non_exist_item:
        logger.info(f'Attention: items {non_exist_item} in current BOM have no initial CDC inventory')
        for x in non_exist_item:
            CDC_inventory[x] = np.zeros(int(item_dict[x]['lt']) + 1)

    reshaped_demand = {x: {y: {} for y in rdc_dict.keys()} for x in selling_item}
    for x in demand_agg:
        if x['item_code'] in selling_item:
            demand_fea = {'0.%s' % i: round_list(x['0.%s' % i]) for i in range(1, 10)}
            demand_fea = process_feature(demand_fea)
            if x['rdc'] == "深圳/东莞":
                x['rdc'] = '深圳'
            reshaped_demand[x['item_code']][x['rdc']] = demand_fea
    demand_feature = reshaped_demand.copy()

    state_dim = len(selling_item) * sum([week_rdc_dict[k] + 1 for k in rdc_dict.keys()]) + sum(
        [math.ceil(int(item_dict[k]['lt']) / 7) + 1 for k in item_dict.keys()])
    # state_dim += 1
    action_dim = len(selling_item) * len(week_rdc_dict) + len(item_dict)

    return inventory_env_input_args(key, selling_item, week_rdc_dict, item_dict, RDC_inventory, CDC_inventory,
                                    demand_feature, state_dim, action_dim, demand_flag)


def process_on_hand_inventory(input):
    if isinstance(input, dict):
        return {k: process_on_hand_inventory(v) for k, v in input.items()}
    if isinstance(input, list) or isinstance(input, np.ndarray):
        if input[0] < 0:
            input[0] = 0.
        return input
    raise TypeError
