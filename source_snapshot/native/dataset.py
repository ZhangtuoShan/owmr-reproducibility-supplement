import numpy as np
import numpy.random as rd
from collections import defaultdict
import json
from pathlib import Path

from demand_utils import (
    demand_mean,
    poisson_tail_lump_cap,
    quantile_series_lead_time_pmf,
    quantile_series_sum_window_pmf,
)

rdc_dict = {
            1: {1: 1, 2: 1},
            2: {1: 1, 2: 3},
            3: {1: 3, 2: 1},
            # 4: {1: 2, 2: 2, 3: 2}, 
            # 5: {1: 2, 2: 3, 3: 4},
            # 6: {1: 2, 2: 4, 3: 3},
            # 7: {1: 3, 2: 2, 3: 4},
            # 8: {1: 3, 2: 4, 3: 2},
            # 9: {1: 4, 2: 3, 3: 2},
            # 10: {1: 4, 2: 2, 3: 3},
            }
holding_cost = {
                1: {0: 1, 1: 1, 2: 1},
                2: {0: 1, 1: 2, 2: 2},
                3: {0: 1, 1: 1, 2: 2},
                4: {0: 1, 1: 2, 2: 1},
                5: {0: 1, 1: 2, 2: 4, 3: 6},
                6: {0: 1, 1: 2, 2: 6, 3: 4},
                7: {0: 1, 1: 4, 2: 2, 3: 6},
                8: {0: 1, 1: 4, 2: 6, 3: 2},
                9: {0: 1, 1: 6, 2: 2, 3: 4},
                10: {0: 1, 1: 6, 2: 4, 3: 2},
                }
backorder_cost = {
                  2: {1: 10, 2: 10},
                  3: {1: 10, 2: 10, 3: 10}}
demand_feature = {
                  1: {1: 10, 2: 10},
                  2: {1: 10, 2: 40},
                  3: {1: 40, 2: 10},
                  4: {1: 10, 2: 10, 3: 10},
                  5: {1: 10, 2: 20, 3: 40},
                  6: {1: 10, 2: 40, 3: 20},
                  7: {1: 20, 2: 10, 3: 40},
                  8: {1: 20, 2: 40, 3: 10},
                  9: {1: 40, 2: 10, 3: 20},
                  10: {1: 40, 2: 20, 3: 10},
                  }
item_lead_time = {1: 1}
RDC_initial_inventory = {1: 20, 2: 20, 3: 60}
CDC_initial_inventory = 50
# all_RDC_inventory = {2: {1: np.array([20, 0, 0]), 
#                          2: np.array([40, 0, 0, 0])},
#                      3: {1: np.array([20, 0, 0]), 
#                          2: np.array([40, 0, 0, 0]),
#                          3: np.array([60, 0, 0, 0])},
#                     }
# all_CDC_inventory = np.array([50, 0, 0])

def data_kaynov():
    data_sets = []
    backorder_proportion = 1.0
    item_lead_time = 2

    holding_cost = {0: 0.5, 1: 1, 2: 1, 3: 1}
    backorder_cost = {1: 9, 2: 9, 3: 9}

    RDC_initial_inventory = {1: 3, 2: 3, 3: 3}
    CDC_initial_inventory = 0

    def build_initial_inventory(rdc_lead_times, cdc_lead_time):
        rdc_inventory = {}
        for rdc in rdc_lead_times.keys():
            rdc_inventory[rdc] = np.hstack((RDC_initial_inventory[rdc], np.zeros(rdc_lead_times[rdc])))
        cdc_inventory = np.hstack((CDC_initial_inventory, np.zeros(cdc_lead_time)))
        return rdc_inventory, cdc_inventory

    rdc_dict = {1: 1, 2: 1, 3: 1}
    demand_feature = {1: 3, 2: 3, 3: 3}
    all_RDC_inventory, all_CDC_inventory = build_initial_inventory(rdc_dict, item_lead_time)

    data_sets.append((rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, all_RDC_inventory, all_CDC_inventory))

    demand_feature = {1: ('uniform', 0, 6), 2: ('normal', 3, 1), 3: ('poisson', 3)}
    all_RDC_inventory, all_CDC_inventory = build_initial_inventory(rdc_dict, item_lead_time)
    data_sets.append((rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, all_RDC_inventory, all_CDC_inventory))

    demand_feature = {1: ('normal', 1, 25), 2: ('normal', 5, 1), 3: ('poisson', 0.5)}
    all_RDC_inventory, all_CDC_inventory = build_initial_inventory(rdc_dict, item_lead_time)
    data_sets.append((rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, all_RDC_inventory, all_CDC_inventory))

    demand_feature = {1: 3, 2: 3, 3: 3}
    rdc_dict = {1: 1, 2: 2, 3: 3}
    all_RDC_inventory, all_CDC_inventory = build_initial_inventory(rdc_dict, item_lead_time)
    data_sets.append((rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, all_RDC_inventory, all_CDC_inventory))

    item_lead_time = 5
    demand_feature = {1: ('normal', 1, 25), 2: ('normal', 5, 1), 3: ('poisson', 0.5)}
    rdc_dict = {1: 3, 2: 3, 3: 3}
    all_RDC_inventory, all_CDC_inventory = build_initial_inventory(rdc_dict, item_lead_time)
    data_sets.append((rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, all_RDC_inventory, all_CDC_inventory))


    item_lead_time = 2
    rdc_dict = {i: 2 for i in range(1, 11)}
    holding_cost = {i: 3 for i in range(11)}
    backorder_cost = {i: 60 for i in range(1, 11)}
    RDC_initial_inventory = {i: 3 for i in range(1, 11)}
    demand_feature = {i: ('normal', 5, 14*14) for i in range(1, 11)}

    all_RDC_inventory, all_CDC_inventory = build_initial_inventory(rdc_dict, item_lead_time)
    data_sets.append((rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, all_RDC_inventory, all_CDC_inventory))

    demand_feature = {1: ('normal', 0, 400), 2: ('normal', 2, 16*16), 3: ('normal', 4, 12*12), 4: ('normal', 6, 64), 5: ('normal', 8, 16), 6: ('normal', 10, 0), 7: ('poisson', 0.5), 8: ('poisson', 3), 9: ('poisson', 9), 10: ('poisson', 12)}
    all_RDC_inventory, all_CDC_inventory = build_initial_inventory(rdc_dict, item_lead_time)
    data_sets.append((rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, all_RDC_inventory, all_CDC_inventory))

    return data_sets


def data_small_rvi():
    """Small two-retailer instances for finite-MDP/RVI diagnostics.

    Demand uses tail-lumped truncated Poisson distributions:
    ('truncated_poisson', lambda, max_demand) represents
    min(Poisson(lambda), max_demand), so all tail probability above max_demand
    is assigned to max_demand.
    """

    def build_initial_inventory(rdc_lead_times, cdc_lead_time, rdc_initial=0, cdc_initial=0):
        rdc_inventory = {}
        for rdc, lead_time in rdc_lead_times.items():
            rdc_inventory[rdc] = np.hstack((float(rdc_initial), np.zeros(int(lead_time))))
        cdc_inventory = np.hstack((float(cdc_initial), np.zeros(int(cdc_lead_time))))
        return rdc_inventory, cdc_inventory

    data_sets = []

    # S1: symmetric low-demand instance.
    rdc_dict = {1: 1, 2: 1}
    demand_feature = {
        1: ("truncated_poisson", 1, 4),
        2: ("truncated_poisson", 1, 4),
    }
    holding_cost = {0: 1, 1: 2, 2: 2}
    backorder_cost = {1: 5, 2: 5}
    item_lead_time = 1
    all_RDC_inventory, all_CDC_inventory = build_initial_inventory(rdc_dict, item_lead_time)
    data_sets.append((
        rdc_dict,
        demand_feature,
        holding_cost,
        backorder_cost,
        item_lead_time,
        all_RDC_inventory,
        all_CDC_inventory,
    ))

    # S2: heterogeneous lead-time and demand instance.
    rdc_dict = {1: 1, 2: 2}
    demand_feature = {
        1: ("truncated_poisson", 1, 4),
        2: ("truncated_poisson", 2, 6),
    }
    holding_cost = {0: 1, 1: 2, 2: 3}
    backorder_cost = {1: 8, 2: 4}
    item_lead_time = 1
    all_RDC_inventory, all_CDC_inventory = build_initial_inventory(rdc_dict, item_lead_time)
    data_sets.append((
        rdc_dict,
        demand_feature,
        holding_cost,
        backorder_cost,
        item_lead_time,
        all_RDC_inventory,
        all_CDC_inventory,
    ))

    return data_sets


def data_small_diagnostic_instances():
    return data_small_rvi()


def _build_table_case(rdc_lead_times, demand_feature, holding_cost, backorder_cost,
                      cdc_lead_time, rdc_initial_inventory, cdc_echelon_level):
    """Build one table benchmark case from echelon-level initial data.

    The table reports y0^L as the warehouse echelon level. In this codebase the
    CDC initial inventory array stores physical CDC on-hand inventory followed
    by its pipeline, while RDC arrays store physical RDC on-hand plus pipeline.
    With zero initial pipelines, CDC on-hand is therefore y0^L - sum_n x_{n,0}.
    """
    rdc_inventory = {}
    for rdc, lead_time in rdc_lead_times.items():
        rdc_inventory[rdc] = np.hstack((
            float(rdc_initial_inventory[rdc]),
            np.zeros(int(lead_time), dtype=float),
        ))

    cdc_on_hand = float(cdc_echelon_level) - sum(float(v) for v in rdc_initial_inventory.values())
    if cdc_on_hand < -1e-9:
        raise ValueError(
            f"Invalid table case: CDC echelon level {cdc_echelon_level} is below "
            f"sum of RDC initial inventories {sum(rdc_initial_inventory.values())}."
        )
    cdc_inventory = np.hstack((max(0.0, cdc_on_hand), np.zeros(int(cdc_lead_time), dtype=float)))
    return (
        rdc_lead_times.copy(),
        demand_feature.copy(),
        holding_cost.copy(),
        backorder_cost.copy(),
        int(cdc_lead_time),
        rdc_inventory,
        cdc_inventory,
    )


def data_random_table_benchmarks():
    """Fifteen table benchmark cases: 2-1..2-5, 6-1..6-5, 20-1..20-5.

    The first five 2-RDC cases are encoded from the table. Their CDC lead times
    were generated once from the same integer range used by
    generate_data_with_random_para(), using numpy.random seed 10, except 2-1
    which is fixed to CDC lead time 1:
        2-1: 1, 2-2: 2, 2-3: 1, 2-4: 4, 2-5: 1.

    Note: the table row 2-5 lists x0=(60, 263) with y0^L=188, which would imply
    negative CDC on-hand inventory. It is encoded as x0=(60, 63), the only
    one-digit correction that keeps y0^L consistent with a nonnegative CDC
    on-hand level. The last ten cases use the actual matched generated cases
    identified from generate_data_with_random_para(..., random_seed=10).
    """
    data_sets = []
    backorder_cost_2 = {1: 10, 2: 10}
    cdc_lead_times_2 = [1, 2, 1, 4, 1]

    table_2_cases = [
        (
            {1: 1, 2: 1},
            {1: 10, 2: 10},
            {0: 1, 1: 2.00, 2: 2.00},
            {1: 20, 2: 20},
            60,
        ),
        (
            {1: 1, 2: 2},
            {1: 23, 2: 15},
            {0: 1, 1: 3.55, 2: 1.59},
            {1: 23, 2: 45},
            99,
        ),
        (
            {1: 1, 2: 4},
            {1: 28, 2: 10},
            {0: 1, 1: 1.81, 2: 3.87},
            {1: 90, 2: 81},
            235,
        ),
        (
            {1: 2, 2: 3},
            {1: 17, 2: 21},
            {0: 1, 1: 2.10, 2: 3.59},
            {1: 28, 2: 33},
            104,
        ),
        (
            {1: 3, 2: 4},
            {1: 38, 2: 14},
            {0: 1, 1: 2.93, 2: 2.83},
            {1: 60, 2: 63},
            188,
        ),
    ]

    for cdc_lead_time, (rdc_lead_times, demand, holding, rdc_initial, cdc_echelon) in zip(
            cdc_lead_times_2, table_2_cases
    ):
        data_sets.append(_build_table_case(
            rdc_lead_times,
            demand,
            holding,
            backorder_cost_2,
            cdc_lead_time,
            rdc_initial,
            cdc_echelon,
        ))

    random_6 = generate_data_with_random_para(rdc_num=6, sample_num=10, random_seed=10)
    random_20 = generate_data_with_random_para(rdc_num=20, sample_num=10, random_seed=10)

    data_sets.extend(random_6[i] for i in [0, 1, 2, 3, 4])
    data_sets.extend(random_20[i] for i in [0, 2, 3, 4, 5])

    return data_sets


def data_random_table_benchmark_metadata():
    """Metadata aligned with data_random_table_benchmarks()."""
    return [
        {"case": "2-1", "source": "table", "cdc_lead_time": 1},
        {"case": "2-2", "source": "table", "cdc_lead_time": 2},
        {"case": "2-3", "source": "table", "cdc_lead_time": 1},
        {"case": "2-4", "source": "table", "cdc_lead_time": 4},
        {"case": "2-5", "source": "table_with_x0_2_typo_corrected_to_63", "cdc_lead_time": 1},
        {"case": "6-1", "source": "generate_data_with_random_para(rdc_num=6, random_seed=10)[0]", "cdc_lead_time": 2},
        {"case": "6-2", "source": "generate_data_with_random_para(rdc_num=6, random_seed=10)[1]", "cdc_lead_time": 2},
        {"case": "6-3", "source": "generate_data_with_random_para(rdc_num=6, random_seed=10)[2]", "cdc_lead_time": 1},
        {"case": "6-4", "source": "generate_data_with_random_para(rdc_num=6, random_seed=10)[3]", "cdc_lead_time": 3},
        {"case": "6-5", "source": "generate_data_with_random_para(rdc_num=6, random_seed=10)[4]", "cdc_lead_time": 2},
        {"case": "20-1", "source": "generate_data_with_random_para(rdc_num=20, random_seed=10)[0]", "cdc_lead_time": 2},
        {"case": "20-2", "source": "generate_data_with_random_para(rdc_num=20, random_seed=10)[2]", "cdc_lead_time": 1},
        {"case": "20-3", "source": "generate_data_with_random_para(rdc_num=20, random_seed=10)[3]", "cdc_lead_time": 4},
        {"case": "20-4", "source": "generate_data_with_random_para(rdc_num=20, random_seed=10)[4]", "cdc_lead_time": 1},
        {"case": "20-5", "source": "generate_data_with_random_para(rdc_num=20, random_seed=10)[5]", "cdc_lead_time": 1},
    ]


NONSTATIONARY_RDC_NAME_TO_ID = {
    '深圳/东莞': 1,
    '北京': 2,
    '上海': 3,
    '成都': 4,
}
NONSTATIONARY_RDC_LEAD_TIMES = {
    1: 1,
    2: 2,
    3: 2,
    4: 2,
}

# Follow-up cases selected to complement the original low-demand (50102914)
# and medium, declining-demand (98011894) finite-horizon experiments.
NONSTATIONARY_REPRESENTATIVE_FOLLOWUP_ITEM_CODES = (
    '02351RFH-005',  # upward trend with demand concentrated at one RDC
    '98011941',      # highly time-varying demand with an early peak
    '02352QGG-004',  # high-volume, comparatively balanced four-RDC demand
)


def load_weekly_quantile_forecast(path='input_data/data/week pred_国内rdc_20240325.json'):
    """Load the double-encoded weekly quantile forecast JSON."""
    path = Path(path)
    raw = json.load(path.open('r', encoding='utf-8'))
    return json.loads(raw) if isinstance(raw, str) else raw


def _forecast_record_to_quantile_series(record, integerize='round'):
    quantiles = {f'{i / 10:.1f}': [float(x) for x in record[f'{i / 10:.1f}']] for i in range(1, 10)}
    max_q90 = max(quantiles['0.9']) if quantiles['0.9'] else 0.0
    return {
        'type': 'quantile_series',
        'dates': list(record['dates']),
        'quantiles': quantiles,
        'integerize': integerize,
        'tail_policy': 'lower_linear_to_zero_upper_lump',
        'max_demand': int(max(1, np.ceil(max_q90))),
        'item_code': record.get('item_code'),
        'rdc': record.get('rdc'),
    }


def _pmf_quantile(support, pmf, q):
    idx = int(np.searchsorted(np.cumsum(pmf), float(np.clip(q, 0.0, 1.0)), side='left'))
    idx = min(max(idx, 0), len(support) - 1)
    return float(support[idx])


def select_nonstationary_sku_candidates(
        path='input_data/data/week pred_国内rdc_20240325.json',
        num_candidates=3,
        required_rdcs=('深圳/东莞', '北京', '上海', '成都'),
):
    """Pick representative 4-RDC SKUs with low/medium/high but tractable demand."""
    records = load_weekly_quantile_forecast(path)
    grouped = defaultdict(list)
    required_rdcs = tuple(required_rdcs)
    for record in records:
        grouped[record['item_code']].append(record)

    summaries = []
    required_set = set(required_rdcs)
    for item_code, item_records in grouped.items():
        by_rdc = {record['rdc']: record for record in item_records}
        if not required_set.issubset(by_rdc):
            continue
        selected = [by_rdc[rdc] for rdc in required_rdcs]
        avg_q50 = sum(sum(record['0.5']) for record in selected) / 13
        avg_q90 = sum(sum(record['0.9']) for record in selected) / 13
        avg_mean = sum(sum(record['ls']) for record in selected) / 13
        max_q90 = max(max(record['0.9']) for record in selected)
        zero_q50_share = sum(
            1 for record in selected for value in record['0.5'] if float(value) <= 1e-12
        ) / (13 * len(selected))
        summaries.append({
            'item_code': item_code,
            'avg_q50': float(avg_q50),
            'avg_q90': float(avg_q90),
            'avg_mean': float(avg_mean),
            'max_q90': float(max_q90),
            'zero_q50_share': float(zero_q50_share),
        })

    if not summaries:
        return []
    q50_values = np.asarray([row['avg_q50'] for row in summaries], dtype=float)
    targets = np.quantile(q50_values, np.linspace(0.25, 0.75, int(num_candidates)))
    candidates = []
    used = set()
    for target in targets:
        ranked = sorted(
            summaries,
            key=lambda row: (
                abs(row['avg_q50'] - target)
                + 0.01 * row['max_q90']
                + 5.0 * row['zero_q50_share']
            ),
        )
        for row in ranked:
            if row['item_code'] not in used:
                candidates.append(row)
                used.add(row['item_code'])
                break
    return candidates


def data_nonstationary_quantile_skus(
        item_codes=None,
        candidate_count=3,
        path='input_data/data/week pred_国内rdc_20240325.json',
        holding_cost_unit=1,
        backorder_cost_unit=10,
        cdc_lead_time=2,
        integerize='round',
        initial_inventory_quantile=None,
):
    """Build finite-horizon nonstationary OWMR datasets from weekly forecasts.

    Initial inventory follows a cold-start protection protocol. Each RDC holds
    enough inventory for demand before its first regular replenishment can
    arrive. The CDC then protects the aligned upstream blackout window that
    starts after each RDC's local protection window. Pipelines start empty, so
    the same demand is not protected once at the RDC and again at the CDC.
    """
    records = load_weekly_quantile_forecast(path)
    grouped = defaultdict(dict)
    for record in records:
        grouped[record['item_code']][record['rdc']] = record

    required_rdcs = tuple(NONSTATIONARY_RDC_NAME_TO_ID.keys())
    if item_codes is None:
        item_codes = [
            row['item_code']
            for row in select_nonstationary_sku_candidates(
                path=path,
                num_candidates=candidate_count,
                required_rdcs=required_rdcs,
            )
        ]

    data_sets = []
    metadata = []
    rdc_dict = NONSTATIONARY_RDC_LEAD_TIMES.copy()
    holding_cost = {0: float(holding_cost_unit)}
    holding_cost.update({rdc_id: float(holding_cost_unit) for rdc_id in rdc_dict.keys()})
    backorder_cost = {rdc_id: float(backorder_cost_unit) for rdc_id in rdc_dict.keys()}

    for item_code in item_codes:
        by_rdc = grouped[item_code]
        missing = [rdc for rdc in required_rdcs if rdc not in by_rdc]
        if missing:
            raise ValueError(f'item_code={item_code} is missing RDC forecasts: {missing}')

        demand_feature = {}
        for rdc_name, rdc_id in NONSTATIONARY_RDC_NAME_TO_ID.items():
            demand_feature[rdc_id] = _forecast_record_to_quantile_series(by_rdc[rdc_name], integerize=integerize)

        horizons = {
            len(spec['quantiles']['0.1']) for spec in demand_feature.values()
        }
        date_sequences = {
            tuple(spec['dates']) for spec in demand_feature.values()
        }
        if len(horizons) != 1 or len(date_sequences) != 1:
            raise ValueError(
                f'item_code={item_code} must have one aligned forecast horizon/date sequence across RDCs.'
            )
        horizon = horizons.pop()
        if horizon <= 0:
            raise ValueError(f'item_code={item_code} has an empty forecast horizon.')

        rdc_inventory = {}
        initial_service_levels = {}
        for rdc_id, lead_time in rdc_dict.items():
            if initial_inventory_quantile is None:
                holding = max(float(holding_cost[rdc_id]), 1e-8)
                shortage = max(float(backorder_cost[rdc_id]), 0.0)
                service_level = shortage / (shortage + holding)
                service_level = float(np.clip(service_level, 0.5, 0.9))
            else:
                service_level = float(np.clip(initial_inventory_quantile, 0.0, 1.0))
            support, pmf = quantile_series_lead_time_pmf(
                demand_feature[rdc_id], 0, int(lead_time)
            )
            on_hand = round(_pmf_quantile(support, pmf, service_level))
            rdc_inventory[rdc_id] = np.hstack((float(on_hand), np.zeros(int(lead_time))))
            initial_service_levels[rdc_id] = service_level

        cdc_windows = [
            (demand_feature[rdc_id], int(rdc_dict[rdc_id]), int(cdc_lead_time))
            for rdc_id in rdc_dict.keys()
        ]
        cdc_support, cdc_pmf = quantile_series_sum_window_pmf(cdc_windows)
        cdc_service_level = max(initial_service_levels.values())
        cdc_on_hand = round(_pmf_quantile(cdc_support, cdc_pmf, cdc_service_level))
        cdc_inventory = np.hstack((float(cdc_on_hand), np.zeros(int(cdc_lead_time))))

        data_sets.append((
            rdc_dict.copy(),
            demand_feature,
            holding_cost.copy(),
            backorder_cost.copy(),
            int(cdc_lead_time),
            rdc_inventory,
            cdc_inventory,
        ))
        metadata.append({
            'item_code': item_code,
            'rdc_names': dict(NONSTATIONARY_RDC_NAME_TO_ID),
            'dates': demand_feature[1]['dates'],
            'horizon': horizon,
            'initial_inventory_protocol': 'cold_start_aligned_windows_v2',
            'initial_inventory_quantile': initial_inventory_quantile,
            'initial_service_levels': initial_service_levels,
            'cdc_initial_service_level': cdc_service_level,
        })
    return data_sets, metadata


def generate_ratio_data():
    ratios = [1, 2, 5, 20, 100]
    # ratios = [10, 30, 50, 75]
    rdc_lead_time = {1: 1, 2: 1}
    holding_cost = {0: 1, 1: 2, 2: 2}
    backorder_cost = {1: 10, 2: 10}
    demand_feature = {1: 10, 2: 10}
    item_lead_time = 1
    RDC_initial_inventory = {1: 30, 2: 30}
    CDC_initial_inventory = 50

    all_CDC_inventory = np.hstack((CDC_initial_inventory, np.zeros(item_lead_time)))

    data_sets = []

    for r in ratios:
        new_demand_feature = demand_feature.copy()
        new_demand_feature[2] *= r
        new_rdc_lead_time = rdc_lead_time.copy()
        new_rdc_lead_time = {k: v for k, v in new_rdc_lead_time.items()}
        all_RDC_inventory = {}
        for rdc in rdc_lead_time.keys():
            all_RDC_inventory[rdc] = np.hstack((RDC_initial_inventory[rdc], np.zeros(new_rdc_lead_time[rdc])))
        data_sets.append((new_rdc_lead_time, new_demand_feature, holding_cost, backorder_cost, item_lead_time, all_RDC_inventory, all_CDC_inventory))
    
    for r in ratios:
        new_holding_cost = holding_cost.copy()
        new_holding_cost[2] *= r
        new_rdc_lead_time = rdc_lead_time.copy()
        new_rdc_lead_time = {k: v for k, v in new_rdc_lead_time.items()}
        all_RDC_inventory = {}
        for rdc in rdc_lead_time.keys():
            all_RDC_inventory[rdc] = np.hstack((RDC_initial_inventory[rdc], np.zeros(new_rdc_lead_time[rdc])))
        data_sets.append((new_rdc_lead_time, demand_feature, new_holding_cost, backorder_cost, item_lead_time, all_RDC_inventory, all_CDC_inventory))

    for r in ratios:
        new_rdc_lead_time = rdc_lead_time.copy()
        new_rdc_lead_time[2] *= r
        new_rdc_lead_time = {k: v for k, v in new_rdc_lead_time.items()}
        all_RDC_inventory = {}
        for rdc in rdc_lead_time.keys():
            all_RDC_inventory[rdc] = np.hstack((RDC_initial_inventory[rdc], np.zeros(new_rdc_lead_time[rdc])))
        data_sets.append((new_rdc_lead_time, demand_feature, holding_cost, backorder_cost, item_lead_time, all_RDC_inventory, all_CDC_inventory))

    return data_sets


SENSITIVITY_ANALYSIS_SOURCE_INDICES = (0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14)
SENSITIVITY_5_POISSON_RATE = 1000.0
SENSITIVITY_5_TAIL_PROB = 1e-10
SENSITIVITY_5_DEMAND_CAP = poisson_tail_lump_cap(
    SENSITIVITY_5_POISSON_RATE,
    SENSITIVITY_5_TAIL_PROB,
)


def data_sensitivity_analysis_13():
    """Return the 13 unique instances used by the sensitivity-analysis table.

    ``generate_ratio_data`` creates 15 entries because the unit multiplier is
    repeated in the demand, holding-cost, and lead-time sweeps.  The paper
    table keeps the first baseline and removes the two later duplicates.
    """
    all_cases = generate_ratio_data()
    return [all_cases[index] for index in SENSITIVITY_ANALYSIS_SOURCE_INDICES]


def data_sensitivity_analysis_13_corrected():
    """Return a versioned sensitivity dataset with a valid high-rate case 5.

    The legacy scalar Poisson path intentionally retains its historical global
    demand cap.  Sensitivity-5 instead declares an explicit tail-lumped support
    so its Poisson(1000) demand is not silently collapsed at 500.
    """
    cases = data_sensitivity_analysis_13()
    corrected = list(cases)
    case = list(corrected[4])
    demand_feature = case[1].copy()
    demand_feature[2] = (
        "truncated_poisson",
        SENSITIVITY_5_POISSON_RATE,
        SENSITIVITY_5_DEMAND_CAP,
    )
    case[1] = demand_feature
    corrected[4] = tuple(case)
    return corrected


def data_sensitivity_analysis_13_metadata():
    """Describe the publication order and varied retailer-2 parameter."""
    cases = data_sensitivity_analysis_13()
    metadata = []
    for table_index, (source_index, data) in enumerate(
            zip(SENSITIVITY_ANALYSIS_SOURCE_INDICES, cases), start=1):
        rdc_lead_time, demand_feature, holding_cost, *_ = data
        if table_index <= 5:
            varied_parameter = "lambda_2"
        elif table_index <= 9:
            varied_parameter = "h_2"
        else:
            varied_parameter = "L_2"
        metadata.append({
            "case": f"sensitivity-{table_index}",
            "table_index": table_index,
            "source_index": source_index,
            "varied_parameter": varied_parameter,
            "L_2": int(rdc_lead_time[2]),
            "lambda_2": float(demand_feature[2]),
            "h_2": float(holding_cost[2]),
        })
    return metadata


def data_sensitivity_analysis_13_corrected_metadata():
    """Describe the corrected sensitivity dataset and its demand protocol."""
    metadata = data_sensitivity_analysis_13_metadata()
    corrected_cases = data_sensitivity_analysis_13_corrected()
    output = []
    for row, data in zip(metadata, corrected_cases):
        item = row.copy()
        demand_spec = data[1][2]
        item["lambda_2"] = float(demand_mean(demand_spec))
        item["demand_spec_2"] = demand_spec
        item["demand_protocol"] = (
            "poisson_1000_tail_lump_1e-10"
            if item["case"] == "sensitivity-5"
            else "legacy"
        )
        output.append(item)
    return output


STRUCTURAL_SENSITIVITY_DEMAND_RAW = (1.0, 5.0, 10.0, 20.0)
# A ten-fold priority spread is strong enough to identify service
# heterogeneity while preserving the homogeneous design's mean cost exactly.
# Using the originally proposed (2, 10, 50, 100) unchanged would quadruple the
# average backorder cost and confound heterogeneity with cost scale.
STRUCTURAL_SENSITIVITY_BACKORDER_RAW = (2.0, 6.0, 12.0, 20.0)
STRUCTURAL_SENSITIVITY_TOTAL_DEMAND = 20.0
STRUCTURAL_SENSITIVITY_MEAN_BACKORDER_COST = 10.0


def _scale_to_sum(values, target_sum):
    values = np.asarray(values, dtype=float)
    return values * (float(target_sum) / float(values.sum()))


def _build_structural_sensitivity_case(demand_rates, backorder_costs):
    """Build one scale-controlled 4-RDC structural-analysis instance."""
    demand_rates = np.asarray(demand_rates, dtype=float)
    backorder_costs = np.asarray(backorder_costs, dtype=float)
    if demand_rates.shape != (4,) or backorder_costs.shape != (4,):
        raise ValueError("Structural sensitivity cases require four RDCs")

    rdc_lead_time = {rdc: 2 for rdc in range(1, 5)}
    demand_feature = {
        rdc: float(demand_rates[rdc - 1]) for rdc in range(1, 5)
    }
    holding_cost = {0: 1.0, **{rdc: 2.0 for rdc in range(1, 5)}}
    backorder_cost = {
        rdc: float(backorder_costs[rdc - 1]) for rdc in range(1, 5)
    }
    cdc_lead_time = 2

    # Initial stocks only suppress cold-start transients. Formal evaluation
    # still discards a 500-period warm-up window.
    rdc_inventory = {
        rdc: np.hstack((
            np.ceil(demand_feature[rdc] * (rdc_lead_time[rdc] + 1)),
            np.zeros(rdc_lead_time[rdc]),
        ))
        for rdc in range(1, 5)
    }
    cdc_exposure = sum(
        demand_feature[rdc]
        * (cdc_lead_time + rdc_lead_time[rdc] + 1)
        for rdc in range(1, 5)
    )
    cdc_inventory = np.hstack((np.ceil(cdc_exposure), np.zeros(cdc_lead_time)))
    return (
        rdc_lead_time,
        demand_feature,
        holding_cost,
        backorder_cost,
        cdc_lead_time,
        rdc_inventory,
        cdc_inventory,
    )


def data_structural_sensitivity():
    """Return the four scale-controlled cases for structural analysis.

    Demand heterogeneity preserves aggregate mean demand. Service
    heterogeneity preserves the arithmetic mean backorder cost. This prevents
    the two treatment indicators from being simple demand/cost scale changes.
    """
    homogeneous_demand = np.repeat(
        STRUCTURAL_SENSITIVITY_TOTAL_DEMAND / 4.0, 4
    )
    heterogeneous_demand = _scale_to_sum(
        STRUCTURAL_SENSITIVITY_DEMAND_RAW,
        STRUCTURAL_SENSITIVITY_TOTAL_DEMAND,
    )
    homogeneous_backorder = np.repeat(
        STRUCTURAL_SENSITIVITY_MEAN_BACKORDER_COST, 4
    )
    heterogeneous_backorder = _scale_to_sum(
        STRUCTURAL_SENSITIVITY_BACKORDER_RAW,
        4.0 * STRUCTURAL_SENSITIVITY_MEAN_BACKORDER_COST,
    )
    designs = (
        (homogeneous_demand, homogeneous_backorder),
        (heterogeneous_demand, homogeneous_backorder),
        (homogeneous_demand, heterogeneous_backorder),
        (heterogeneous_demand, heterogeneous_backorder),
    )
    return [
        _build_structural_sensitivity_case(demand, backorder)
        for demand, backorder in designs
    ]


def data_structural_sensitivity_metadata():
    """Describe the orthogonal heterogeneity design used in the paper."""
    labels = (
        ("homogeneous", False, False),
        ("demand-heterogeneous", True, False),
        ("service-heterogeneous", False, True),
        ("fully-heterogeneous", True, True),
    )
    metadata = []
    for data, (case, demand_heterogeneous, service_heterogeneous) in zip(
            data_structural_sensitivity(), labels):
        demand = np.asarray(list(data[1].values()), dtype=float)
        backorder = np.asarray(list(data[3].values()), dtype=float)
        metadata.append({
            "case": case,
            "demand_heterogeneous": demand_heterogeneous,
            "service_heterogeneous": service_heterogeneous,
            "demand_rates": demand.tolist(),
            "backorder_costs": backorder.tolist(),
            "total_mean_demand": float(demand.sum()),
            "mean_backorder_cost": float(backorder.mean()),
            "demand_cv": float(demand.std() / demand.mean()),
            "backorder_cost_cv": float(backorder.std() / backorder.mean()),
        })
    return metadata


def structural_sensitivity_scarcity_settings():
    """Return exogenous CDC targets for low/medium/high competition.

    The exposure window is the corrected joint window ``L0 + Lk + 1``.
    ``eta`` is therefore a dimensionless scarcity index rather than an
    endogenous policy output. All compared methods receive the same fixed CDC
    echelon order-up-to target in each setting.
    """
    data = data_structural_sensitivity()[0]
    rdc_lead_time, demand_feature, _, _, cdc_lead_time, *_ = data
    exposure = sum(
        demand_mean(demand_feature[rdc])
        * (cdc_lead_time + rdc_lead_time[rdc] + 1)
        for rdc in rdc_lead_time
    )
    # Targets were calibrated with short common-demand RB trajectories. They
    # produce roughly 2%, 45%, and 99% CDC-binding periods, respectively,
    # without turning the high-competition case into a near-zero-service case.
    settings = (
        ("low-competition", 133),
        ("medium-competition", 120),
        ("high-competition", 105),
    )
    return [
        {
            "setting": setting,
            "eta": float(exposure / fixed_cdc_bsl),
            "joint_exposure": float(exposure),
            "fixed_cdc_bsl": int(fixed_cdc_bsl),
        }
        for setting, fixed_cdc_bsl in settings
    ]


def generate_random_ratio_data(ratios, sample_num=3, random_seed=10):
    rd.seed(random_seed)

    data_sets = []
    for _ in range(sample_num):
        item_lead_time = rd.randint(2, 6)

        rdc_dict = {k: rd.randint(2, 6) for k in range(1, rdc_num + 1)}
        demand_feature = {k: rd.randint(10, 51) for k in range(1, rdc_num + 1)}
        holding_cost = {k: 1 + 9 * rd.random() for k in range(1, rdc_num + 1)}
        holding_cost[0] = 1.
        backorder_cost = {k: 10 for k in range(1, rdc_num + 1)}

        RDC_initial_inventory = {k: rd.randint(10, 101) for k in range(1, rdc_num + 1)}
        CDC_initial_inventory = rd.randint(40, 101)
        all_RDC_inventory, all_CDC_inventory = {}, {}
        for rdc in rdc_dict.keys():
            all_RDC_inventory[rdc] = np.hstack((RDC_initial_inventory[rdc], np.zeros(rdc_dict[rdc])))
        all_CDC_inventory = np.hstack((CDC_initial_inventory, np.zeros(item_lead_time)))
        
        data_sets.append((rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, all_RDC_inventory, all_CDC_inventory))
    rdc_num = len(ratios)
    rdc_lead_time = defaultdict(int)



def select_dict_length(dic, length):
    return {k: v for k, v in dic.items() if len(v) == length}

def generate_data():
    data_sets = []
    for v in rdc_dict.values():
        rdc_num = len(v)
        hc = select_dict_length(holding_cost, rdc_num + 1)
        bc_input = backorder_cost[rdc_num]
        df = select_dict_length(demand_feature, rdc_num)
        item_lt = item_lead_time[1]

        RDC_inv_input, CDC_inv_input = {}, {}
        for i in range(rdc_num):
            RDC_inv_input[i + 1] = np.hstack((RDC_initial_inventory[i + 1], np.zeros(v[i + 1])))
        CDC_inv_input = np.hstack((CDC_initial_inventory, np.zeros(item_lt)))
        for v1 in hc.values():
            hc_input = v1
            for v2 in df.values():
                df_input = v2
                data_sets.append((v, df_input, hc_input, bc_input, item_lt, RDC_inv_input, CDC_inv_input))
    return data_sets


def generate_certain_data():
    data_sets = []

    # rdc_dict = {1: 2, 2: 2}
    # demand_feature = {1: 10, 2: 40}
    # holding_cost = {0: 1, 1: 1, 2: 2}
    # backorder_cost = {1: 10, 2: 10}
    # rdc_dict = {1: 4, 2: 4, 3: 5, 4: 5, 5: 3, 6: 3, 7: 4, 8: 3, 9: 4, 10: 5, 11: 5, 12: 4, 13: 4, 14: 2, 15: 4, 16: 5, 17: 5, 18: 2, 19: 5, 20: 4, 21: 5, 22: 5, 23: 3, 24: 4, 25: 3, 26: 2, 27: 5, 28: 4, 29: 3, 30: 4, 31: 4, 32: 2, 33: 3, 34: 4, 35: 5, 36: 5, 37: 5, 38: 2, 39: 5, 40: 4, 41: 5, 42: 2, 43: 5, 44: 4, 45: 3, 46: 3, 47: 3, 48: 2, 49: 4, 50: 5}
    # demand_feature = {1: 42, 2: 34, 3: 50, 4: 18, 5: 32, 6: 21, 7: 18, 8: 34, 9: 50, 10: 36, 11: 26, 12: 38, 13: 36, 14: 40, 15: 11, 16: 37, 17: 12, 18: 34, 19: 23, 20: 23, 21: 35, 22: 50, 23: 16, 24: 49, 25: 47, 26: 27, 27: 16, 28: 48, 29: 47, 30: 40, 31: 12, 32: 46, 33: 20, 34: 35, 35: 28, 36: 42, 37: 26, 38: 32, 39: 50, 40: 41, 41: 48, 42: 14, 43: 33, 44: 49, 45: 10, 46: 48, 47: 24, 48: 19, 49: 49, 50: 36}
    # holding_cost = {1: 9.128545657569294, 2: 1.0879297940771717, 3: 3.032314539970913, 4: 2.188496145447882, 5: 9.16788651864446, 6: 9.192075729937867, 7: 6.229521164214145, 8: 1.7874374727151485, 9: 2.0519149561695835, 10: 7.93964129777069, 11: 7.595157419168698, 12: 1.7841552123926099, 13: 4.217168783078963, 14: 7.958904595770354, 15: 2.183244414505805, 16: 5.840491904685766, 17: 7.792512096042542, 18: 3.4527355839202163, 19: 6.0986529631659545, 20: 5.290164736449958, 21: 6.009586232876555, 22: 4.96663351672421, 23: 7.243636278357205, 24: 7.464134622876825, 25: 7.807441221060387, 26: 1.3358980811020518, 27: 7.109518572036832, 28: 5.29498720817141, 29: 1.9015424890383894, 30: 6.527766103432213, 31: 8.541230580695764, 32: 7.605046338929501, 33: 3.8940902444071654, 34: 1.6106977997821033, 35: 1.3359721335784527, 36: 6.032127735644339, 37: 2.4476742934169304, 38: 3.4109936713188267, 39: 3.1214076079239725, 40: 1.1741271545604457, 41: 2.3651778195548125, 42: 1.3051226181930558, 43: 9.836513722790905, 44: 4.244367791386528, 45: 8.429180757058152, 46: 4.710136676657097, 47: 3.278612316789459, 48: 3.130397888321444, 49: 7.958040393490425, 50: 7.041312146130685, 0: 1.0}
    # backorder_cost = {1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10, 9: 10, 10: 10, 11: 10, 12: 10, 13: 10, 14: 10, 15: 10, 16: 10, 17: 10, 18: 10, 19: 10, 20: 10, 21: 10, 22: 10, 23: 10, 24: 10, 25: 10, 26: 10, 27: 10, 28: 10, 29: 10, 30: 10, 31: 10, 32: 10, 33: 10, 34: 10, 35: 10, 36: 10, 37: 10, 38: 10, 39: 10, 40: 10, 41: 10, 42: 10, 43: 10, 44: 10, 45: 10, 46: 10, 47: 10, 48: 10, 49: 10, 50: 10}
    # # item lead time: 2
    # all_RDC_inventory = {1: np.array([86.,  0.,  0.,  0.,  0.]), 2: np.array([86.,  0.,  0.,  0.,  0.]), 3: np.array([68.,  0.,  0.,  0.,  0.,  0.]), 4: np.array([75.,  0.,  0.,  0.,  0.,  0.]), 5: np.array([92.,  0.,  0.,  0.]), 6: np.array([33.,  0.,  0.,  0.]), 7: np.array([34.,  0.,  0.,  0.,  0.]), 8: np.array([82.,  0.,  0.,  0.]), 9: np.array([64.,  0.,  0.,  0.,  0.]), 10: np.array([90.,  0.,  0.,  0.,  0.,  0.]), 11: np.array([31.,  0.,  0.,  0.,  0.,  0.]), 12: np.array([64.,  0.,  0.,  0.,  0.]), 13: np.array([29.,  0.,  0.,  0.,  0.]), 14: np.array([64.,  0.,  0.]), 15: np.array([13.,  0.,  0.,  0.,  0.]), 16: np.array([35.,  0.,  0.,  0.,  0.,  0.]), 17: np.array([51.,  0.,  0.,  0.,  0.,  0.]), 18: np.array([24.,  0.,  0.]), 19: np.array([83.,  0.,  0.,  0.,  0.,  0.]), 20: np.array([81.,  0.,  0.,  0.,  0.]), 21: np.array([48.,  0.,  0.,  0.,  0.,  0.]), 22: np.array([59.,  0.,  0.,  0.,  0.,  0.]), 23: np.array([32.,  0.,  0.,  0.]), 24: np.array([57.,  0.,  0.,  0.,  0.]), 25: np.array([41.,  0.,  0.,  0.]), 26: np.array([56.,  0.,  0.]), 27: np.array([15.,  0.,  0.,  0.,  0.,  0.]), 28: np.array([24.,  0.,  0.,  0.,  0.]), 29: np.array([86.,  0.,  0.,  0.]), 30: np.array([26.,  0.,  0.,  0.,  0.]), 31: np.array([30.,  0.,  0.,  0.,  0.]), 32: np.array([92.,  0.,  0.]), 33: np.array([19.,  0.,  0.,  0.]), 34: np.array([66.,  0.,  0.,  0.,  0.]), 35: np.array([47.,  0.,  0.,  0.,  0.,  0.]), 36: np.array([16.,  0.,  0.,  0.,  0.,  0.]), 37: np.array([40.,  0.,  0.,  0.,  0.,  0.]), 38: np.array([77.,  0.,  0.]), 39: np.array([81.,  0.,  0.,  0.,  0.,  0.]), 40: np.array([80.,  0.,  0.,  0.,  0.]), 41: np.array([63.,  0.,  0.,  0.,  0.,  0.]), 42: np.array([66.,  0.,  0.]), 43: np.array([23.,  0.,  0.,  0.,  0.,  0.]), 44: np.array([20.,  0.,  0.,  0.,  0.]), 45: np.array([70.,  0.,  0.,  0.]), 46: np.array([43.,  0.,  0.,  0.]), 47: np.array([52.,  0.,  0.,  0.]), 48: np.array([10.,  0.,  0.]), 49: np.array([98.,  0.,  0.,  0.,  0.]), 50: np.array([64.,  0.,  0.,  0.,  0.,  0.])}
    # all_CDC_inventory = [96.,  0.,  0.]

    # rdc_dict = {1: 5, 2: 4, 3: 4, 4: 4, 5: 4, 6: 3, 7: 5, 8: 3, 9: 3, 10: 3, 11: 5, 12: 3, 13: 3, 14: 3, 15: 5, 16: 3, 17: 4, 18: 4, 19: 2, 20: 4, 21: 4, 22: 4, 23: 2, 24: 3, 25: 4, 26: 4, 27: 4, 28: 4, 29: 3, 30: 4, 31: 4, 32: 2, 33: 4, 34: 5, 35: 2, 36: 4, 37: 2, 38: 5, 39: 5, 40: 5, 41: 4, 42: 2, 43: 5, 44: 3, 45: 3, 46: 5, 47: 3, 48: 2, 49: 2, 50: 5}
    # demand_feature = {1: 14, 2: 24, 3: 45, 4: 32, 5: 40, 6: 49, 7: 14, 8: 33, 9: 47, 10: 12, 11: 45, 12: 12, 13: 10, 14: 17, 15: 39, 16: 18, 17: 26, 18: 17, 19: 42, 20: 30, 21: 21, 22: 22, 23: 26, 24: 48, 25: 46, 26: 42, 27: 19, 28: 11, 29: 47, 30: 18, 31: 49, 32: 40, 33: 49, 34: 15, 35: 49, 36: 42, 37: 44, 38: 34, 39: 17, 40: 29, 41: 47, 42: 29, 43: 47, 44: 34, 45: 42, 46: 25, 47: 32, 48: 43, 49: 47, 50: 12}
    # holding_cost = {1: 8.488762851814737, 2: 1.1211270264978197, 3: 5.584018523627248, 4: 5.3159383012712755, 5: 1.2668944671795541, 6: 1.1028567414165962, 7: 3.986596002110449, 8: 1.5528797074476546, 9: 6.3835615725082695, 10: 8.973282281703609, 11: 4.709203978275466, 12: 1.3444497901197612, 13: 1.727581634117251, 14: 1.7372308807166372, 15: 8.284306004393814, 16: 8.206246025964981, 17: 1.1521883021923693, 18: 8.75164892707556, 19: 9.201702019172036, 20: 4.607890161716459, 21: 3.5820162850393698, 22: 8.587403967374547, 23: 6.553400976796748, 24: 2.265388541068548, 25: 3.058800523443243, 26: 8.573481740576764, 27: 8.599226261573783, 28: 2.981313940425748, 29: 8.300238757769458, 30: 5.82358180500204, 31: 6.493856040841899, 32: 5.714828948136535, 33: 6.756662336460064, 34: 1.0464071323052537, 35: 1.2226490507405758, 36: 7.210237467841411, 37: 2.5443371760599205, 38: 7.2453760631979085, 39: 3.567390608096315, 40: 1.4959532870351957, 41: 6.240006123274084, 42: 3.6623685390903593, 43: 3.905345220177742, 44: 9.702904490694968, 45: 2.7544119830477807, 46: 4.126568546310526, 47: 1.9361265859963819, 48: 4.6160366722704005, 49: 1.8356288057856203, 50: 5.543594258155293, 0: 1.0}
    # backorder_cost = {1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10, 9: 10, 10: 10, 11: 10, 12: 10, 13: 10, 14: 10, 15: 10, 16: 10, 17: 10, 18: 10, 19: 10, 20: 10, 21: 10, 22: 10, 23: 10, 24: 10, 25: 10, 26: 10, 27: 10, 28: 10, 29: 10, 30: 10, 31: 10, 32: 10, 33: 10, 34: 10, 35: 10, 36: 10, 37: 10, 38: 10, 39: 10, 40: 10, 41: 10, 42: 10, 43: 10, 44: 10, 45: 10, 46: 10, 47: 10, 48: 10, 49: 10, 50: 10}
    # # item_lead_time = 2
    # all_RDC_inventory = {1: np.array([91.,  0.,  0.,  0.,  0.,  0.]), 2: np.array([58.,  0.,  0.,  0.,  0.]), 3: np.array([19.,  0.,  0.,  0.,  0.]), 4: np.array([38.,  0.,  0.,  0.,  0.]), 5: np.array([44.,  0.,  0.,  0.,  0.]), 6: np.array([86.,  0.,  0.,  0.]), 7: np.array([83.,  0.,  0.,  0.,  0.,  0.]), 8: np.array([45.,  0.,  0.,  0.]), 9: np.array([95.,  0.,  0.,  0.]), 10: np.array([81.,  0.,  0.,  0.]), 11: np.array([19.,  0.,  0.,  0.,  0.,  0.]), 12: np.array([39.,  0.,  0.,  0.]), 13: np.array([22.,  0.,  0.,  0.]), 14: np.array([15.,  0.,  0.,  0.]), 15: np.array([82.,  0.,  0.,  0.,  0.,  0.]), 16:np.array([66.,  0.,  0.,  0.]), 17: np.array([47.,  0.,  0.,  0.,  0.]), 18: np.array([47.,  0.,  0.,  0.,  0.]), 19: np.array([51.,  0.,  0.]), 20: np.array([83.,  0.,  0.,  0.,  0.]), 21: np.array([46.,  0.,  0.,  0.,  0.]), 22: np.array([16.,  0.,  0.,  0.,  0.]), 23: np.array([31.,  0.,  0.]), 24: np.array([59.,  0.,  0.,  0.]), 25: np.array([75.,  0.,  0.,  0.,  0.]), 26: np.array([63.,  0.,  0.,  0.,  0.]), 27: np.array([16.,  0.,  0.,  0.,  0.]), 28: np.array([69.,  0.,  0.,  0.,  0.]), 29: np.array([12.,  0.,  0.,  0.]), 30: np.array([55.,  0.,  0.,  0.,  0.]), 31: np.array([58.,  0.,  0.,  0.,  0.]), 32: np.array([56.,  0.,  0.]), 33: np.array([98.,  0.,  0.,  0.,  0.]), 34: np.array([92.,  0.,  0.,  0.,  0.,  0.]), 35: np.array([46.,  0.,  0.]), 36: np.array([12.,  0.,  0.,  0.,  0.]), 37: np.array([83.,  0.,  0.]), 38: np.array([45.,  0.,  0.,  0.,  0.,  0.]), 39: np.array([91.,  0.,  0.,  0.,  0.,  0.]), 40: np.array([46.,  0.,  0.,  0.,  0.,  0.]), 41: np.array([38.,  0.,  0.,  0.,  0.]), 42: np.array([57.,  0.,  0.]), 43: np.array([29.,  0.,  0.,  0.,  0.,  0.]), 44: np.array([12.,  0.,  0.,  0.]), 45: np.array([74.,  0.,  0.,  0.]), 46: np.array([77.,  0.,  0.,  0.,  0.,  0.]), 47: np.array([84.,  0.,  0.,  0.]), 48: np.array([45.,  0.,  0.]), 49: np.array([63.,  0.,  0.]), 50: np.array([17.,  0.,  0.,  0.,  0.,  0.])}
    # all_CDC_inventory = [48.,  0.,  0.]
    item_lead_time = 2

    rdc_dict = {1: 2, 2: 2, 3: 2}
    demand_feature = {1: 10, 2: 10, 3: 10}
    holding_cost = {0: 1, 1: 4, 2: 2, 3: 6}
    backorder_cost = {1: 10, 2: 10, 3: 10}
    
    all_RDC_inventory, all_CDC_inventory = {}, {}
    for rdc in rdc_dict.keys():
        all_RDC_inventory[rdc] = np.hstack((RDC_initial_inventory[rdc], np.zeros(rdc_dict[rdc])))
    all_CDC_inventory = np.hstack((CDC_initial_inventory, np.zeros(item_lead_time)))

    data_sets.append((rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, all_RDC_inventory, all_CDC_inventory))

    return data_sets


def generate_data_with_random_para(rdc_num=3, sample_num=3, random_seed=10):
    rd.seed(random_seed)

    data_sets = []
    for _ in range(sample_num):
        item_lead_time = rd.randint(2, 6) - 1

        rdc_dict = {k: rd.randint(2, 6) - 1 for k in range(1, rdc_num + 1)}
        demand_feature = {k: rd.randint(10, 51) for k in range(1, rdc_num + 1)}
        holding_cost = {k: 1 + 9 * rd.random() for k in range(1, rdc_num + 1)}
        holding_cost[0] = 1.
        backorder_cost = {k: 10 for k in range(1, rdc_num + 1)}

        RDC_initial_inventory = {k: rd.randint(10, 101) for k in range(1, rdc_num + 1)}
        CDC_initial_inventory = rd.randint(40, 101)
        all_RDC_inventory, all_CDC_inventory = {}, {}
        for rdc in rdc_dict.keys():
            all_RDC_inventory[rdc] = np.hstack((RDC_initial_inventory[rdc], np.zeros(rdc_dict[rdc])))
        all_CDC_inventory = np.hstack((CDC_initial_inventory, np.zeros(item_lead_time)))
        
        data_sets.append((rdc_dict, demand_feature, holding_cost, backorder_cost, item_lead_time, all_RDC_inventory, all_CDC_inventory))

    return data_sets
