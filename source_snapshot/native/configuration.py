import math

RISKS = [1, 2, 3, 4, 5]
ABC = ['e']
ABORTED_STATUS = ['EOS', 'EOP', 'Obsolete', '失效', '过期', '作废', 'Inactive', '消耗', '停产', 'Develop_I',
                  'EOM_I', 'EOP_A', 'EOP_I', 'EOS_A', 'GA_I', 'Pilot_I', 'Develop_T', 'EOM_T', 'EOP_T',
                  'EOS_T', 'GA_T', 'Pilot_T', 'Temporary']
START_WITH = ['97', '99', '25']
ABORTED_UNIT = ['M']
type_drop = ['PH', 'SW', 'SV', 'PTO', 'ATO', 'POC', 'AOC']
PROPORTION_SPECIAL = 0.05

# PERIOD = '20230918'
DECLINE_WINDOW_LENGTH = 24
DECLINE_PERIOD_ONE_LENGTH_FACTOR = 0.25
DECLINE_PERIOD_TWO_LENGTH_FACTOR = 0.5
DECLINE_RATIO = 0.05

merge_max_length = 50
merge_min_length = 3
repeted_ratio_high = 0.95
repeted_ratio_low = 0.05


# 多级库存模块
PERIOD = '2024-03-25'
PERIOD_CODE = ''.join(PERIOD.split('-'))
INPUT_PATH = './input_data/'
OUTPUT_PATH = f'./output/{PERIOD_CODE}/'
BOM_DECOMPOSE_PATH = INPUT_PATH + 'bom_decomposed_国内数通_9期均值'
ALL_ITEMS_PATH = INPUT_PATH + 'all_items.csv'
ALL_BOM_PATH = INPUT_PATH + 'all_bom.csv'
TRAINING_PATH = OUTPUT_PATH + 'agents'
DEPT_PATH = INPUT_PATH + '产品部门_dept'
START_DATE = PERIOD

domestic_or_abroad = 'abroad'
if domestic_or_abroad == 'domestic':
    rdc_dict = {'深圳': 2, '北京': 3, '上海': 3, '成都': 3}
    MAX_WEEK = 13
    RDC_INITIAL_INVENTORY_PATH = INPUT_PATH + f'rdc_inventory_国内_{PERIOD}.json'
    CDC_INITIAL_INVENTORY_PATH = INPUT_PATH + f'cdc_inventory_国内_{PERIOD}.json'
    PREDICTION_PATH = INPUT_PATH + f'week pred_国内rdc_{PERIOD_CODE}_筛选数通能源编码.json'
elif domestic_or_abroad == 'abroad':
    rdc_dict = {'欧洲': 32, '印度': 22, '迪拜': 21, '巴西': 36, '巴拿马': 33, '墨西哥': 28}
    MAX_WEEK = 20
    RDC_INITIAL_INVENTORY_PATH = INPUT_PATH + f'rdc_inventory_海外_{PERIOD}.json'
    CDC_INITIAL_INVENTORY_PATH = INPUT_PATH + f'cdc_inventory_海外_{PERIOD}.json'
    PREDICTION_PATH = INPUT_PATH + f'week pred_海外rdc_{PERIOD_CODE}_筛选能源编码.json'
week_rdc_dict = {k: max(1, math.ceil((v - 1) / 7)) for k, v in rdc_dict.items()}

