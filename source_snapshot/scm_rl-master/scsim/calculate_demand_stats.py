import numpy as np

mean_demand_stores = [0, 2, 4, 6, 8, 10, 0.5, 3, 9, 12] # for poisson std = mu^0.5
std_demand_stores = [20, 16, 12, 8, 4, 0, 0.5, 3, 9, 12]

normal = 'normal'
poisson = 'poisson'

sample_size = 10_000_000

demand_distribution = [normal, normal, normal, normal, normal, normal, poisson, poisson, poisson, poisson]

samples = []
for dist_type, mean, std in zip(demand_distribution, mean_demand_stores, std_demand_stores):
    if dist_type == 'normal':
        demand_sample = np.random.normal(mean, std, size=sample_size)
    elif dist_type == 'poisson':
        demand_sample = np.random.poisson(mean, size=sample_size)
    else:
        raise ValueError()
    demand_sample = np.clip(np.round(demand_sample), 0, 500)
    samples.append(demand_sample)



means = []
stds = []
for sample in samples:
    mean, std = np.mean(sample), np.std(sample)
    means.append(np.round(mean, decimals=3))
    stds.append(np.round(std, decimals=3))
from loguru import logger

logger.info('real_demand_means: {}'.format(str(means)))
logger.info('real_demand_stds: {}'.format(str(stds)))
