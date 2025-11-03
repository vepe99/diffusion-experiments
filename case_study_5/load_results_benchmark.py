import os
os.environ["KERAS_BACKEND"] = "torch"

import pickle
import numpy as np
import pandas as pd
import itertools
import sbibm

from model_settings_benchmark import MODELS, SAMPLER_SETTINGS

benchmarks = itertools.product(range(len(MODELS)), range(len(sbibm.get_available_tasks())))
storage = 'plots/sbi_benchmark/'

results_dict = {}
for model_name in MODELS.keys():
    for sampler in SAMPLER_SETTINGS.keys():
        if sampler.startswith('sde') and not model_name.startswith('diffusion'):
            continue
        results_dict.update({
                model_name+'-'+sampler:
                    np.ones(len(sbibm.get_available_tasks())) * np.nan
            })
results_dict_std = results_dict.copy()

for model_i, task_i in benchmarks:
    task_name = sbibm.get_available_tasks()[task_i]
    model_name = list(MODELS.keys())[model_i]

    with open(f'{storage}c2st_results_{model_name}_{task_name}.pkl', 'rb') as f:
        c2st_results = pickle.load(f)

    for sampler in SAMPLER_SETTINGS.keys():
        if sampler.startswith('sde') and not model_name.startswith('diffusion'):
            continue
        mean_c2st = np.mean(c2st_results[sampler])
        std_c2st = np.std(c2st_results[sampler])

        results_dict[model_name+'-'+sampler][task_i] = mean_c2st
        results_dict_std[model_name+'-'+sampler][task_i] = std_c2st

df = pd.DataFrame.from_dict(results_dict, orient='index', columns=sbibm.get_available_tasks())
df_std = pd.DataFrame.from_dict(results_dict_std, orient='index', columns=sbibm.get_available_tasks())
df.index.name = 'Model-Sampler'
df_std.index.name = 'Model-Sampler'
df = df.join(df_std, rsuffix='_std')
ordered_columns = []
for task in sbibm.get_available_tasks():
    ordered_columns.append(task)
    ordered_columns.append(task + '_std')
df = df[ordered_columns]
df['sampler'] = [name.split('-')[1] for name in df.index]
df['model'] = [name.split('-')[0] for name in df.index]
df.to_csv(f'{storage}c2st_benchmark_results.csv')
