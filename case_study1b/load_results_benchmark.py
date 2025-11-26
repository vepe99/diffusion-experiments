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
results_dict_std = {}
results_dict_time = {}
results_dict_time_std = {}
for model_name in MODELS.keys():
    for sampler in SAMPLER_SETTINGS.keys():
        if sampler.startswith('sde') and not model_name.startswith('diffusion'):
            continue
        results_dict.update({
                model_name+'-'+sampler:
                    np.ones(len(sbibm.get_available_tasks())) * np.nan
            })
        results_dict_std.update({
                model_name+'-'+sampler:
                    np.ones(len(sbibm.get_available_tasks())) * np.nan
            })
        results_dict_time.update({
            model_name + '-' + sampler:
                np.ones(len(sbibm.get_available_tasks())) * np.nan
        })
        results_dict_time_std.update({
            model_name + '-' + sampler:
                np.ones(len(sbibm.get_available_tasks())) * np.nan
        })

for model_i, task_i in benchmarks:
    task_name = sbibm.get_available_tasks()[task_i]
    model_name = list(MODELS.keys())[model_i]

    with open(f'{storage}c2st_results_{model_name}_{task_name}.pkl', 'rb') as f:
        c2st_results = pickle.load(f)

    for sampler in SAMPLER_SETTINGS.keys():
        if sampler.startswith('sde') and not model_name.startswith('diffusion'):
            continue
        results_dict[model_name+'-'+sampler][task_i] = np.mean([c[0] for c in c2st_results[sampler]])
        results_dict_std[model_name+'-'+sampler][task_i] = np.std([c[0] for c in c2st_results[sampler]])
        results_dict_time[model_name+'-'+sampler][task_i] = np.mean([c[1] for c in c2st_results[sampler]])
        results_dict_time_std[model_name + '-' + sampler][task_i] = np.std([c[1] for c in c2st_results[sampler]])

df = pd.DataFrame.from_dict(results_dict, orient='index', columns=sbibm.get_available_tasks())
df_std = pd.DataFrame.from_dict(results_dict_std, orient='index', columns=sbibm.get_available_tasks())
df_time = pd.DataFrame.from_dict(results_dict_time, orient='index', columns=sbibm.get_available_tasks())
df_time_std = pd.DataFrame.from_dict(results_dict_time_std, orient='index', columns=sbibm.get_available_tasks())
df.index.name = 'Model-Sampler'
df_std.index.name = 'Model-Sampler'
df_time.index.name = 'Model-Sampler'
df_time_std.index.name = 'Model-Sampler'
df_std = df_std.add_suffix('_std', axis='columns')
df_time = df_time.add_suffix('_time', axis='columns')
df_time_std = df_time_std.add_suffix('_time_std', axis='columns')
df = pd.concat([df, df_std, df_time, df_time_std], axis=1)
ordered_columns = []
for task in sbibm.get_available_tasks():
    ordered_columns.append(task)
    ordered_columns.append(task + '_std')
    ordered_columns.append(task + '_time')
    ordered_columns.append(task + '_time_std')
df = df[ordered_columns]
df['sampler'] = ['-'.join(name.split('-')[1:]) for name in df.index]
df['model'] = [name.split('-')[0] for name in df.index]
df.to_csv(f'{storage}c2st_benchmark_results.csv')
print(df)
