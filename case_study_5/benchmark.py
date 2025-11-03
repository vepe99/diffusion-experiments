import os
os.environ["KERAS_BACKEND"] = "torch"

import pickle
import itertools
import torch
import sbibm
from sbibm.metrics import c2st
import bayesflow as bf

from model_settings_benchmark import load_model, MODELS, SAMPLER_SETTINGS

job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
benchmarks = list(itertools.product(range(len(MODELS)), range(len(sbibm.get_available_tasks()))))
model_i, task_i = benchmarks[job_id]
storage = 'plots/sbi_benchmark/'

task_name = sbibm.get_available_tasks()[task_i]
print(task_name)
sim_budget = 10_000
num_samples = 10_000
task = sbibm.get_task(task_name)  # See sbibm.get_available_tasks() for all tasks
conf_tuple = list(MODELS.values())[model_i]
model_name = list(MODELS.keys())[model_i]
print(model_name)

if task_name == 'lotka_volterra':
    # sbibm requires julia for this task
    simulator_bf = bf.simulators.LotkaVolterra()
    sample_dict = simulator_bf.sample((sim_budget,))
    thetas = sample_dict['parameters']
    xs = sample_dict['observables']

elif task_name == 'sir':
    # sbibm requires julia for this task
    simulator_bf = bf.simulators.SIR(subsample=10)
    sample_dict = simulator_bf.sample((sim_budget,))
    thetas = sample_dict['parameters']
    xs = sample_dict['observables']
else:
    prior = task.get_prior()
    thetas = prior(num_samples=sim_budget)

    simulator = task.get_simulator()
    xs = simulator(thetas)

    thetas = thetas.numpy()
    xs = xs.numpy()

print(f"Simulated {sim_budget} parameter-observation pairs.")
print(thetas.shape, xs.shape)

# check whether we need a summary network
need_summary_network = False
if need_summary_network:
    obs_key = "summary_variables"
    sum_dim = min(thetas.shape[-1]*2, 8)
else:
    obs_key = "inference_conditions"
    sum_dim = 0
training_data = {'inference_variables': thetas, obs_key: xs}
workflow = load_model(adapter=None, conf_tuple=conf_tuple, sum_dim=sum_dim,
                      training_data=training_data,
                      storage=storage,
                      problem_name=task_name, model_name=model_name,
                      use_ema=True)

c2st_results = {'ode': [], 'sde': []}
for sampler in SAMPLER_SETTINGS.keys():
    if sampler.startswith('sde') and not model_name.startswith('diffusion'):
        continue
    for num_observation in range(1, 11):
        observation = task.get_observation(num_observation=num_observation).numpy()
        if 'consistency' in model_name:
            posterior_samples_dict = workflow.sample(conditions={obs_key: observation}, num_samples=num_samples)
        else:
            posterior_samples_dict = workflow.sample(conditions={obs_key: observation}, num_samples=num_samples,
                                                     **SAMPLER_SETTINGS[sampler])
        posterior_samples = torch.as_tensor(posterior_samples_dict['inference_variables'][0])

        reference_samples = task.get_reference_posterior_samples(num_observation=num_observation)
        c2st_accuracy = c2st(reference_samples, posterior_samples).numpy()[0]
        c2st_results[sampler].append(c2st_accuracy)
        print(f"{num_observation} C2ST accuracy: {c2st_accuracy:.3f}")

with open(f'{storage}c2st_results_{model_name}_{task_name}.pkl', 'wb') as f:
    pickle.dump(c2st_results, f)
print('Done.')
