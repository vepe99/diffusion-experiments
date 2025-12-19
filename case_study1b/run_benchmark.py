#%%
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import pickle
import itertools
import time
from pathlib import Path
import logging

import torch
import sbibm
from sbibm.metrics import c2st
import bayesflow as bf

from case_study1b.model_settings_benchmark import load_model, MODELS, SAMPLER_SETTINGS, NUM_BATCHES_PER_EPOCH, BATCH_SIZE


logging.getLogger("bayesflow").setLevel(logging.DEBUG)

job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
benchmarks = list(itertools.product(range(len(MODELS)), range(len(sbibm.get_available_tasks()))))
model_i, task_i = benchmarks[job_id]
BASE = Path(__file__).resolve().parent

task_name = sbibm.get_available_tasks()[task_i]
print(task_name)
sim_budget = 'online'
task = sbibm.get_task(task_name)
conf_tuple = list(MODELS.values())[model_i]
model_name = list(MODELS.keys())[model_i]
print(model_name)

if 'offline' in model_name:
    sim_budget = NUM_BATCHES_PER_EPOCH*BATCH_SIZE

#%%
if task_name == 'lotka_volterra':
    # sbibm requires julia for this task
    simulator = bf.simulators.LotkaVolterra(subsample='original')

elif task_name == 'sir':
    # sbibm requires julia for this task
    simulator = bf.simulators.SIR(subsample='original', scale_by_total=True)  # sbibm uses scale_by_total=False
else:
    prior = task.get_prior()
    sbibm_simulator = task.get_simulator()

    def sbibm_simulator_bf():
        thetas = prior(num_samples=1)
        xs = sbibm_simulator(thetas)
        return dict(parameters=thetas.numpy()[0], observables=xs.numpy()[0])

    simulator = bf.make_simulator(sbibm_simulator_bf)

if isinstance(sim_budget, int):
    training_data = simulator.sample((sim_budget,))
    print(f"Simulated {sim_budget} parameter-observation pairs.")
    print(training_data['parameters'].shape, training_data['observables'].shape)
else:
    training_data = None
    print(f"Using online simulation.")

#%%
workflow = load_model(conf_tuple=conf_tuple,
                      training_data=training_data,
                      simulator=simulator,
                      storage=BASE / "models",
                      problem_name=task_name, model_name=model_name)

if 'flow' in model_name:
    diagnostics = workflow.compute_default_diagnostics(test_data=500,
                                                    approximator_kwargs=dict(method='tsit5', steps='adaptive'))
elif 'consistency' in model_name:
    diagnostics = workflow.compute_default_diagnostics(test_data=500)
else:
    diagnostics = workflow.compute_default_diagnostics(test_data=500,
                                                    approximator_kwargs=dict(method='two_step_adaptive', steps='adaptive'))
#for k, fig in diagnostics.items():
#    fig.savefig(BASE / 'plots' / f'diagnostic_{model_name}_{task_name}_{k}.png')
diagnostics.to_csv(BASE / 'plots' / f'diagnostic_{model_name}_{task_name}.csv')

#%%
c2st_results = {sampler: [] for sampler in SAMPLER_SETTINGS.keys()}
for sampler in SAMPLER_SETTINGS.keys():
    if sampler.startswith('sde') and not model_name.startswith('diffusion'):
        continue
    elif 'consistency' in model_name and sampler != 'ode-euler':
        continue
    print(f"Evaluating sampler: {sampler}")
    for num_observation in range(1, 11):
        observation = task.get_observation(num_observation=num_observation).numpy()
        if task_name == 'sir':
            observation = observation / simulator.total_count  # sbibm SIR uses scale_by_total=False
        #elif task_name == 'lotka_volterra':
        #    # error in earlier version of bayesflow simulator for lotka volterra, needs to be reshaped
        #    observation = np.array([observation[0, :10], observation[0, 10:]]).T.flatten()[np.newaxis]
        reference_samples = task.get_reference_posterior_samples(num_observation=num_observation)
        num_samples = reference_samples.shape[0]
        start_time = time.perf_counter()
        if 'consistency' in model_name:
            posterior_samples_dict = workflow.sample(conditions={'observables': observation}, num_samples=num_samples)
        else:
            posterior_samples_dict = workflow.sample(conditions={'observables': observation}, num_samples=num_samples,
                                                     **SAMPLER_SETTINGS[sampler])
        end_time = time.perf_counter()
        posterior_samples = torch.as_tensor(posterior_samples_dict['parameters'][0])
        posterior_samples = posterior_samples[torch.isfinite(posterior_samples).all(dim=-1)]
        c2st_accuracy = c2st(reference_samples, posterior_samples).numpy().item()
        c2st_results[sampler].append((c2st_accuracy, end_time - start_time))
        print(f"{num_observation} C2ST accuracy: {c2st_accuracy}")
        print(f"Sampling time: {end_time - start_time} seconds.")

with open(BASE / 'metrics' / f'c2st_results_{model_name}_{task_name}.pkl', 'wb') as f:
    pickle.dump(c2st_results, f)
print('Done.')
