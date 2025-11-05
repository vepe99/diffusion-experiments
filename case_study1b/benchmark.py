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
sim_budget = 'online'
task = sbibm.get_task(task_name)
conf_tuple = list(MODELS.values())[model_i]
model_name = list(MODELS.keys())[model_i]
print(model_name)

if task_name == 'lotka_volterra':
    # sbibm requires julia for this task
    simulator = bf.simulators.LotkaVolterra(subsample='original')

elif task_name == 'sir':
    # sbibm requires julia for this task
    simulator = bf.simulators.SIR(subsample='original')
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
    print(training_data['inference_variables'].shape, training_data['inference_conditions'].shape)
else:
    training_data = None
    print(f"Using online simulation.")

workflow = load_model(conf_tuple=conf_tuple,
                      training_data=training_data,
                      simulator=simulator,
                      storage=storage,
                      problem_name=task_name, model_name=model_name,
                      use_ema=True)

c2st_results = {'ode': [], 'sde': []}
for sampler in SAMPLER_SETTINGS.keys():
    if sampler.startswith('sde') and not model_name.startswith('diffusion'):
        continue
    for num_observation in range(1, 11):
        observation = task.get_observation(num_observation=num_observation).numpy()
        reference_samples = task.get_reference_posterior_samples(num_observation=num_observation)
        num_samples = reference_samples.shape[0]
        print(observation.shape, reference_samples.shape)
        if 'consistency' in model_name:
            posterior_samples_dict = workflow.sample(conditions={'inference_conditions': observation}, num_samples=num_samples)
        else:
            posterior_samples_dict = workflow.sample(conditions={'inference_conditions': observation}, num_samples=num_samples,
                                                     **SAMPLER_SETTINGS[sampler])
        posterior_samples = torch.as_tensor(posterior_samples_dict['inference_variables'][0])
        print(posterior_samples.shape)
        c2st_accuracy = c2st(reference_samples, posterior_samples).numpy().item()
        c2st_results[sampler].append(c2st_accuracy)
        print(f"{num_observation} C2ST accuracy: {c2st_accuracy}")

        # bf.diagnostics.pairs_posterior(
        #     estimates=posterior_samples.numpy(),
        #     priors=reference_samples.numpy(),
        # )
        # plt.show()
        # break

with open(f'{storage}c2st_results_{model_name}_{task_name}.pkl', 'wb') as f:
    pickle.dump(c2st_results, f)
print('Done.')
