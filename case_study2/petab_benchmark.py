#%% md
# # PEtab benchmark model with BayesFlow
#%%
# pip install git+https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab.git@master#subdirectory=src/python
# pypesto, amici, petab, fides, joblib
#%%
import os
import sys
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"
else:
    print(f"Using '{os.environ['KERAS_BACKEND']}' backend")

import numpy as np
import pandas as pd
from copy import deepcopy
import pickle
from joblib import Parallel, delayed
from typing import Union
from collections import defaultdict

import benchmark_models_petab as benchmark_models
import petab
import pypesto.optimize as optimize
import pypesto.sample as sample
import pypesto.petab
from scipy import stats

import keras
import bayesflow as bf

import amici
import logging
amici.swig_wrappers.logger.setLevel(logging.CRITICAL)
pypesto.logging.log(level=logging.ERROR, name="pypesto.petab", console=True)

from petab_helper import scale_values, values_to_linear_scale, amici_pred_to_array, apply_noise_to_data

# print all model names
print(benchmark_models.MODELS)
# generate petab problem
job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
problem_name = sys.argv[1] if len(sys.argv) > 1 else "Beer_MolBioSystems2014"
print(problem_name)
storage = f'plots/{problem_name}/'
petab_problem = benchmark_models.get_problem(problem_name)

# decrease upper bounds for offset, scaling and noise parameters
scale_params_id = [name for name in petab_problem.parameter_df.index.values if name[:6] == 'offset' or name[:5] == 'scale']
petab_problem.parameter_df.loc[scale_params_id, 'upperBound'] = 100  # instead of 1000
sd_params_id = [name for name in petab_problem.parameter_df.index.values if name[:3] == 'sd_']
petab_problem.parameter_df.loc[sd_params_id, 'upperBound'] = 10  # instead of 1000

if problem_name == "Raimundez_PCB2020":
    # Elba added normal priors for the scaling params
    scale_params_id = [name for name in petab_problem.parameter_df.index.values if name[:2] == 's_']
    petab_problem.parameter_df.loc[scale_params_id, 'objectivePriorType'] = "normal"
    petab_problem.parameter_df.loc[scale_params_id, 'objectivePriorParameters'] = "1;10"
    petab_problem.parameter_df.loc[scale_params_id, 'parameterScale'] = "lin"

# add normal prior (on scale) around real parameters values
real_data_params = petab_problem.parameter_df.nominalValue
real_data_params = real_data_params[petab_problem.parameter_df['estimate'] == 1]
std = 0.5
for i in real_data_params.index:
    if petab_problem.parameter_df.loc[i, 'estimate'] == 0:
        continue
    # set prior mean depending on scale
    if petab_problem.parameter_df.loc[i, 'parameterScale'] == 'log':
        mean = np.log(real_data_params.loc[i])
    elif petab_problem.parameter_df.loc[i, 'parameterScale'] == 'log10':
        mean = np.log10(real_data_params.loc[i])
    else:
        mean = real_data_params.loc[i]
    if not 'objectivePriorType' in petab_problem.parameter_df or pd.isna(petab_problem.parameter_df.loc[i, 'objectivePriorType']):
        petab_problem.parameter_df.loc[i, 'objectivePriorType'] = "parameterScaleNormal"
        petab_problem.parameter_df.loc[i, 'objectivePriorParameters'] = f"{mean};{std}"


for i, row in petab_problem.parameter_df.iterrows():
    if 'objectivePriorType' in row and not pd.isna(row['objectivePriorType']):
        if row['estimate'] == 0:
            print(f"Parameter {i} has a {row['objectivePriorType']} prior but is not estimated, setting to nan")
            petab_problem.parameter_df.loc[i, 'objectivePriorType'] = np.nan
        # validate petab problem, if scale for parameter is defined, prior must be on the same scale
        if row['parameterScale'] != 'lin' and not row['objectivePriorType'].startswith('parameterScale'):
            raise ValueError(f"Parameter {i} has parameterScale {row['parameterScale']} but {row['objectivePriorType']} prior")

# load problem
importer = pypesto.petab.PetabImporter(petab_problem, simulator_type="amici")
factory = importer.create_objective_creator()

model = factory.create_model(verbose=False)
amici_predictor = factory.create_predictor()
amici_predictor.amici_objective.amici_solver.setAbsoluteTolerance(1e-8)

# Creating the pypesto problem from PEtab
pypesto_problem = importer.create_problem(
    startpoint_kwargs={"check_fval": True, "check_grad": True}
)

#%%
def prior():
    lb = petab_problem.parameter_df['lowerBound'].values
    ub = petab_problem.parameter_df['upperBound'].values
    param_names_id = petab_problem.parameter_df.index.values
    param_scale = petab_problem.parameter_df['parameterScale'].values
    if 'objectivePriorType' in petab_problem.parameter_df.columns:
        prior_type = petab_problem.parameter_df['objectivePriorType'].values
    else:
        prior_type = [np.nan] * len(param_names_id)
    estimate_param = petab_problem.parameter_df['estimate'].values

    prior_dict = {}
    for i, name in enumerate(param_names_id):
        if estimate_param[i] == 0:
            prior_dict[name] = petab_problem.parameter_df['nominalValue'].values[i]  # linear space
        elif prior_type[i] == 'uniform':  # linear space
            prior_dict[name] = np.random.uniform(low=lb[i], high=ub[i])
        elif prior_type[i] == 'parameterScaleUniform' or pd.isna(prior_type[i]):
            # scale bounds to scaled space
            lb_scaled_i = scale_values(lb[i], param_scale[i])
            ub_scaled_i = scale_values(ub[i], param_scale[i])
            val = np.random.uniform(low=lb_scaled_i, high=ub_scaled_i)
            # scale to linear space
            prior_dict[name] = values_to_linear_scale(val, param_scale[i])
        elif prior_type[i] == 'parameterScaleNormal':
            mean, std = petab_problem.parameter_df['objectivePriorParameters'].values[i].split(';')
            lb_scaled_i = scale_values(lb[i], param_scale[i])
            ub_scaled_i = scale_values(ub[i], param_scale[i])
            a, b = (lb_scaled_i - float(mean)) / float(std), (ub_scaled_i - float(mean)) / float(std)
            rv = stats.truncnorm.rvs(loc=float(mean), scale=float(std), a=a, b=b)
            # scale to linear space
            prior_dict[name] = values_to_linear_scale(rv, param_scale[i])
        elif prior_type[i] == 'normal':
            mean, std = petab_problem.parameter_df['objectivePriorParameters'].values[i].split(';')
            a, b = (lb[i] - float(mean)) / float(std), (ub[i] - float(mean)) / float(std)
            rv = stats.truncnorm.rvs(loc=float(mean), scale=float(std), a=a, b=b)
            prior_dict[name] = rv
        elif prior_type[i] == 'laplace':
            loc, scale = petab_problem.parameter_df['objectivePriorParameters'].values[i].split(';')
            for t in range(10):
                rv = np.random.laplace(loc=float(loc), scale=float(scale))
                if lb[i] <= rv <= ub[i]:  # sample from truncated laplace
                    break
            prior_dict[name] = rv
        else:
            raise ValueError("Unknown prior type:", prior_type[i])
        # scale params and make list
        prior_dict[name] = np.array([scale_values(prior_dict[name], param_scale[i])])

    # prepare variables for simulation
    x = np.array([prior_dict[name][0] for name in pypesto_problem.x_names])
    prior_dict['amici_params'] = x  # scaled parameters for amici
    return prior_dict

def simulator_amici(amici_params):
    pred = amici_predictor(amici_params)  # expect amici_params to be scaled
    sim, failed = amici_pred_to_array(pred, amici_params,
                                      factory=factory, petab_problem=petab_problem, pypesto_problem=pypesto_problem)
    return dict(sim_data=sim, sim_failed=failed)
#%%
prior_sample = prior()
test = simulator_amici(prior_sample['amici_params'])
print(test['sim_data'].shape, prior_sample['amici_params'].shape, np.nansum(test['sim_data']))

#%%
def run_mcmc(petab_problem, pypesto_problem, true_params=None, n_optimization_starts=0, n_chains=10, n_samples=10000,
             n_procs=10, verbose=False) -> Union[pypesto.result.Result, tuple[pypesto.result.Result, petab.Problem, pypesto.Problem]]:
    _petab_problem = deepcopy(petab_problem)
    if true_params is None:
        # use true data
        pass
    else:
        # this is needed to create a new measurement df and recompile the problem for amici
        pred = amici_predictor(true_params)
        _, failed = amici_pred_to_array(pred, true_params,
                                      factory=factory, petab_problem=petab_problem, pypesto_problem=pypesto_problem)
        if failed:
            print("Simulation failed for true parameters")
            return None, None, None
        _measurement_df = factory.prediction_to_petab_measurement_df(pred) # to create new measurement df
        _measurement_df = apply_noise_to_data(_measurement_df, true_params, field='measurement',
                                              pypesto_problem=pypesto_problem, petab_problem=_petab_problem)
        _petab_problem.measurement_df = _measurement_df
    _importer = pypesto.petab.PetabImporter(_petab_problem, simulator_type="amici")
    _factory = _importer.create_objective_creator()
    _model = _factory.create_model(verbose=False)

    _pypesto_problem = _importer.create_problem(
        startpoint_kwargs={"check_fval": True, "check_grad": True}
    )

    if isinstance(_pypesto_problem.objective, pypesto.objective.AggregatedObjective):
        _pypesto_problem.objective._objectives[0].amici_solver.setAbsoluteTolerance(1e-8)
        #_pypesto_problem.objective._objectives[0].amici_solver.setSensitivityMethod(amici.SensitivityMethod.adjoint)
    else:
        _pypesto_problem.objective.amici_solver.setAbsoluteTolerance(1e-8)
        #_pypesto_problem.objective.amici_solver.setSensitivityMethod(amici.SensitivityMethod.adjoint)

    if n_optimization_starts == 0:
        print("Skipping optimization, sample start points for chains from prior")
        _result = None
        x0 = [_pypesto_problem.get_reduced_vector(prior()['amici_params']) for _ in range(n_chains)]
    else:
        # do the optimization
        _result = optimize.minimize(
            problem=_pypesto_problem,
            optimizer=optimize.FidesOptimizer(verbose=0),
            #optimizer=optimize.ScipyOptimizer(method='L-BFGS-B'),
            n_starts=n_optimization_starts,
            engine=pypesto.engine.MultiProcessEngine(n_procs=n_procs) if n_procs > 1 else None,
            progress_bar=verbose
        )
        x0 = [_pypesto_problem.get_reduced_vector(_result.optimize_result.x[0])]
        if x0[0] is None:
            print("Warning: x0 contains nan, replace with prior sample")
            x0[0] = _pypesto_problem.get_reduced_vector(prior()['amici_params'])
        x0 += [_pypesto_problem.get_reduced_vector(prior()['amici_params']) for _ in range(n_chains - 1)]

    _sampler = sample.AdaptiveParallelTemperingSampler(
        internal_sampler=sample.AdaptiveMetropolisSampler(
            options=dict(decay_constant=0.7, threshold_sample=2000)
        ),
        n_chains=n_chains,
        options=dict(show_progress=verbose)
    )

    _result = sample.sample(
        problem=_pypesto_problem,
        n_samples=n_samples,
        sampler=_sampler,
        result=_result,
        x0=x0
    )
    sample.geweke_test(_result)

    if true_params is None:
        return _result
    return _result, _petab_problem, _pypesto_problem
#%%
def get_mcmc_posterior_samples(res):
    burn_in = sample.geweke_test(res)
    if burn_in == res.sample_result.trace_x.shape[1]:
        print("Warning: All samples are considered burn-in.")
        _samples = res.sample_result.trace_x[0]  # only use first chain
    else:
        _samples = res.sample_result.trace_x[0, burn_in:]  # only use first chain
    #_samples = pypesto_problem.get_full_vector(_samples)
    #scales = petab_problem.parameter_df.loc[res.problem.x_names, 'parameterScale'].values
    #_samples = values_to_linear_scale(_samples, scales)
    return _samples

# # BayesFlow workflow
#%%
simulator = bf.make_simulator([prior, simulator_amici])
#%%
num_training_batches = 512
num_validation_sets = 100
batch_size = 64
epochs = 100#0 todo
#%%
@delayed
def sample_and_simulate():
    """Single iteration of sampling and simulation"""
    prior_sample = prior()
    test = simulator_amici(prior_sample['amici_params'])

    # Combine both dictionaries
    result = {**prior_sample, **test}
    return result

def simulate_parallel(n_samples):
    """Parallel sampling and simulation"""
    results = Parallel(n_jobs=n_cpus, verbose=10)(
        sample_and_simulate() for _ in range(n_samples)
    )
    results_dict = defaultdict(list)

    for r in results:
        for key, value in r.items():
            results_dict[key].append(value)
    for key, value_list in results_dict.items():
        results_dict[key] = np.array(value_list)
    return results_dict
#%%
if os.path.exists(f"{storage}validation_data_petab_{problem_name}.pkl"):
    with open(f'{storage}validation_data_petab_{problem_name}.pkl', 'rb') as f:
        validation_data = pickle.load(f)
    try:
        with open(f'{storage}training_data_petab_{problem_name}.pkl', 'rb') as f:
            training_data = pickle.load(f)
    except FileNotFoundError:
        training_data = None
        print("Training data not found")
else:
    training_data = simulate_parallel(num_training_batches * batch_size)
    validation_data = simulate_parallel(num_validation_sets)

    with open(f'{storage}training_data_petab_{problem_name}.pkl', 'wb') as f:
        pickle.dump(training_data, f)
    with open(f'{storage}validation_data_petab_{problem_name}.pkl', 'wb') as f:
        pickle.dump(validation_data, f)
    exit()

# remove failed simulations
train_mask = ~training_data['sim_failed']
for key in training_data.keys():
    training_data[key] = training_data[key][train_mask]
val_mask = ~validation_data['sim_failed']
for key in validation_data.keys():
    validation_data[key] = validation_data[key][val_mask]
print(f"Failed Training data: {np.sum(~train_mask)} / {len(train_mask)}, "
      f"Failed Validation data: {np.sum(~val_mask)} / {len(val_mask)}")

test_mean = np.nanmean(np.log(validation_data['sim_data']+1), axis=(0,1), keepdims=True)
test_std = np.nanstd(np.log(validation_data['sim_data']+1), axis=(0,1), keepdims=True)
print(validation_data['sim_data'].shape)
#%%
param_names = [name for i, name in enumerate(pypesto_problem.x_names) if i in pypesto_problem.x_free_indices]
lbs = np.array([lb for i, lb in enumerate(petab_problem.lb_scaled) if i in pypesto_problem.x_free_indices])
ubs = np.array([ub for i, ub in enumerate(petab_problem.ub_scaled) if i in pypesto_problem.x_free_indices])

num_samples = 1000
#%% md
# # MCMC sampling for comparison
#%%
def run_mcmc_single(petab_prob, pypesto_prob, true_params, n_starts, n_mcmc_samples, n_final_samples, n_chains):
    import amici
    import logging
    amici.swig_wrappers.logger.setLevel(logging.CRITICAL)
    pypesto.logging.log(level=logging.ERROR, name="pypesto.petab", console=True)

    try:
        r, _, _ = run_mcmc(
            petab_problem=petab_prob,
            pypesto_problem=pypesto_prob,
            true_params=true_params,
            n_optimization_starts=n_starts,
            n_samples=n_mcmc_samples,
            n_chains=n_chains,
            n_procs=n_cpus
        )
    except np.linalg.LinAlgError as e:
        print("LinAlgError during MCMC:", e)
        return np.full((n_final_samples, len(pypesto_prob.x_free_indices)), np.nan)

    if r is None:
        return np.full((n_final_samples, len(pypesto_prob.x_free_indices)), np.nan)

    ps = get_mcmc_posterior_samples(r)
    # num_samples random samples from posterior
    idx = np.random.choice(ps.shape[0], size=n_final_samples)
    return ps[idx]

# mcmc_path = f'{storage}mcmc_samples_{problem_name}_{job_id}.pkl'
# if not os.path.exists(mcmc_path):
#     mcmc_posterior_samples = run_mcmc_single(
#                 petab_prob=petab_problem,
#                 pypesto_prob=pypesto_problem,
#                 true_params=validation_data['amici_params'][job_id],
#                 n_starts=10,
#                 n_mcmc_samples=1e5,
#                 n_final_samples=num_samples,
#                 n_chains=10
#     )
#
#     with open(mcmc_path, 'wb') as f:
#         pickle.dump(mcmc_posterior_samples, f)

#%%
adapter = (
    bf.adapters.Adapter()
    .drop('amici_params')  # only used for simulation
    .to_array()
    .convert_dtype("float64", "float32")
    .concatenate(param_names, into="inference_variables")
    .constrain("inference_variables", lower=lbs, upper=ubs, inclusive='both')  # after concatenate such that we can apply an array as constraint

    .as_time_series("sim_data")
    .log("sim_data", p1=True)
    #.standardize("sim_data", mean=test_mean, std=test_std)
    #.nan_to_num("sim_data", default_value=-3.0)
    .rename("sim_data", "summary_variables")
)

#%%
from model_settings import EPOCHS, BATCH_SIZE, MODELS, NUM_SAMPLES_INFERENCE, SAMPLER_SETTINGS
model_name = list(MODELS.keys())[job_id]
conf_tuple = MODELS[model_name]
print(model_name)

workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    summary_network=bf.networks.FusionTransformer(summary_dim=len(param_names)*2),  # FusionTransformer
    inference_network=conf_tuple[0](**conf_tuple[1]),
    standardize='all'
)
#%%
model_path = f'{storage}petab_benchmark_diffusion_model_{problem_name}_{model_name}.keras'
if not os.path.exists(model_path):
    history = workflow.fit_offline(
        training_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=validation_data,
        verbose=2
    )
    workflow.approximator.save(model_path)
else:
    workflow.approximator = keras.models.load_model(model_path)
#%%
diagnostics_plots = workflow.plot_default_diagnostics(test_data=validation_data, num_samples=NUM_SAMPLES_INFERENCE,
                                                      calibration_ecdf_kwargs={"difference": True, 'stacked': True})
for k in diagnostics_plots.keys():
    diagnostics_plots[k].savefig(f"{storage}petab_benchmark_{problem_name}_{model_name}_{k}.pdf")

if model_name.startswith('diffusion'):
    for solver_name in SAMPLER_SETTINGS:
        diagnostics = workflow.compute_default_diagnostics(test_data=validation_data, num_samples=NUM_SAMPLES_INFERENCE, approximator_kwargs=SAMPLER_SETTINGS[solver_name])
        print(solver_name)
        print(diagnostics.median(axis=1))
        print('\n')

print('Done')
