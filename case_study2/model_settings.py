import os
import numpy as np

import bayesflow as bf
import keras

from bayesflow.diagnostics.metrics import (root_mean_squared_error,
                                           posterior_contraction,
                                           calibration_error,
                                           classifier_two_sample_test)


EPOCHS = 1000
BATCH_SIZE = 64
NUM_SAMPLES_INFERENCE = 1000
MODELS = {
        ## TimeSeriesNetwork summary network
        "flow_matching": (bf.networks.FlowMatching, {}),
        "ot_flow_matching": (bf.networks.FlowMatching, {"use_optimal_transport": True}),
        "consistency_model": (bf.networks.ConsistencyModel, {"total_steps": EPOCHS*BATCH_SIZE}),
        "stable_consistency_model": (bf.experimental.StableConsistencyModel, {"embedding_kwargs": {"embed_dim": 2}}),
        "diffusion_edm_vp": (bf.networks.DiffusionModel, {
            "noise_schedule": "edm",
            "prediction_type": "F",
            "schedule_kwargs": {"variance_type": "preserving"}}),
        "diffusion_edm_ve": (bf.networks.DiffusionModel, {
            "noise_schedule": "edm",
            "prediction_type": "F",
            "schedule_kwargs": {"variance_type": "exploding"}}),
        "diffusion_cosine_F": (bf.networks.DiffusionModel, {
            "noise_schedule": "cosine",
            "prediction_type": "F", }),
        "diffusion_cosine_v": (bf.networks.DiffusionModel, {
            "noise_schedule": "cosine",
            "prediction_type": "velocity"}),
        "diffusion_cosine_noise": (bf.networks.DiffusionModel, {
            "noise_schedule": "cosine",
            "prediction_type": "noise"}),
        ## FusionTransformer summary network
        "flow_matching_ft": (bf.networks.FlowMatching, {}),
        "ot_flow_matching_ft": (bf.networks.FlowMatching, {"use_optimal_transport": True}),
        "consistency_model_ft": (bf.networks.ConsistencyModel, {"total_steps": EPOCHS * BATCH_SIZE}),
        "stable_consistency_model_ft": (bf.experimental.StableConsistencyModel, {"embedding_kwargs": {"embed_dim": 2}}),
        "diffusion_edm_vp_ft": (bf.networks.DiffusionModel, {
            "noise_schedule": "edm",
            "prediction_type": "F",
            "schedule_kwargs": {"variance_type": "preserving"}}),
        "diffusion_edm_ve_ft": (bf.networks.DiffusionModel, {
            "noise_schedule": "edm",
            "prediction_type": "F",
            "schedule_kwargs": {"variance_type": "exploding"}}),
        "diffusion_cosine_F_ft": (bf.networks.DiffusionModel, {
            "noise_schedule": "cosine",
            "prediction_type": "F", }),
        "diffusion_cosine_v_ft": (bf.networks.DiffusionModel, {
            "noise_schedule": "cosine",
            "prediction_type": "velocity"}),
        "diffusion_cosine_noise_ft": (bf.networks.DiffusionModel, {
            "noise_schedule": "cosine",
            "prediction_type": "noise"}),
    }


SAMPLER_SETTINGS = {
    'ode': {
        'method': 'rk45',
        'steps': 200
    },
    'sde': {
        'method': 'euler_maruyama',
        'steps': 200
    },
    'sde-pc': {
        'method': 'euler_maruyama',
        'steps': 200,
        'corrector_steps': 1,
    }
}

def load_model(adapter, conf_tuple, param_names, training_data, validation_data, storage, problem_name, model_name):
    if 'ft' in model_name:
        summary_network = bf.networks.FusionTransformer(summary_dim=len(param_names) * 2)
    else:
        summary_network = bf.networks.TimeSeriesNetwork(summary_dim=len(param_names) * 2,
                                                        recurrent_dim=256, recurrent_type='LSTM')

    workflow = bf.BasicWorkflow(
        adapter=adapter,
        training_data=training_data,
        validation_data=validation_data,
        summary_network=summary_network,
        inference_network=conf_tuple[0](**conf_tuple[1]),
        standardize='all'
    )
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
    return workflow


def compute_metrics(model_name, workflow, test_data, mcmc_posterior_samples, get_samples_from_dict):
    metrics = []
    for solver_name in SAMPLER_SETTINGS:
        if solver_name.startswith('sde') and not model_name.startswith('diffusion'):
            continue
        if 'consistency' in model_name:
            workflow_samples_dict = workflow.sample(conditions=test_data, num_samples=NUM_SAMPLES_INFERENCE)
        else:
            workflow_samples_dict = workflow.sample(conditions=test_data, num_samples=NUM_SAMPLES_INFERENCE,
                                                    **SAMPLER_SETTINGS[solver_name])
        workflow_samples = get_samples_from_dict(workflow_samples_dict)
        c2st_array = []
        for ws, ms in zip(workflow_samples, mcmc_posterior_samples):
            c2st_array.append(classifier_two_sample_test(ws, ms))
        metrics.append({
            'model': model_name,
            'sampler': solver_name,
            'nrmse': root_mean_squared_error(workflow_samples_dict, test_data)['values'].mean(),
            'posterior_contraction': posterior_contraction(workflow_samples_dict, test_data)['values'].mean(),
            'posterior_calibration_error': calibration_error(workflow_samples_dict, test_data)['values'].mean(),
            'c2st': np.mean(c2st_array)
        })
    return metrics
