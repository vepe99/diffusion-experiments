import os
os.environ["KERAS_BACKEND"] = "torch"

import bayesflow as bf
import keras

from ema_callback import EMA, save_ema_models


EPOCHS = 10#00
BATCH_SIZE = 128
NUM_SAMPLES_INFERENCE = 1000

SUBNET_KWARGS = {
    "widths": (256, 256, 256, 256, 256),
}

MODELS = {
        ## TimeSeriesNetwork summary network
        "flow_matching": (bf.networks.FlowMatching, {"subnet_kwargs": SUBNET_KWARGS}),
        "flow_matching_edm": (bf.networks.FlowMatching, {'time_power_law_alpha': -0.6, "subnet_kwargs": SUBNET_KWARGS}),
        "ot_flow_matching": (bf.networks.FlowMatching, {"use_optimal_transport": True, "subnet_kwargs": SUBNET_KWARGS}),
        "consistency_model": (bf.networks.ConsistencyModel, {"total_steps": EPOCHS*BATCH_SIZE, "subnet_kwargs": SUBNET_KWARGS}),
        "stable_consistency_model": (bf.experimental.StableConsistencyModel, {"subnet_kwargs": SUBNET_KWARGS}),
        "diffusion_edm_vp": (bf.networks.DiffusionModel, {
            "noise_schedule": "edm",
            "prediction_type": "F",
            "schedule_kwargs": {"variance_type": "preserving"},
            "subnet_kwargs": SUBNET_KWARGS}),
        "diffusion_edm_vp_ema": (bf.networks.DiffusionModel, {
                    "noise_schedule": "edm",
                    "prediction_type": "F",
                    "schedule_kwargs": {"variance_type": "preserving"},
            "subnet_kwargs": SUBNET_KWARGS}),
        "diffusion_edm_ve": (bf.networks.DiffusionModel, {
            "noise_schedule": "edm",
            "prediction_type": "F",
            "schedule_kwargs": {"variance_type": "exploding"},
            "subnet_kwargs": SUBNET_KWARGS}),
        "diffusion_cosine_F": (bf.networks.DiffusionModel, {
            "noise_schedule": "cosine",
            "prediction_type": "F", }),
        "diffusion_cosine_v": (bf.networks.DiffusionModel, {
            "noise_schedule": "cosine",
            "prediction_type": "velocity",
            "subnet_kwargs": SUBNET_KWARGS}),
        "diffusion_cosine_noise": (bf.networks.DiffusionModel, {
            "noise_schedule": "cosine",
            "prediction_type": "noise",
            "subnet_kwargs": SUBNET_KWARGS}),
    }


SAMPLER_SETTINGS = {
    'ode': {
        'method': 'rk45',
        'steps': 250
    },
    'sde': {
        'method': 'euler_maruyama',
        'steps': 250
    }
}

def load_model(adapter, conf_tuple, sum_dim, training_data, storage, problem_name, model_name, use_ema=True):
    if sum_dim > 0:
        summary_network = bf.networks.TimeSeriesNetwork(summary_dim=sum_dim)
    else:
        summary_network = None

    workflow = bf.BasicWorkflow(
        adapter=adapter,
        training_data=training_data,
        summary_network=summary_network,
        inference_network=conf_tuple[0](**conf_tuple[1]),
        inference_variables='inference_variables',
        summary_variables='summary_variables' if sum_dim > 0 else None,
        inference_conditions='inference_conditions' if sum_dim == 0 else None,
    )

    model_path = f'{storage}benchmark_{problem_name}_{model_name}.keras'
    model_path_ema = f'{storage}benchmark_{problem_name}_{model_name}_ema.keras'
    if 'ema' in model_name:
        cbs = [EMA()]
    else:
        cbs = None

    if not os.path.exists(model_path):
        workflow.fit_offline(
            training_data,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2,
            callbacks=cbs
        )
        #workflow.approximator.save(model_path)

        if 'ema' in model_name:
            save_ema_models(workflow.approximator, cbs[0], path_ema=model_path_ema, path_noema=model_path)
            workflow.approximator = keras.models.load_model(model_path_ema)
    else:
        if use_ema and 'ema' in model_name:
            workflow.approximator = keras.models.load_model(model_path_ema)
        else:
            workflow.approximator = keras.models.load_model(model_path)
    return workflow
