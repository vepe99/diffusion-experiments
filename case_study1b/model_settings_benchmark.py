import os
os.environ["KERAS_BACKEND"] = "torch"

import bayesflow as bf
import keras

from ema_callback import EMA, save_ema_models


EPOCHS = 1_000
BATCH_SIZE = 128
NUM_BATCHES_PER_EPOCH = 32_000 // BATCH_SIZE  # for online training

SUBNET_KWARGS = {
    "widths": (256, 256, 256, 256, 256),
}

MODELS = {
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

NUM_STEPS_SAMPLER = 500
MIN_STEPS = 50
SAMPLER_SETTINGS = {
    'ode-euler': {
        'method': 'euler',
        'steps': NUM_STEPS_SAMPLER,
        'min_steps': MIN_STEPS
    },
    'ode-rk45': {
        'method': 'rk45',
        'steps': NUM_STEPS_SAMPLER,
        'min_steps': MIN_STEPS
    },
    'ode-tsit5': {
        'method': 'tsit5',
        'steps': NUM_STEPS_SAMPLER,
        'min_steps': MIN_STEPS
    },
    'ode-rk45-adaptive': {
        'method': 'rk45',
        'steps': "adaptive",
        'min_steps': MIN_STEPS
    },
    'ode-tsit5-adaptive': {
        'method': 'tsit5',
        'steps': "adaptive",
        'min_steps': MIN_STEPS
    },
    'sde-euler': {
        'method': 'euler_maruyama',
        'steps': NUM_STEPS_SAMPLER,
        'min_steps': MIN_STEPS
    },
    'sde-euler-adaptive': {
        'method': 'euler_maruyama',
        'steps': "adaptive",
        'min_steps': MIN_STEPS
    },
    'sde-euler-pc': {
        'method': 'euler_maruyama',
        'steps': NUM_STEPS_SAMPLER,
        'corrector_steps': 1,
        'min_steps': MIN_STEPS
    },
    'sde-sea': {
        'method': 'sea',
        'steps': NUM_STEPS_SAMPLER,
        'min_steps': MIN_STEPS
    },
    'sde-shark': {
        'method': 'shark',
        'steps': NUM_STEPS_SAMPLER,
        'min_steps': MIN_STEPS
    },
    'sde-two_step-adaptive': {
        'method': 'two_step_adaptive',
        'steps': "adaptive",
        'min_steps': MIN_STEPS
    },
    'sde-langevin': {
        'method': 'langevin',
        'steps': NUM_STEPS_SAMPLER * 4,
        'min_steps': MIN_STEPS
    },
    'sde-langevin-pc': {
        'method': 'langevin',
        'steps': NUM_STEPS_SAMPLER * 4,
        'corrector_steps': 5,
        'min_steps': MIN_STEPS
    },
}


ADAPTER_SETTINGS = {
    'lotka_volterra': 'log',
    'gaussian_mixture': (-10, 10),
    'gaussian_linear_uniform': (-1, 1),
    'two_moons': (-1, 1),
    'bernoulli_glm': None,
    'sir': 'log',
    'gaussian_linear': None,
    'slcp': (-3, 3),
    'slcp_distractors': (-3, 3),
    'bernoulli_glm_raw': None
}


def create_adapter(config):
    if config is None:
        return (
            bf.adapters.Adapter()
            .to_array()
            .convert_dtype("float64", "float32")
            .rename('parameters', 'inference_variables')
            .rename('observables', 'inference_conditions')
        )
    elif config == 'log':
        return (
            bf.adapters.Adapter()
            .to_array()
            .convert_dtype("float64", "float32")
            .log("parameters")
            .rename('parameters', 'inference_variables')
            .rename('observables', 'inference_conditions')
        )
    elif isinstance(config, tuple) and len(config) == 2:
        return (
            bf.adapters.Adapter()
            .to_array()
            .convert_dtype("float64", "float32")
            .constrain("parameters", lower=config[0], upper=config[1], epsilon=1e-6, inclusive='both')
            .rename('parameters', 'inference_variables')
            .rename('observables', 'inference_conditions')
        )
    else:
        raise ValueError("Unknown adapter configuration")


def load_model(conf_tuple, simulator, training_data, storage, problem_name, model_name, sum_dim=0, use_ema=True):
    if sum_dim > 0:
        summary_network = bf.networks.TimeSeriesNetwork(summary_dim=sum_dim)
    else:
        summary_network = None

    adapter = create_adapter(ADAPTER_SETTINGS[problem_name])

    workflow = bf.BasicWorkflow(
        adapter=adapter,
        simulator=simulator,
        summary_network=summary_network,
        inference_network=conf_tuple[0](**conf_tuple[1]),
        standardize='all'
    )

    model_path = f'{storage}benchmark_{problem_name}_{model_name}.keras'
    model_path_ema = f'{storage}benchmark_{problem_name}_{model_name}_ema.keras'
    if 'ema' in model_name:
        cbs = [EMA()]
    else:
        cbs = None


    if not os.path.exists(model_path):
        if simulator is not None:
            workflow.fit_online(
                num_batches_per_epoch=NUM_BATCHES_PER_EPOCH,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=2,
                callbacks=cbs
            )
        else:
            workflow.fit_offline(
                training_data,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=2,
                callbacks=cbs
            )
        workflow.approximator.save(model_path)

        if 'ema' in model_name:
            save_ema_models(workflow.approximator, cbs[0], path_ema=model_path_ema, path_noema=model_path)
            workflow.approximator = keras.models.load_model(model_path_ema)
    else:
        if use_ema and 'ema' in model_name:
            workflow.approximator = keras.models.load_model(model_path_ema)
        else:
            workflow.approximator = keras.models.load_model(model_path)
    return workflow
