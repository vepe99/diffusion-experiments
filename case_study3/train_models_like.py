import os
import pickle

import bayesflow as bf
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import clear_session

from cnn import ResNetSubnet


def train_model(model_kwargs, dataset_kwargs, conf_tuple, data_dict):
    model_name = model_kwargs["model_name"]
    proj_dir = os.path.join(f"{model_name}", "NLE")
    ckpt_dir = os.path.join(proj_dir, "checkpoints")
    figure_dir = os.path.join(proj_dir, "figures")
    os.makedirs(proj_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    shape = dataset_kwargs["shape"]
    model_config = "shape_config_8_16" if str(shape[0]) in "shape_config_8_16" else "shape_config_32_64_128_256"
    bf.utils.logging.info(f"Using model config: {model_config} for shape {shape}")
    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .rename("params_expanded", "inference_conditions")
        .rename("field", "inference_variables")
    )
    inference_net_kwargs = conf_tuple[1] | {
        "concatenate_subnet_input": False,
        "subnet":ResNetSubnet,
        "subnet_kwargs": model_kwargs[model_config]["subnet_kwargs"]
    }
    inference_network = conf_tuple[0](**inference_net_kwargs)
    workflow = bf.workflows.BasicWorkflow(
        inference_network=inference_network,
        adapter=adapter,
        standardize=None,
        checkpoint_filepath=ckpt_dir,
        checkpoint_name=f"{dataset_kwargs['shape'][0]}_{model_config}",
    )
    history = workflow.fit_offline(
        data=data_dict["train"],
        epochs=EPOCHS,
        validation_data=data_dict["validation"],
        batch_size=BATCH_SIZE,
    )

    f = bf.diagnostics.plots.loss(history)
    f.savefig(os.path.join(figure_dir, f"like_loss_{dataset_kwargs['shape'][0]}_{model_config}.png"))
    plt.close(f)



def get_data_dict(dataset_kwargs):
    dataset_name = dataset_kwargs["name"]
    if dataset_name == "grf_like":
        from FyeldGenerator import generate_field

        def generate_power_spectrum(alpha, scale):
            def power_spectrum(k):
                base = np.power(k, -alpha) * scale ** 2
                return base

            return power_spectrum

        rng = np.random.default_rng(seed=dataset_kwargs["seed"])
        shape=dataset_kwargs["shape"]
        def distribution(shape):
            a = rng.normal(loc=0, scale=1., size=shape)
            b = rng.normal(loc=0, scale=1., size=shape)
            return a + 1j * b

        def prior():
            log_std = rng.normal(scale=0.3)
            alpha = rng.normal(loc=3, scale=0.5)
            params_expanded = np.array([log_std, alpha])
            params_expanded = np.ones(shape + (2,)) * params_expanded[None, None, :]
            return {
                "log_std": log_std,
                "alpha": alpha,
                "params_expanded": params_expanded
            }

        def likelihood(log_std, alpha):
            field = generate_field(
                distribution, generate_power_spectrum(alpha, np.exp(log_std)), shape
            )
            return {"field": field[..., None]/50.}

        simulator = bf.make_simulator([prior, likelihood])
        data_dict = {
            "train": simulator.sample(NUM_SAMPLES_INFERENCE),
            "validation": simulator.sample(int(NUM_SAMPLES_INFERENCE*0.1)),
        }
    else:
        data_dict = {
            "train": pickle.load(open(f"data/{dataset_name}_train_data.pkl", "rb")),
            "validation": pickle.load(open(f"data/{dataset_name}_validation_data.pkl", "rb")),
        }
    return data_dict


if __name__ == "__main__":
    EPOCHS = 500
    BATCH_SIZE = 32
    NUM_SAMPLES_INFERENCE = 5000
    INTEGRATE_KWARGS = {"method": "rk45", "steps": 200}
    shapes = [(2**n, 2**n) for n in range(3, 9)]
    seed = 42
    model_kwargs = {
        "shape_config_32_64_128_256": {
            "subnet_kwargs": {
                "widths": 3*(32,),
                "activation": "mish",
            },
        },
        "shape_config_8_16": {
            "subnet_kwargs": {
                "widths": 3*(32,),
                "activation": "mish",
            },
        }
    }

    MODELS_extended = {
        "flow_matching": (bf.networks.FlowMatching, {
        }),
        "ot_flow_matching": (bf.networks.FlowMatching, {
            "use_optimal_transport": True}),
        "consistency_model": (bf.networks.ConsistencyModel, {
            "total_steps": EPOCHS * BATCH_SIZE}),
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
    }

    MODELS = {
        "flow_matching": (bf.networks.FlowMatching, {
            "integrate_kwargs": INTEGRATE_KWARGS,
        }),
        "consistency_model": (bf.networks.ConsistencyModel, {
            "total_steps": EPOCHS * BATCH_SIZE}),
        "diffusion_edm_vp": (bf.networks.DiffusionModel, {
            "noise_schedule": "edm",
            "prediction_type": "F",
            "schedule_kwargs": {"variance_type": "preserving"},
            "integrate_kwargs": INTEGRATE_KWARGS,
        }),
    }

    DATASETS = ["grf_like"]
    for dataset in DATASETS:
        for shape in shapes:
            dataset_kwargs = {
                "shape": shape,
                "seed": seed,
                "name": dataset,
            }
            data_dict = get_data_dict(dataset_kwargs)
            for model_name, conf_tuple in MODELS.items():
                current_model_kwargs = {"model_name": model_name} | model_kwargs
                bf.utils.logging.info(f"Training {model_name} on {dataset}...")
                train_model(current_model_kwargs, dataset_kwargs, conf_tuple, data_dict)
                clear_session()
