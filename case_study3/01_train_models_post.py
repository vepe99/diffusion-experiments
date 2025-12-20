import os
import pickle

import bayesflow as bf
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import clear_session

from resnet import ResNetSummary


def train_model(model_kwargs, dataset_kwargs, conf_tuple, data_dict):
    model_name = model_kwargs["model_name"]
    proj_dir = os.path.join(f"{model_name}", "NPE")
    ckpt_dir = os.path.join(proj_dir, "checkpoints")
    figure_dir = os.path.join(proj_dir, "figures")
    os.makedirs(proj_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    shape = dataset_kwargs["shape"]
    model_config = "shape_config_8_16" if str(shape[0]) in "shape_config_8_16" else "shape_config_32_64_128_256"
    bf.utils.logging.info(f"Using model config: {model_config} for shape {shape}")
    summary_network = ResNetSummary(**model_kwargs[model_config]["summary_kwargs"])
    inference_net_kwargs = conf_tuple[1] | {"subnet_kwargs": model_kwargs[model_config]["subnet_kwargs"]}
    inference_network = conf_tuple[0](**inference_net_kwargs)
    workflow = bf.workflows.BasicWorkflow(
        summary_network=summary_network,
        inference_network=inference_network,
        inference_variables=["log_std", "alpha"],
        summary_variables=["field"],
        standardize="summary_variables",
        checkpoint_filepath=ckpt_dir,
        checkpoint_name=f"{dataset_kwargs["shape"][0]}_{model_config}",
    )
    history = workflow.fit_offline(
        data=data_dict["train"],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=data_dict["validation"],
    )
    f = bf.diagnostics.plots.loss(history)
    f.savefig(os.path.join(figure_dir, f"post_loss_{dataset_kwargs["shape"][0]}_{model_config}.png"))
    plt.close(f)
    small_training_data = {k: v[:len(data_dict["validation"]["field"])] for k, v in data_dict["train"].items()}
    for data_name, test_data  in zip(["train", "validation"], [small_training_data, data_dict["validation"]]):
        plot_fns = {
            "recovery": bf.diagnostics.recovery,
            "calibration": bf.diagnostics.calibration_ecdf,
        }
        fs = workflow.plot_custom_diagnostics(
            test_data=test_data,
            plot_fns=plot_fns,
        )
        for k, f in fs.items():
            f.savefig(os.path.join(figure_dir, f"{k}_{data_name}_{dataset_kwargs["shape"][0]}_{model_config}.png"))
            plt.close(f)

        targets = {"alpha": test_data["alpha"], "log_std": test_data["log_std"]}
        # Numbers
        wf_samples = workflow.sample(num_samples=1000, conditions={"field": test_data["field"]})
        nrmse = bf.diagnostics.metrics.root_mean_squared_error(targets=targets, estimates=wf_samples)
        ce = bf.diagnostics.metrics.calibration_error(targets=targets, estimates=wf_samples)  # 0 is perfect [0, 0.5]
        clg = bf.diagnostics.metrics.calibration_log_gamma(targets=targets, estimates=wf_samples)
        validation_dict = {
            "nrmse": nrmse,
            "ce": ce,
            "clg": clg,
        }
        validation_path = os.path.join(proj_dir, f"numbers_{data_name}_{dataset_kwargs["shape"][0]}_{model_config}.npz")
        kv = {}
        for k, m in validation_dict.items():
            v = np.asarray(m["values"]).astype(float)
            v[~np.isfinite(v)] = np.nan  # keep file valid
            kv[f"{k}_values"] = v
            kv[f"{k}_names"] = np.asarray(m["variable_names"], dtype=object)
            kv[f"{k}_metric_name"] = np.asarray(m["metric_name"], dtype=object)
        np.savez_compressed(validation_path, **kv)


def get_data_dict(dataset_kwargs):
    dataset_name = dataset_kwargs["name"]
    if dataset_name == "grf_post":
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
            return {"log_std": rng.normal(scale=0.3), "alpha": rng.normal(loc=3, scale=0.5)}

        def likelihood(log_std, alpha):
            field = generate_field(
                distribution, generate_power_spectrum(alpha, np.exp(log_std)), shape
            )
            return {"field": field[..., None]}

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
    shapes = [(2**n, 2**n) for n in range(3, 5)]
    seed = 42
    model_kwargs = {
        "shape_config_32_64_128_256": {
            "summary_kwargs": {
                "summary_dim": 8,
                "widths": [16, 32],
                "use_batch_norm": False,
                "dropout": 0.0,
            },
            "subnet_kwargs": {
                "widths": 3 * (32,),
                "dropout": 0.0,
            },
        },
        "shape_config_8_16": {
            "summary_kwargs": {
                "summary_dim": 4,
                "widths": 2*(8,),
                "use_batch_norm": False,
                "dropout": 0.0,
            },
            "subnet_kwargs": {
                "widths": 3 * (32,),
                "dropout": 0.0,
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

    DATASETS = ["grf_post"]
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
