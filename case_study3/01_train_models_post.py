import pickle
import numpy as np

import bayesflow as bf

from resnet import ResNetSummary

from keras.utils import clear_session
import matplotlib.pyplot as plt


def train_model(model_name, dataset_name, conf_tuple, data_dict):
    summary_network = ResNetSummary(
        summary_dim=8,
        widths=[16, 32],
        use_batch_norm=False,
        dropout=0.0
    )
    inference_net_kwargs = conf_tuple[1] | {"subnet_kwargs": SUBNET_KWARGS}
    inference_network = conf_tuple[0](**inference_net_kwargs)
    workflow = bf.workflows.BasicWorkflow(
        summary_network=summary_network,
        inference_network=inference_network,
        inference_variables=["log_std", "alpha"],
        summary_variables=["field"],
        standardize="summary_variables",
        checkpoint_filepath=f"checkpoints/{dataset_name}_{model_name}",
    )
    history = workflow.fit_offline(
        data=data_dict["train"],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=data_dict["validation"],
    )
    f = bf.diagnostics.plots.loss(history)
    f.savefig(f"figures/{dataset_name}_{model_name}_post_loss.png")
    plt.close(f)
    small_training_data = {k: v[:100] for k, v in data_dict["train"].items()}
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
            f.savefig(f"figures/{dataset_name}_{model_name}_{k}_{data_name}.png")
            plt.close(f)

def get_data_dict(dataset_name):
    if dataset_name == "grf_post":
        from FyeldGenerator import generate_field

        def generate_power_spectrum(alpha, scale):
            def power_spectrum(k):
                base = np.power(k, -alpha) * scale ** 2
                return base

            return power_spectrum

        rng = np.random.default_rng(seed=42)
        shape=(128, 128)
        def distribution(shape=shape):
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
    EPOCHS = 300
    BATCH_SIZE = 32
    NUM_SAMPLES_INFERENCE = 5000
    SUBNET_KWARGS = {"widths": 3 * (32,), "dropout": 0.0}
    MODELS = {
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

    DATASETS = ["grf_post"]
    import os
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    if not os.path.exists("figures"):
        os.mkdir("figures")
    for dataset in DATASETS:

        data_dict = get_data_dict(dataset)

        for model_name, conf_tuple in MODELS.items():
            bf.utils.logging.info(f"Training {model_name} on {dataset}...")

            train_model(model_name, dataset, conf_tuple, data_dict)
            clear_session()
