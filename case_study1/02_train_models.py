import pickle
import numpy as np

import bayesflow as bf

from keras.utils import clear_session


EPOCHS = 100
BATCH_SIZE = 128
NUM_SAMPLES_INFERENCE = 1000
MODELS = {
        "flow_matching": (bf.networks.FlowMatching, {}),
        "ot_flow_matching": (bf.networks.FlowMatching, {"use_optimal_transport": True}),
        "consistency_model": (bf.networks.ConsistencyModel, {"total_steps": EPOCHS*BATCH_SIZE}),
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

DATASETS = ["two_moons", "inverse_kinematics"]

def train_model(model_name, dataset_name, conf_tuple, data):

    adapter = (
        bf.adapters.Adapter()
        .rename("parameters", "inference_variables")
        .rename("observables", "inference_conditions")
    )

    inference_net = conf_tuple[0](**conf_tuple[1])

    workflow = bf.workflows.BasicWorkflow(
        adapter=adapter,
        inference_network=inference_net,
        checkpoint_filepath=f"checkpoints/{model_name}_{dataset_name}",
        standardize="all"
    )

    _ = workflow.fit_offline(
        data=data["train"],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=data["validation"]
    )

    if dataset_name == "two_moons":
        # Posterior samples on, as typical in SBI (e.g., https://arxiv.org/abs/1905.07488)
        obs = np.zeros((1, 2), dtype=np.float32)
        
    elif dataset_name == "inverse_kinematics":
        # Posterior samples for position (0, 1.5) from https://arxiv.org/abs/2101.10763
        obs = np.array([[0, 1.5]], dtype=np.float32)

    samples = workflow.sample(conditions={"observables": obs}, num_samples=NUM_SAMPLES_INFERENCE)
    np.save(f"samples/{model_name}_{dataset_name}.npy", samples["parameters"].squeeze())

if __name__ == "__main__":
    
    for dataset in DATASETS:

        data_dict = {
            "train": pickle.load(open(f"data/{dataset}_train_data.pkl", "rb")),
            "validation": pickle.load(open(f"data/{dataset}_validation_data.pkl", "rb")),
        }

        for model_name, conf_tuple in MODELS.items():
            bf.utils.logging.info(f"Training {model_name} on {dataset}")

            train_model(model_name, dataset, conf_tuple, data_dict)
            clear_session()
