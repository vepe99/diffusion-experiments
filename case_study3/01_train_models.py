import pickle
import numpy as np

import bayesflow as bf

from keras.utils import clear_session


EPOCHS = 300
BATCH_SIZE = 64
NUM_SAMPLES_INFERENCE = 3000
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
    )

    _ = workflow.fit_offline(
        data=data["train"],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=data["validation"]
    )


if __name__ == "__main__":
    
    for dataset in DATASETS:

        data_dict = {
            "train": pickle.load(open(f"data/{dataset}_train_data.pkl", "rb")),
            "validation": pickle.load(open(f"data/{dataset}_validation_data.pkl", "rb")),
        }

        for model_name, conf_tuple in MODELS.items():
            bf.utils.logging.info(f"Training {model_name} on {dataset}...")

            train_model(model_name, dataset, conf_tuple, data_dict)
            clear_session()
