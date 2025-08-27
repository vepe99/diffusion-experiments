import pickle

from keras.utils import clear_session

import bayesflow as bf


MODELS = {
        "flow_matching": (bf.networks.FlowMatching, {}),
        "ot_flow_matching": (bf.networks.FlowMatching, {"use_optimal_transport": True}),
        "consistency_model": (bf.networks.ConsistencyModel, {}),
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
        "diffusion_cosine_epsilon": (bf.networks.DiffusionModel, {
            "noise_schedule": "cosine", 
            "prediction_type": "epsilon"}),
    }

EPOCHS = 1
BATCH_SIZE = 128


def train_model(model_name, conf_tuple, data):

    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .rename("theta", "inference_variables")
        .rename("x", "inference_conditions")
    )

    inference_net = conf_tuple[0](**conf_tuple[1])

    workflow = bf.workflows.Workflow(
        adapter=adapter,
        inference_network=inference_net,
        checkpoint_filepath=f"checkpoints/{model_name}",
        standardize="all"
    )

    _ = workflow.fit_offline(
        data=data["train"],
        validation_data=data["validation"]
    )


if __name__ == '__main__':

    data_dict = {
        "train": pickle.load(open("data/two_moons_train_data.pkl", "rb")),
        "validation": pickle.load(open("data/two_moons_validation_data.pkl", "rb")),
        "test": pickle.load(open("data/two_moons_test_data.pkl", "rb"))
    }

    for model_name, conf_tuple in MODELS.items():
        train_model(model_name, conf_tuple, data_dict)
        clear_session()
