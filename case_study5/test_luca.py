from autocvd import autocvd
autocvd(num_gpus=1)

import os
os.environ["KERAS_BACKEND"] = "jax"


import bayesflow as bf
from bayesflow.networks import TimeSeriesTransformer, FusionTransformer
# from dataset_corrected import SlopesBayesDataset
# from model_CNN import CNN


from omegaconf import DictConfig
from tqdm import tqdm, trange
import numpy as np 


def train():

    print("CREATING DATASETS")
    training_data = {}
    obs = np.ones((1000, 2556, 1701))
    par = np.ones((1000, 4))
    training_data['observations'] = obs
    training_data['parameters'] = par

    validation_data = {}
    obs = np.ones((200, 700, 500))
    par = np.ones((200, 4))
    validation_data['observations'] = obs
    validation_data['parameters'] = par

    print("CREATED DATASETS")

   
    print(f"Calling the adapter...")
    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        # .as_time_series("observations")
        .rename("parameters", "inference_variables")
        .rename("observations", "summary_variables")
    )

    print(f"Calling the summary_net")

    # summary net
    summary_net = FusionTransformer() #   TimeSeriesTransformer() #

    print(f"Calling the inference net...")
    # inference net
    inference_net = bf.networks.FlowMatching()

    print(f"Defining the workflow...")

    # workflow
    workflow = bf.BasicWorkflow(
        adapter=adapter,
        inference_network=inference_net,
        summary_network=summary_net,
        standardize=None,  # no need to standardize due to log-transform
        verbos = 2, 
    )

    print(f"Start the training...")
    # training
    history = workflow.fit_offline(
        data=training_data, 
        epochs=10, 
        batch_size=32, 
        validation_data=validation_data
    )


if __name__ == "__main__":
    train()