from autocvd import autocvd
autocvd(num_gpus = 1)
import os
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.core.config_store import ConfigStore
import numpy as np

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

import bayesflow as bf


import logging
logging.getLogger('bayesflow').setLevel(logging.DEBUG)

# from case_study5.project_stream.train_config import TrainConfig
from train_config import TrainConfig

cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)


@hydra.main(version_base=None, config_path="config", config_name="train_config",)
def main(cfg: TrainConfig):
    print(cfg)
    model_path = os.path.join(cfg.base_dir, cfg.results_dir, )
    os.makedirs(model_path, exist_ok=True)
    param_names_global = list(cfg.parameters_global)
    sim_data = str(cfg.sim_data)
    inference_conditions = str(cfg.inference_conditions[0]) #jut 1
    # param_names_global = ['m_Triaxial_halo', 'r_Triaxial_halo', 'q2_Triaxial_halo', 'rho_thin_disk', 'hr_thin_disk', 'hz_thin_disk', 'rho_thick_disk', 'hr_thick_disk', 'hz_thick_disk']
    
    adapter = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .concatenate(param_names_global, into="inference_variables")
        .rename(sim_data, "summary_variables")
        # .convert_dtype("float32", "int", include="j", exclude=["inference_variables", "summary_variables"])
        .rename(inference_conditions, "inference_conditions")
        # .one_hot("inference_conditions", num_classes=3)
    )
    workflow_global = bf.BasicWorkflow(
        adapter=adapter,
        summary_network=bf.networks.SetTransformer(summary_dim=64, 
                                                #    num_heads=(4, 4),
                                                dropout=0.1),
        inference_network=bf.networks.CompositionalDiffusionModel(),
        standardize=["inference_variables", "summary_variables"]
    )
    # train_data_path = os.path.join(cfg.base_dir, cfg.data_dir)
    train_data_path = "/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/streams/data/"
    training_data = dict(np.load(os.path.join(train_data_path, "training_data_150000.npz"), allow_pickle=True))
    print("Training data keys", training_data.keys())

    history = workflow_global.fit_offline(
        training_data,
        epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        verbose=cfg.verbose,
    )
    workflow_global.approximator.save(os.path.join(model_path, 'global_model.keras'))


if __name__ == "__main__":
    main()