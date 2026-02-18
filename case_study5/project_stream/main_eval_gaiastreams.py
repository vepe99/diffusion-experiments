from autocvd import autocvd
autocvd(num_gpus = 1)
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
from chainconsumer import Chain, ChainConsumer, make_sample
import pandas as pd

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"
import keras
import bayesflow as bf


import logging
logging.getLogger('bayesflow').setLevel(logging.DEBUG)

from eval_config import EvalConfig
from utils_train import AugmentationsClass #we will need to use the augmentations on the test_set


cs = ConfigStore.instance()
cs.store(name="eval_config", node=EvalConfig)

@hydra.main(version_base=None, config_path="config", config_name="eval_config",)
def main(cfg: EvalConfig):

    print(cfg)
    print("##############")
    model_path = os.path.join(cfg.base_dir, cfg.model_dir, 'global_model.keras' )
    print('Loading model from ', model_path)
    print("##############")
    param_names_global = list(cfg.parameters_global)
    sim_data = str(cfg.sim_data)
    inference_conditions = str(cfg.inference_conditions[0])

    adapter = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .concatenate(param_names_global, into="inference_variables")
        .rename(sim_data, "summary_variables")
        .rename(inference_conditions, "inference_conditions")
    )
    with open(os.path.join(cfg.base_dir, cfg.model_dir, '.hydra', 'config.yaml'), "r") as f:
        model_config = yaml.safe_load(f)
    print(model_config)
    workflow_global = bf.BasicWorkflow(
        adapter=adapter,
        summary_network=bf.networks.SetTransformer(summary_dim=model_config['global_model']['summary_dim'], 
                                                   num_heads=(model_config['global_model']['num_heads'],model_config['global_model']['num_heads'],),
                                                   embed_dims = (model_config['global_model']['summary_dim'], model_config['global_model']['summary_dim'],),
                                                   dropout=0.1),
        inference_network=bf.networks.CompositionalDiffusionModel(),
        standardize=["inference_variables", "summary_variables"]
    )
    workflow_global.approximator = keras.models.load_model(model_path)

    test_data_path = '/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/gaia_observed_streams.npz'
    print('Loading test data from ', test_data_path)
    test_data = dict(np.load(test_data_path, allow_pickle=True))
    test_data = {k: test_data[k] for k in [cfg.sim_data, "j"] }
    # Augmentation
    augmentations_class = AugmentationsClass(cfg)
    augmentations = []
    if "remove_los_velocity" in cfg.augmentations:
        augmentations.append(augmentations_class.remove_los_velocity)
    test_data[cfg.sim_data] = test_data[cfg.sim_data].reshape(-1, test_data[cfg.sim_data].shape[-2], test_data[cfg.sim_data].shape[-1])
    test_data['j'] = test_data['j'].reshape(-1, 1)
    print('Test data sim shape before augmentation: ', test_data[cfg.sim_data].shape)
    for aug in augmentations:
        test_data = aug(test_data)
    test_data[cfg.sim_data] = test_data[cfg.sim_data].reshape(-1, len(cfg.target_streams.keys()), test_data[cfg.sim_data].shape[-2], test_data[cfg.sim_data].shape[-1])
    test_data['j'] = test_data['j'].reshape(-1,len(cfg.target_streams.keys()), 1)
    print('Test data sim shape after augmentation: ', test_data[cfg.sim_data].shape)
    print('Test data keys: ', test_data.keys())
    with open(os.path.join(cfg.base_dir, cfg.data_dir, '.hydra', 'config.yaml'), "r") as f:
        test_sim_config = yaml.safe_load(f)

    def prior_global_score(x, cfg=cfg, test_sim_config=test_sim_config):
        
        score = {}
        
        for k in cfg.parameters_global:
            # print(f"Computing prior score for {k} with type {test_sim_config['priors_global'][k]['type']}")
            if test_sim_config['priors_global'][k]['type'] == 'uniform':
                score[k] = np.zeros_like(x[k])
            elif test_sim_config['priors_global'][k]['type'] == 'normal':
                mean = test_sim_config['priors_global'][k]['prior_parameters'][0]
                std = test_sim_config['priors_global'][k]['prior_parameters'][1]
                score[k] = -(x[k] - mean) / std**2 
        return score

    logging.info("Starting Partial-Pooling (global) inference...")
    global_posterior = workflow_global.compositional_sample(
                        num_samples=cfg.num_samples,
                        conditions={cfg.sim_data: test_data[cfg.sim_data], 
                                    "j": test_data["j"]},
                        compute_prior_score=prior_global_score,
                        # compositional_bridge_d1=1/cfg.inverse_compositional_bridge_d1,
                        # mini_batch_size=cfg.mini_batch_size,
                        mini_batch_size=cfg.mini_batch_size,
                        batch_size = cfg.batch_size,
                        method=cfg.method,
                        steps=cfg.steps,
                        max_steps=cfg.max_steps
                        )
    os.makedirs(name= os.path.join(cfg.base_dir, cfg.results_dir), exist_ok=True)
    ps = global_posterior.copy()
    np.savez(os.path.join(cfg.base_dir, cfg.results_dir, 'global_posterior.npz'), **ps)
    ###############
    # PLOTS GLOBAL#
    ###############
    #corner plot
    # dataset_id = 0
    # fig = bf.diagnostics.plots.pairs_posterior(
    #     estimates=ps,
    #     # targets=test_data,
    #     dataset_id=dataset_id,
    #     variable_names=cfg.paramater_global_pretty,
    # )
    print('shapes of posterior samples: ', {k: v.shape for k, v in ps.items()})
    for k in ps.keys():
        ps[k] = ps[k].reshape(-1,)
    df = pd.DataFrame(ps) 
    print('Df columns before renaming: ', df.columns)
    df.columns = list(cfg.paramater_global_pretty)
    print('Df columns after renaming: ', df.columns)
    c = ChainConsumer()
    c.add_chain(Chain(samples=df, name="An Example Contour"))
    fig = c.plotter.plot()
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, f'global_cornerplot.pdf'))
    print(f'Saved global corner plot')
    plt.show()
    

    ###############
    # local model # 
    ###############

if __name__ == "__main__":
    main()