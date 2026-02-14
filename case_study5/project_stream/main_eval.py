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

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"
import keras
import bayesflow as bf


import logging
logging.getLogger('bayesflow').setLevel(logging.DEBUG)

from eval_config import EvalConfig

cs = ConfigStore.instance()
cs.store(name="train_config", node=EvalConfig)

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
    workflow_global = bf.BasicWorkflow(
        adapter=adapter,
        summary_network=bf.networks.SetTransformer(summary_dim=64, 
                                                dropout=0.1),
        inference_network=bf.networks.CompositionalDiffusionModel(),
        standardize=["inference_variables", "summary_variables"]
    )
    workflow_global.approximator = keras.models.load_model(model_path)

    test_data_path = os.path.join(cfg.base_dir, cfg.data_dir, f'simulation_multistream_{cfg.multistream_n_simulation}.npz')
    print('Loading test data from ', test_data_path)
    test_data = dict(np.load(test_data_path, allow_pickle=True))
    test_data = {k: test_data[k] for k in cfg.parameters_global + [cfg.sim_data, "j"] }
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
                        num_samples=cfg.n_samples,
                        conditions={cfg.sim_data: test_data[cfg.sim_data], 
                                    "j": test_data["j"]},
                        compute_prior_score=prior_global_score,
                        compositional_bridge_d1=1/cfg.inverse_compositional_bridge_d1,
                        mini_batch_size=cfg.mini_batch_size,
                        batch_size = 10,
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
    #true vs predicted recovery plots
    fig = bf.diagnostics.recovery(
        estimates=ps,
        targets=test_data,
        variable_names=cfg.paramater_global_pretty
        # variable_names = param_names_global
    )
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'global_recovery.pdf'))
    print('Saved global recovery plot')
    plt.show()
    #corner plot
    dataset_id = 0
    fig = bf.diagnostics.plots.pairs_posterior(
        estimates=ps,
        targets=test_data,
        dataset_id=dataset_id,
        variable_names=cfg.paramater_global_pretty,
        # variable_names = param_names_global,
    )
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, f'global_cornerplot_datasetid_{dataset_id}.pdf'))
    print(f'Saved global corner plot for dataset id {dataset_id}')
    plt.show()
    #calibration plot
    fig = bf.diagnostics.calibration_ecdf(
        estimates=ps,
        targets=test_data,
        difference=True,
        variable_names=cfg.paramater_global_pretty
        # variable_names = param_names_global
    )
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'global_calibration.pdf'))
    print('Saved global calibration plot')
    plt.show()
    #histograms
    global_posterior_stream_1 = {k: ps[k] for k in list(ps.keys())[:4]}
    test_data_stream_1 = {k: test_data[k] for k in list(global_posterior_stream_1.keys())}
    fig_1 = bf.diagnostics.plots.calibration_histogram(
        estimates=global_posterior_stream_1, 
        targets=test_data_stream_1,
        variable_names=cfg.paramater_global_pretty[:4]
        # variable_names = param_names_global
    )
    fig_1.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'global_histograms_1.pdf'))
    plt.show()
    global_posterior_stream_2 = {k: ps[k] for k in list(ps.keys())[4:]}
    test_data_stream_2 = {k: test_data[k] for k in list(global_posterior_stream_2.keys())}
    fig_2 = bf.diagnostics.plots.calibration_histogram(
        estimates=global_posterior_stream_2, 
        targets=test_data_stream_2,
        variable_names=cfg.paramater_global_pretty[4:]
        # variable_names = param_names_global
    )
    fig_2.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'global_histograms_2.pdf'))
    plt.show()
    print('Saved global histograms plot')

    # z_score contraction
    fig = bf.diagnostics.plots.z_score_contraction(
        estimates=ps, 
        targets=test_data,
        variable_names=cfg.paramater_global_pretty
        # variable_names = param_names_global
    )
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'global_z_score_contraction.pdf'))
    print('Saved global z-score contraction plot')
    plt.show()
    

    ###############
    # local model # 
    ###############

if __name__ == "__main__":
    main()