from autocvd import autocvd
autocvd(num_gpus = 1)
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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
from scipy import  special 


import logging
logging.getLogger('bayesflow').setLevel(logging.DEBUG)

from eval_config import EvalConfig
from utils_train_jax import AugmentationsClass #we will need to use the augmentations on the test_set


cs = ConfigStore.instance()
cs.store(name="eval_config", node=EvalConfig)


# def find_1600(model_path):
#     import zipfile
#     import json

#     with zipfile.ZipFile(model_path, 'r') as z:
#         config_str = z.read('config.json').decode('utf-8')

#     # Find all occurrences of 1600 with context
#     lines = config_str.replace(',', ',\n').replace('{', '{\n').replace('}', '\n}')
#     for i, line in enumerate(lines.split('\n')):
#         if '1600' in line:
            # print(f"Line {i}: {line.strip()}")

# def print_kerasmodel(model_path):
#     import zipfile
#     import json

#     def print_all_keys(obj, path="root", file=None):
#         if isinstance(obj, dict):
#             for key, value in obj.items():
#                 current_path = f"{path}.{key}"
#                 line = f"{current_path} : {repr(value) if not isinstance(value, (dict, list)) else ''}\n"
#                 file.write(line)
#                 print_all_keys(value, current_path, file)
#         elif isinstance(obj, list):
#             for i, item in enumerate(obj):
#                 print_all_keys(item, f"{path}[{i}]", file)

#     with zipfile.ZipFile(model_path, 'r') as z:
#         config_str = z.read('config.json').decode('utf-8')
#         config = json.loads(config_str)

#     output_path = model_path.replace('.keras', '_config_dump.txt')
#     with open(output_path, 'w') as f:
#         print_all_keys(config, file=f)

#     print(f"Config dumped to {output_path}")


def fix_keras_model(model_path, ):
    import zipfile
    import json
    fixed_model_path = model_path.replace('.keras', '_fixed.keras')
    if not os.path.exists(fixed_model_path):
        with zipfile.ZipFile(model_path, 'r') as zin:
            with zipfile.ZipFile(fixed_model_path, 'w') as zout:
                for item in zin.infolist():
                    data = zin.read(item.filename)
                    if item.filename == 'config.json':
                        config_str = data.decode('utf-8')
                        config_str = config_str.replace(
                            '__bayesflow_type__ArrayImpl',
                            '__bayesflow_type__ndarray'
                        )
                        data = config_str.encode('utf-8')
                    zout.writestr(item, data)
        print(f"Created fixed model at {fixed_model_path}")
    return fixed_model_path



@hydra.main(version_base=None, config_path="config", config_name="eval_config",)
def main(cfg: EvalConfig):

    print(cfg)
    print("##############")
    model_path = os.path.join(cfg.base_dir, cfg.model_dir, 'global_model.keras' )
    # Fix ArrayImpl serialization issue in the .keras fil
    model_path = fix_keras_model(model_path, )
    print('Loading model from ', model_path)
    print("##############")
    param_names_global = list(cfg.parameters_global)
    sim_data = str(cfg.sim_data)
    inference_conditions = str(cfg.inference_conditions[0])
    test_data_path = os.path.join(cfg.base_dir, cfg.data_dir, f'simulation_multistream_{cfg.multistream_n_simulation}.npz')
    print('Loading test data from ', test_data_path)
    test_data = dict(np.load(test_data_path, allow_pickle=True))
    keys_to_drop = set(test_data.keys()) - set(param_names_global) - {sim_data} - set(inference_conditions)
    keys_to_drop = list(keys_to_drop) 


    adapter = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .drop(keys_to_drop)
        .concatenate(param_names_global, into="inference_variables")
        .rename(sim_data, "summary_variables")
        .rename(inference_conditions, "inference_conditions")
    )
    with open(os.path.join(cfg.base_dir, cfg.model_dir, '.hydra', 'config.yaml'), "r") as f:
        model_config = yaml.safe_load(f)
    print(model_config)
    if cfg.noise_schedule is not None:
        inference_network = bf.networks.CompositionalDiffusionModel(
                                                        subnet_kwargs={
                                                        "widths": [model_config['global_model']['inference_mlp_width']] * model_config['global_model']['inference_mlp_depth'],
                                                        "time_embedding_dim": model_config['global_model']['inference_time_embedding_dim'],
                                                        },
                                                        schedule_kwargs = {**cfg.noise_schedule,},
                                                        )
    else:
        #probably needs to fix it to the training noise schedule 
        inference_network = bf.networks.CompositionalDiffusionModel(subnet_kwargs={
                                                        "widths": [model_config['global_model']['inference_mlp_width']] * model_config['global_model']['inference_mlp_depth'],
                                                        "time_embedding_dim": model_config['global_model']['inference_time_embedding_dim'],
                                                        },)
    workflow_global = bf.BasicWorkflow(
        adapter=adapter,
        summary_network=bf.networks.SetTransformer(summary_dim=model_config['global_model']['summary_dim'], 
                                                   num_heads=(model_config['global_model']['num_heads'],model_config['global_model']['num_heads'],),
                                                   embed_dims = (model_config['global_model']['embed_dims'], model_config['global_model']['embed_dims'],),
                                                   mlp_depths=(model_config['global_model']['mlp_depths'], model_config['global_model']['mlp_depths']),
                                                   mlp_widths=(model_config['global_model']['mlp_widths'], model_config['global_model']['mlp_widths']),
                                                   dropout=0.1),
        inference_network=inference_network,
        standardize=["inference_variables", "summary_variables"]
    )
    workflow_global.approximator = keras.models.load_model(model_path)
    test_data = {k: test_data[k] for k in cfg.parameters_global + [cfg.sim_data, "j"] }
    # Augmentation
    augmentations_class = AugmentationsClass(cfg)
    augmentations = []
    if "remove_los_velocity" in cfg.augmentations:
        augmentations.append(augmentations_class.remove_los_velocity)
    if "convert_distance_to_parallax" in cfg.augmentations:
        augmentations.append(augmentations_class.convert_distance_to_parallax)
    if "sample_magnitudes" in cfg.augmentations:
        augmentations.append(augmentations_class.sample_magnitudes)
    if "sample_obs_error" in cfg.augmentations:
        augmentations.append(augmentations_class.sample_obs_error)
    if "apply_obs_error" in cfg.augmentations:
        augmentations.append(augmentations_class.apply_obs_error)
    if "observational_window" in cfg.augmentations:
        augmentations.append(augmentations_class.observational_window)
    if "observed_n_stars" in cfg.augmentations:
        augmentations.append(augmentations_class.subsampling_to_observed_n_stars)
    if "flip_dirz" in cfg.augmentations:
        augmentations.append(augmentations_class.flip_dirz)
    if "concatentate_sigma_error_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatentate_sigma_error_to_sim_data)
    if "concatenate_magnitudes_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_magnitudes_to_sim_data)
    if "concatenate_j_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_j_to_sim_data)

    test_data[cfg.sim_data] = test_data[cfg.sim_data].reshape(-1, test_data[cfg.sim_data].shape[-2], test_data[cfg.sim_data].shape[-1])
    test_data['j'] = test_data['j'].reshape(-1, 1)
    print('Test data sim shape before augmentation: ', test_data[cfg.sim_data].shape)
    for aug in augmentations:
        test_data = aug(test_data)
    test_data[cfg.sim_data] = test_data[cfg.sim_data].reshape(-1, len(cfg.target_streams.keys()), test_data[cfg.sim_data].shape[-2], test_data[cfg.sim_data].shape[-1])
    test_data['j'] = test_data['j'].reshape(-1,len(cfg.target_streams.keys()), 1)
    print('Test data sim shape after augmentation: ', test_data[cfg.sim_data].shape)
    print('Test data keys: ', test_data.keys())
    print('Test data attention mask shape: ', test_data['attention_mask'].shape)
    with open(os.path.join(cfg.base_dir, cfg.data_dir, '.hydra', 'config.yaml'), "r") as f:
        test_sim_config = yaml.safe_load(f)
    print('Test simulation config prior: ', test_sim_config['priors_global'])

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
    workflow_global.approximator.inference_network.integrate_kwargs.update({
        'method': cfg.method,
        'steps': cfg.steps,
        'compositional_bridge_d1': 1/cfg.inverse_compositional_bridge_d1,
        'mini_batch_size': cfg.mini_batch_size,
        "max_steps": cfg.max_steps,
        })
    global_posterior = workflow_global.compositional_sample(
                        num_samples=cfg.n_samples,
                        conditions={cfg.sim_data: test_data[cfg.sim_data], 
                                    "j": test_data["j"]},
                        compute_prior_score=prior_global_score,
                        batch_size = cfg.batch_size,
                        kwargs={'attention_mask': test_data['attention_mask']},
                        )
    os.makedirs(name= os.path.join(cfg.base_dir, cfg.results_dir), exist_ok=True)
    ps = global_posterior.copy()
    if cfg.use_streamax_simulator:
        #let's extract q from the dirx, diry, dirz of the halo, to be able to plot it and compare with the true value
        q_min = 0.5
        q_max = 1.5
        r_posterior = np.sqrt(ps['dirx_Triaxial_rotated_halo']**2 + ps['diry_Triaxial_rotated_halo']**2 + ps['dirz_Triaxial_rotated_halo']**2)
        u_uniform_posterior = special.erf(r_posterior/np.sqrt(2)) - np.sqrt(2/np.pi)*r_posterior*np.exp(-(r_posterior**2)/2)
        ps['$q_{NFW}$'] = q_min + (q_max-q_min)*u_uniform_posterior
        r_test = np.sqrt(test_data['dirx_Triaxial_rotated_halo']**2 + test_data['diry_Triaxial_rotated_halo']**2 + test_data['dirz_Triaxial_rotated_halo']**2)
        u_uniform_test = special.erf(r_test/np.sqrt(2)) - np.sqrt(2/np.pi)*r_test*np.exp(-(r_test**2)/2)
        test_data['$q_{NFW}$'] = q_min + (q_max-q_min)*u_uniform_test
        param_names_global = cfg.parameters_global + ['$q_{NFW}$']
        cfg.paramater_global_pretty = cfg.paramater_global_pretty + ['$q_{NFW}$']
        #apply flipping to have all halos with dirz > 0, to avoid the degeneracy in the definition of the angles of the halo and make the plots easier to interpret
        #only needed for the 100 epochs models with onlyhalo
        # mask_posterior = ps['dirz_Triaxial_rotated_halo'] < 0
        # ps['dirz_Triaxial_rotated_halo'][mask_posterior] *= -1
        # ps['dirx_Triaxial_rotated_halo'][mask_posterior] *= -1
        # ps['diry_Triaxial_rotated_halo'][mask_posterior] *= -1
        
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