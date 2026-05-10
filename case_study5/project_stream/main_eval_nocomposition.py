from autocvd import autocvd
autocvd(num_gpus = 1)


import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.core.config_store import ConfigStore
import numpy as np

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"
import keras
import bayesflow as bf
from scipy import  special 


import logging
logging.getLogger('bayesflow').setLevel(logging.DEBUG)

from config.EvalConfig import EvalConfig
from utils.utils_train_jax import AugmentationsClass #we will need to use the augmentations on the test_set

cs = ConfigStore.instance()
cs.store(name="eval_config", node=EvalConfig)

@hydra.main(version_base=None, config_path="config", config_name="eval_config",)
def main(cfg: EvalConfig):

    print(cfg)
    print("##############")
    model_path = os.path.join(cfg.base_dir, cfg.model_dir, 'global_model.keras' )
    # Fix ArrayImpl serialization issue in the .keras file
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
    model_path = fixed_model_path
    print('Loading model from ', model_path)
    print("##############")
    param_names_global = list(cfg.parameters_global)
    sim_data = str(cfg.sim_data)
    inference_conditions = str(cfg.inference_conditions[0])
    test_data_path = os.path.join(cfg.base_dir, cfg.data_dir, f'simulation_multistream_{cfg.multistream_n_simulation}.npz')

    print('Loading test data from ', test_data_path)
    test_data = dict(np.load(test_data_path, allow_pickle=True))

    # Boolean mask: True where a simulation is NOT NaN (shape: n_simulations)
    valid_mask = ~np.isnan(test_data['sim_data_carthesian']).any(axis=(-1, -2, -3))

    # Filter every key in the dict along the simulation axis
    test_data = {k: v[valid_mask] for k, v in test_data.items()}
    keys_to_drop = set(test_data.keys()) - set(param_names_global) - {sim_data} - set(inference_conditions)
    keys_to_drop = list(keys_to_drop) 
    # we need to subsample 
    # num_index_testset = len(test_data[cfg.sim_data])
    # shuffled_index = np.random.permutation(np.arange(num_index_testset))
    # subssample_shuffled_index = shuffled_index[::len(cfg.target_streams.keys())]
    # for k in test_data.keys():
    #     test_data[k] = test_data[k][subssample_shuffled_index]
    # print('We randomly subsample the test set to:', len(subssample_shuffled_index))
    # print('Sim data after subsampling: ', test_data[cfg.sim_data].shape)

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
    # model_config = {'global_model':
    #                 {
    #                     'inference_mlp_width': cfg.global_model.inference_mlp_width,
    #                     'inference_mlp_depth': cfg.global_model.inference_mlp_depth,
    #                     'inference_time_embedding_dim': cfg.global_model.inference_time_embedding_dim,
    #                     'summary_dim': cfg.global_model.summary_dim,
    #                     'num_heads': cfg.global_model.num_heads,
    #                     'embed_dims': cfg.global_model.embed_dims,
    #                     'mlp_depths': cfg.global_model.mlp_depths,
    #                     'mlp_widths': cfg.global_model.mlp_widths,
    #                     'dropout': cfg.global_model.dropout,
    #                 }
    #             }
    print(model_config)
    workflow_global = bf.BasicWorkflow(
        adapter=adapter,
        summary_network=bf.networks.SetTransformer(summary_dim=model_config['global_model']['summary_dim'], 
                                                   num_heads=(model_config['global_model']['num_heads'],model_config['global_model']['num_heads'],),
                                                   embed_dims = (model_config['global_model']['embed_dims'], model_config['global_model']['embed_dims'],),
                                                   mlp_depths=(model_config['global_model']['mlp_depths'], model_config['global_model']['mlp_depths']),
                                                   mlp_widths=(model_config['global_model']['mlp_widths'], model_config['global_model']['mlp_widths']),
                                                   dropout=0.1),
        inference_network=bf.networks.CompositionalDiffusionModel(subnet_kwargs={
                                                        "widths": [model_config['global_model']['inference_mlp_width']] * model_config['global_model']['inference_mlp_depth'],
                                                        "time_embedding_dim": model_config['global_model']['inference_time_embedding_dim'],
                                                        }),
        standardize=["inference_variables", "summary_variables"]
    )
    workflow_global.approximator = keras.models.load_model(model_path)
    # workflow_global.approximator.save_weights(model_path.replace('.keras', '.weights.h5'))
    test_data = {k: test_data[k] for k in cfg.parameters_global + [cfg.sim_data, "j"] }
    # Augmentation
    augmentations_class = AugmentationsClass(cfg)
    augmentations = []
    if "cut_to_300_particles" in cfg.augmentations:
        augmentations.append(augmentations_class.cut_to_300_particles)
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
    if "observational_window_random" in cfg.augmentations:
        augmentations.append(augmentations_class.observational_window_random)
    if "observed_n_stars" in cfg.augmentations:
        augmentations.append(augmentations_class.subsampling_to_observed_n_stars)
    if "mask_vlos" in cfg.augmentations:
        augmentations.append(augmentations_class.mask_vlos)
    if "flip_dirz" in cfg.augmentations:
        augmentations.append(augmentations_class.flip_dirz)
    if "concatentate_sigma_error_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatentate_sigma_error_to_sim_data)
    if "concatenate_magnitudes_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_magnitudes_to_sim_data)
    if "concatenate_vlos_mask_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_vlos_mask_to_sim_data)
    if "concatenate_j_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_j_to_sim_data)

    test_data[cfg.sim_data] = test_data[cfg.sim_data].reshape(-1, test_data[cfg.sim_data].shape[-2], test_data[cfg.sim_data].shape[-1])
    test_data['j'] = test_data['j'].reshape(-1, 1)
    print('Test data sim shape before augmentation: ', test_data[cfg.sim_data].shape)
    for aug in augmentations:
        print(f"Applying augmentation: {aug.__name__}")
        test_data = aug(test_data)
    for k in cfg.parameters_global:
        test_data[k] = np.repeat(test_data[k], 3, axis=0).reshape(-1, 1)
    print('Test data sim shape after augmentation: ', test_data[cfg.sim_data].shape)
    print('Test data keys shape: ', [test_data[k].shape for k in test_data.keys()])
    print('Test data attention mask shape: ', test_data['attention_mask'].shape)


    logging.info("Starting Partial-Pooling (global) inference with no composition...")
    print(workflow_global.approximator.inference_network.integrate_kwargs.keys())
    # workflow_global.approximator.inference_network.integrate_kwargs.update({
    #     'method': cfg.method,
    #     'steps': cfg.steps,
    #     # 'compositional_bridge_d1': 1/cfg.inverse_compositional_bridge_d1,
    #     # 'mini_batch_size': cfg.mini_batch_size,
    #     "max_steps": cfg.max_steps,
    #     })
    global_posterior = workflow_global.sample(
                        num_samples=cfg.n_samples,
                        conditions={cfg.sim_data: test_data[cfg.sim_data], 
                                    "j": test_data["j"]},
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
        
    np.savez(os.path.join(cfg.base_dir, cfg.results_dir, 'posterior.npz'), **ps)

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
    for ax in fig.get_axes():
        ax.grid(False)
        for txt in ax.texts:
            txt.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.3'))
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'recovery.pdf'))
    print('Saved recovery plot')
    plt.show()


    #splititting recovery plot into the three components, each with its 3 parameters
    global_posterior_nfw = {k: ps[k] for k in list(ps.keys())[:3]}
    test_data_stream_nfw = {k: test_data[k] for k in list(global_posterior_nfw.keys())}
    fig_nfw = bf.diagnostics.recovery(
        estimates=global_posterior_nfw,
        targets=test_data_stream_nfw,
        variable_names=cfg.paramater_global_pretty[:3]
        # variable_names = param_names_global[:3]
    )
    for ax in fig_nfw.get_axes():
        ax.grid(False)
        for txt in ax.texts:
            txt.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.3'))
    fig_nfw.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'recovery_nfw.pdf'))
    print('Saved global recovery plot for NFW parameters')
    plt.show()

    #thin disk
    global_posterior_thin_disk = {k: ps[k] for k in list(ps.keys())[3:6]}
    test_data_stream_thin_disk = {k: test_data[k] for k in list(global_posterior_thin_disk.keys())}
    fig_thin_disk = bf.diagnostics.recovery(
        estimates=global_posterior_thin_disk,
        targets=test_data_stream_thin_disk,
        variable_names=cfg.paramater_global_pretty[3:6]
        # variable_names = param_names_global[3:6]
    )
    for ax in fig_thin_disk.get_axes():
        ax.grid(False)
        for txt in ax.texts:
            txt.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.3'))
    fig_thin_disk.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'recovery_thin_disk.pdf'))
    print('Saved global recovery plot for thin disk parameters')
    plt.show()

    #thick disk
    global_posterior_thick_disk = {k: ps[k] for k in list(ps.keys())[6:9]}
    test_data_stream_thick_disk = {k: test_data[k] for k in list(global_posterior_thick_disk.keys())}
    fig_thick_disk = bf.diagnostics.recovery(
        estimates=global_posterior_thick_disk,
        targets=test_data_stream_thick_disk,
        variable_names=cfg.paramater_global_pretty[6:9]
        # variable_names = param_names_global[6:9]
    )
    for ax in fig_thick_disk.get_axes():    
        ax.grid(False)
        for txt in ax.texts:
            txt.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.3'))
    fig_thick_disk.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'recovery_thick_disk.pdf'))
    print('Saved global recovery plot for thick disk parameters')
    plt.show()

    #corner plot
    dataset_id = np.array([0])
    fig = bf.diagnostics.plots.pairs_posterior(
        estimates=ps,
        targets=test_data,
        dataset_id=dataset_id,
        variable_names=cfg.paramater_global_pretty,
        # variable_names = param_names_global,
    )
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, f'cornerplot_datasetid_{dataset_id}.pdf'))
    print(f'Saved corner plot for dataset id {dataset_id}')
    plt.show()
    #calibration plot with diff
    fig = bf.diagnostics.calibration_ecdf(
        estimates=ps,
        targets=test_data,
        difference=True,
        variable_names=cfg.paramater_global_pretty,
    )
    for ax in fig.get_axes():
        ax.grid(False)
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'calibration.pdf'))
    print('Saved calibration plot')
    plt.show()
    #stacked
    from utils.utils_plot import calibration_ecdf
    fig = calibration_ecdf(
        estimates=ps,
        targets=test_data,
        difference=True,
        variable_names=cfg.paramater_global_pretty,
        stacked = True,
        rank_ecdf_color=plt.cm.magma(np.linspace(0, 1, len(cfg.paramater_global_pretty))),

    )
    for ax in fig.get_axes():
        ax.grid(False)
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'calibration_stacked.pdf'))
    plt.show()
    fig = calibration_ecdf(
        estimates=ps,
        targets=test_data,
        difference=False,
        variable_names=cfg.paramater_global_pretty,
        stacked = True,
        rank_ecdf_color=plt.cm.magma(np.linspace(0, 1, len(cfg.paramater_global_pretty))),
    )
    for ax in fig.get_axes():
        ax.grid(False)
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'calibration_stacked_no_diff.pdf'))
    print('Saved stacked calibration plot')
    #calibration plot without diff
    fig = bf.diagnostics.calibration_ecdf(
        estimates=ps,
        targets=test_data,
        difference=False,
        variable_names=cfg.paramater_global_pretty
    )
    for ax in fig.get_axes():
        ax.grid(False)
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'calibration_no_diff.pdf'))
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
    for ax in fig_1.get_axes():
        ax.grid(False)
    fig_1.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'histograms_1.pdf'))
    plt.show()
    global_posterior_stream_2 = {k: ps[k] for k in list(ps.keys())[4:]}
    test_data_stream_2 = {k: test_data[k] for k in list(global_posterior_stream_2.keys())}
    fig_2 = bf.diagnostics.plots.calibration_histogram(
        estimates=global_posterior_stream_2, 
        targets=test_data_stream_2,
        variable_names=cfg.paramater_global_pretty[4:]
        # variable_names = param_names_global
    )
    for ax in fig_2.get_axes():
        ax.grid(False)
    fig_2.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'histograms_2.pdf'))
    plt.show()
    print('Saved histograms plot')

    # z_score contraction
    fig = bf.diagnostics.plots.z_score_contraction(
        estimates=ps, 
        targets=test_data,
        variable_names=cfg.paramater_global_pretty
        # variable_names = param_names_global
    )
    for ax in fig.get_axes():
        ax.grid(False)
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'z_score_contraction.pdf'))
    print('Saved z-score contraction plot')
    plt.show()
    print('Finished evaluation with no composition')
    print('Results for calibration saved in ', os.path.join(cfg.base_dir, cfg.results_dir, 'calibration.pdf'))
    

    ###############
    # local model # 
    ###############


if __name__ == "__main__":
    main()
