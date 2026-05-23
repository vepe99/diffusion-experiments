from autocvd import autocvd
autocvd(num_gpus = 1)


import os
# os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
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
from utils.custom_summary_network import SetTransformer, FusionNetwork



import logging
logging.getLogger('bayesflow').setLevel(logging.DEBUG)

from config.EvalConfig import EvalConfig
from utils.utils_train_jax_new_rotationcurve_fixedvlosmask import (AugmentationsClass,
                                                     compute_standardization, 
                                                     apply_standardization,
                                                     save_stats)#we will need to use the augmentations on the test_set


from numpy.polynomial.legendre import leggauss

def M_enc_vec(r, rho0, a1, gamma, n_gl=100):
    r, rho0, a1, gamma = [np.asarray(v, dtype=np.float64) for v in (r, rho0, a1, gamma)]
    t_nodes, t_weights = leggauss(n_gl)

    log_lo = np.log(1e-8)
    # Clamp r/a1 to strictly positive to avoid log(0) or log(negative)
    ratio = np.clip(r / np.where(a1 > 0, a1, np.nan), 1e-30, None)
    log_hi = np.log(ratio)[..., np.newaxis]

    log_x = 0.5 * (log_hi - log_lo) * t_nodes + 0.5 * (log_hi + log_lo)
    x     = np.exp(log_x)
    jac   = 0.5 * (log_hi - log_lo) * x

    rho0_b  = np.where(rho0 > 0, rho0,  np.nan)[..., np.newaxis]
    a1_b    = np.where(a1  > 0, a1,    np.nan)[..., np.newaxis]
    gamma_b = np.clip(gamma, 1e-6, 2.99)[...,         np.newaxis]

    s         = x * a1_b
    integrand = 4*np.pi * s**2 * rho0_b * x**(-gamma_b) * (1 + x)**(gamma_b - 3)

    return np.sum(t_weights * integrand * jac, axis=-1)


def find_r200_vec(rho0, a1, gamma, rho_crit, r_min=1e-3, r_max=1e4,
                  n_bisect=50, n_gl=100):
    """
    Fully vectorized r200 via bisection.
    rho0, a1, gamma: arrays of any broadcastable shape (...,).
    Returns r200 of shape (...,).
    """
    rho0, a1, gamma = np.broadcast_arrays(
        *[np.asarray(v, dtype=np.float64) for v in (rho0, a1, gamma)]
    )
    lo = np.full(rho0.shape, r_min)
    hi = np.full(rho0.shape, r_max)

    def f(r):
        return M_enc_vec(r, rho0, a1, gamma, n_gl) - (800*np.pi/3) * rho_crit * r**3

    for _ in range(n_bisect):          # 50 steps → sub-nanoparsec precision
        mid   = 0.5 * (lo + hi)
        f_mid = f(mid)
        lo    = np.where(f_mid < 0, mid, lo)
        hi    = np.where(f_mid >= 0, mid, hi)

    return 0.5 * (lo + hi)


rho_crit = 1.40e2   # Msun/kpc^3 at z=0

cs = ConfigStore.instance()
cs.store(name="eval_config", node=EvalConfig)

@hydra.main(version_base=None, config_path="config", config_name="eval_config_new_rotationcurve_agama",)
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
    augmentations_class = AugmentationsClass(cfg)

    test_data = dict(np.load(test_data_path, allow_pickle=True))
    test_data_rotation_curve = dict(np.load(f'./data/plots/agama_rotcurv_multistream/{cfg.multistream_n_simulation}/rotation_curves.npz'))
    mask_r_kpc = (augmentations_class.obs_R >5.5)
    test_data['vcirc_kms'] = test_data_rotation_curve['vcirc_kms'][:, mask_r_kpc, None] #extra dimension (n_observation, len_r_kpc, 1)
    for k in test_data.keys():
        print(f"{k}: {test_data[k].shape}")
    # Boolean mask: True where a simulation is NOT NaN (shape: n_simulations)
    valid_mask = ~np.isnan(test_data['sim_data_carthesian']).any(axis=(-1, -2, -3))

    # Filter every key in the dict along the simulation axis
    test_data = {k: v[valid_mask] for k, v in test_data.items()}
    n_simulation = len(valid_mask)
    # print('Remove index of bad simulation')
    # bad_index = [5, 16, 23, 26, 33, 39, 90, 99] 
    # bad_index = [1, 10, 66, 70, 99]
    # test_data = {k: np.delete(v, bad_index, axis=0) for k, v in test_data.items()}

    keys_to_drop = set(test_data.keys()) - set(param_names_global) - {sim_data} - set(inference_conditions) - set(['vcirc_kms',])
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
        .rename(inference_conditions, "inference_conditions")
        .concatenate(param_names_global, into="inference_variables")
        .rename('attention_mask', 'summary_attention_mask')
        .rename(sim_data, "input_a")
        .rename('vcirc_kms', "input_b")
        .group(
            ["input_a", "input_b",], into="summary_variables")  
        )
    # with open(os.path.join(cfg.base_dir, cfg.model_dir, '.hydra', 'config.yaml'), "r") as f:
    #     model_config = yaml.safe_load(f)
    # print(model_config)
    model_config = {'global_model':
                    {
                        'inference_mlp_width': cfg.global_model.inference_mlp_width,
                        'inference_mlp_depth': cfg.global_model.inference_mlp_depth,
                        'inference_time_embedding_dim': cfg.global_model.inference_time_embedding_dim,
                        'summary_dim': cfg.global_model.summary_dim,
                        'num_heads': cfg.global_model.num_heads,
                        'embed_dims': cfg.global_model.embed_dims,
                        'mlp_depths': cfg.global_model.mlp_depths,
                        'mlp_widths': cfg.global_model.mlp_widths,
                        'dropout': cfg.global_model.dropout,
                    }
                }
    print(model_config)
    summary_network_a = SetTransformer(
            # summary_dim=cfg.global_model.summary_dim,
            # embed_dims=(cfg.global_model.embed_dims, cfg.global_model.embed_dims),
            # num_heads=(
            #     cfg.global_model.num_heads,
            #     cfg.global_model.num_heads,
            # ),
            # mlp_depths=(cfg.global_model.mlp_depths, cfg.global_model.mlp_depths),
            # mlp_widths=(cfg.global_model.mlp_widths, cfg.global_model.mlp_widths),
            dropout=cfg.global_model.dropout,
        )
    summary_network_b = bf.networks.TimeSeriesTransformer()
    head = keras.Sequential(
        [bf.networks.MLP(widths=[128, 128]), keras.layers.Dense(units=32)]
    )
    summary_network = FusionNetwork(
        backbones={"input_a": summary_network_a, "input_b": summary_network_b},
        head=head,
    )
    workflow_global = bf.CompositionalWorkflow(
        adapter=adapter,
        summary_network=summary_network,
        inference_network=bf.networks.DiffusionModel(
            # subnet_kwargs={
            #     "widths": [cfg.global_model.inference_mlp_width]
            #     * cfg.global_model.inference_mlp_depth,
            #     "time_embedding_dim": cfg.global_model.inference_time_embedding_dim,
            # }
        ),
        standardize=["inference_variables","summary_variables"],
        checkpoint_filepath=model_path,
        checkpoint_name="checkpoint_global_model.keras",
    )
    workflow_global.approximator = keras.models.load_model(model_path)
    workflow_global.approximator.save_weights(model_path.replace('.keras', '.weights.h5'))
    test_data = {k: test_data[k] for k in cfg.parameters_global + [cfg.sim_data, "j"] + ['vcirc_kms']}
    # Augmentation

    # --- Coordinate transforms (must be first, before any masking) ---
    augmentations_class = AugmentationsClass(cfg)
    augmentations = []

    # --- Coordinate transforms (must be first, before any masking) ---
    if "remove_los_velocity" in cfg.augmentations:
        augmentations.append(augmentations_class.remove_los_velocity)
    if "convert_distance_to_parallax" in cfg.augmentations:
        augmentations.append(augmentations_class.convert_distance_to_parallax)

    # --- Observational selection (window → subsample → compact) ---
    if "observational_window" in cfg.augmentations:
        augmentations.append(augmentations_class.observational_window)
    if "observational_window_spline" in cfg.augmentations:
        augmentations.append(augmentations_class.observational_window_spline)
    if "observational_window_random" in cfg.augmentations:
        augmentations.append(augmentations_class.observational_window_random)
    if "observed_n_stars" in cfg.augmentations:
        augmentations.append(augmentations_class.subsampling_to_observed_n_stars)
    if "compact_to_attended" in cfg.augmentations:
        augmentations.append(augmentations_class.compact_to_attended)

    # --- Photometric augmentation (magnitudes → errors → apply) ---
    if "sample_magnitudes" in cfg.augmentations:
        augmentations.append(augmentations_class.sample_magnitudes)
    if "sample_obs_error" in cfg.augmentations:
        augmentations.append(augmentations_class.sample_obs_error)
    if "apply_obs_error" in cfg.augmentations:
        augmentations.append(augmentations_class.apply_obs_error)

    # --- v_los masking (must be after apply_obs_error) ---
    if "mask_vlos" in cfg.augmentations:
        augmentations.append(augmentations_class.mask_vlos)

    # --- Symmetry augmentations ---
    if "flip_dirz" in cfg.augmentations:
        augmentations.append(augmentations_class.flip_dirz)

    if "add_noise_to_vcirc" in cfg.augmentations:
        augmentations.append(augmentations_class.add_noise_to_vcirc)
    if "log10_vcirc" in cfg.augmentations:
        augmentations.append(augmentations_class.log10_vcirc)

    # --- Feature concatenations (must be last) ---
    if "concatentate_sigma_error_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatentate_sigma_error_to_sim_data)
    if "concatenate_magnitudes_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_magnitudes_to_sim_data)
    if "concatenate_vlos_mask_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_vlos_mask_to_sim_data)
    if "concatenate_j_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_j_to_sim_data)

    test_data[cfg.sim_data] = test_data[cfg.sim_data].reshape(-1, test_data[cfg.sim_data].shape[-2], test_data[cfg.sim_data].shape[-1])
    test_data['vcirc_kms'] = np.repeat(test_data['vcirc_kms'], 3, axis=0)
    test_data['j'] = test_data['j'].reshape(-1, 1)
    print('Test data sim shape before augmentation: ', test_data[cfg.sim_data].shape)
    for aug in augmentations:
        print(f"Applying augmentation: {aug.__name__}")
        test_data = aug(test_data)
    for k in cfg.parameters_global:
        test_data[k] = np.repeat(test_data[k], 3, axis=0).reshape(-1, 1)
    print('Test data sim shape after augmentation: ', test_data[cfg.sim_data].shape)
    # print('Test data keys shape: ', [test_data[k].shape for k in test_data.keys()])
    # print('Test data attention mask shape: ', test_data['attention_mask'].shape)
    for k in test_data.keys():
        print(f"{k}: {test_data[k].shape}")


    logging.info("Starting Partial-Pooling (global) inference with no composition...")
    print(workflow_global.approximator.inference_network.integrate_kwargs.keys())
    # workflow_global.approximator.inference_network.integrate_kwargs.update({
    #     'method': cfg.method,
    #     'steps': cfg.steps,
    #     # 'compositional_bridge_d1': 1/cfg.inverse_compositional_bridge_d1,
    #     # 'mini_batch_size': cfg.mini_batch_size,
    #     "max_steps": cfg.max_steps,
    #     })
    conditions = {
        "input_a": test_data[cfg.sim_data],         # (300, 300, 15)
        "input_b": test_data["vcirc_kms"],            # (300, 34, 1)  <-- missing
        "summary_attention_mask": test_data["attention_mask"],      # (300, 1, 300)
        "j": test_data["j"],                            # (300, 1)
    }
    global_posterior = workflow_global.sample(
                        num_samples=cfg.n_samples,
                        conditions=conditions,
                        batch_size = cfg.batch_size,
                        kwargs={'summary_attention_mask': test_data['attention_mask']},
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
        mask_posterior = ps['dirz_Triaxial_rotated_halo'] < 0
        ps['dirz_Triaxial_rotated_halo'][mask_posterior] *= -1
        ps['dirx_Triaxial_rotated_halo'][mask_posterior] *= -1
        ps['diry_Triaxial_rotated_halo'][mask_posterior] *= -1


    # ...existing code...
    ps['$M_Disk$'] = 4 * np.pi * ps['Sigma_Disk'] * ps['r_Disk']**2 * ps['z_Disk']
    test_data['$M_Disk$'] = 4 * np.pi * test_data['Sigma_Disk'] * test_data['r_Disk']**2 * test_data['z_Disk']

    cfg.paramater_global_pretty = cfg.paramater_global_pretty + ['$M_D$',]
    paramater_global_pretty = cfg.paramater_global_pretty
    np.savez(os.path.join(cfg.base_dir, cfg.results_dir, 'posterior.npz'), **ps)


    # --- r200 from posterior samples ---
    # ps[param] has shape (n_observations, n_samples): flatten, compute, reshape back
    # Adjust parameter names below to match your cfg.parameters_global entries
    # _rho0_ps  = ps['rho_TwoPowerTriaxial_halo'].ravel()        # <-- adjust key name
    # _a1_ps    = ps['a_TwoPowerTriaxial_halo'].ravel()          # <-- adjust key name
    # _gamma_ps = ps['gamma_TwoPowerTriaxial_halo'].ravel()       # <-- adjust key name

    # print("Computing r200 and M200 on posterior samples (vectorized)...")
    # _r200_ps = find_r200_vec(_rho0_ps, _a1_ps, _gamma_ps, rho_crit=rho_crit)
    # _m200_ps = (800*np.pi/3) * rho_crit * _r200_ps**3

    # ps['$r_{200}$'] = _r200_ps.reshape(ps['rho_TwoPowerTriaxial_halo'].shape)
    # ps['$M_{200}$'] = _m200_ps.reshape(ps['rho_TwoPowerTriaxial_halo'].shape)


    # # --- r200 from true (test) parameters ---
    # # test_data[param] has shape (n_observations, 1)
    # _rho0_td  = test_data['rho_TwoPowerTriaxial_halo'].ravel()   # <-- adjust key name
    # _a1_td    = test_data['a_TwoPowerTriaxial_halo'].ravel()     # <-- adjust key name
    # _gamma_td = test_data['gamma_TwoPowerTriaxial_halo'].ravel()  # <-- adjust key name

    # print("Computing r200 and M200 on test data (vectorized)...")
    # _r200_td = find_r200_vec(_rho0_td, _a1_td, _gamma_td, rho_crit=rho_crit)
    # _m200_td = (800*np.pi/3) * rho_crit * _r200_td**3

    # test_data['$r_{200}$'] = _r200_td.reshape(test_data['rho_TwoPowerTriaxial_halo'].shape)
    # test_data['$M_{200}$'] = _m200_td.reshape(test_data['rho_TwoPowerTriaxial_halo'].shape)

    # # Register for plotting
    # cfg.paramater_global_pretty = cfg.paramater_global_pretty + ['$r_{200}$', '$M_{200}$']
    

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
    # #histograms
    # global_posterior_stream_1 = {k: ps[k] for k in list(ps.keys())[:4]}
    # test_data_stream_1 = {k: test_data[k] for k in list(global_posterior_stream_1.keys())}
    # fig_1 = bf.diagnostics.plots.calibration_histogram(
    #     estimates=global_posterior_stream_1, 
    #     targets=test_data_stream_1,
    #     variable_names=cfg.paramater_global_pretty[:4]
    #     # variable_names = param_names_global
    # )
    # for ax in fig_1.get_axes():
    #     ax.grid(False)
    # fig_1.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'histograms_1.pdf'))
    # plt.show()
    # global_posterior_stream_2 = {k: ps[k] for k in list(ps.keys())[4:]}
    # test_data_stream_2 = {k: test_data[k] for k in list(global_posterior_stream_2.keys())}
    # fig_2 = bf.diagnostics.plots.calibration_histogram(
    #     estimates=global_posterior_stream_2, 
    #     targets=test_data_stream_2,
    #     variable_names=cfg.paramater_global_pretty[4:]
    #     # variable_names = param_names_global
    # )
    # for ax in fig_2.get_axes():
    #     ax.grid(False)
    # fig_2.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'histograms_2.pdf'))
    # plt.show()
    # print('Saved histograms plot')

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
