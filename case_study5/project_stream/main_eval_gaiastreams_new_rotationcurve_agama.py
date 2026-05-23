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
from chainconsumer import Chain, ChainConsumer, ChainConfig
import pandas as pd
from scipy import special
import jax.numpy as jnp 

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"
import keras
import bayesflow as bf
import jax


import logging
logging.getLogger('bayesflow').setLevel(logging.DEBUG)

from config.EvalConfig import EvalConfig
from utils.utils_train_jax_new_rotationcurve_fixedvlosmask import AugmentationsClass #we will need to use the augmentations on the test_set
from utils.custom_summary_network import SetTransformer, FusionNetwork

cs = ConfigStore.instance()
cs.store(name="eval_config", node=EvalConfig)

@hydra.main(version_base=None, config_path="config", config_name="eval_config_gaia_new_rotationcurve_agama",)
def main(cfg: EvalConfig):

    print(cfg)
    print("##############")
    model_path = os.path.join(cfg.base_dir, cfg.model_dir, 'global_model.keras' )
    print('Loading model from ', model_path)
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
    # with open(os.path.join(cfg.base_dir, cfg.model_dir, '.hydra', 'config.yaml'), "r") as f:
    #     model_config = yaml.safe_load(f)
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

    test_data_path = '/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz'
    print('Loading test data from ', test_data_path)
    test_data = dict(np.load(test_data_path, allow_pickle=True))
    test_data = {k: test_data[k] for k in [cfg.sim_data, "j", "attention_mask", "magnitudes", 'vlos_error', 'vlos_mask'] }
    for k in [cfg.sim_data, "attention_mask", "magnitudes", 'vlos_error', 'vlos_mask']:
        print(f"{k} shape before truncation: {test_data[k].shape}")
        if len(test_data[k].shape) == 2:
            test_data[k] = test_data[k][:, :300]
            if k == "vlos_mask":
                test_data[k] = test_data[k][:, None, :]
        elif len(test_data[k].shape) == 3:
            test_data[k] = test_data[k][:, :, :300]
        elif len(test_data[k].shape) == 4:
            test_data[k] = test_data[k][:, :, :300]
        print(f"{k} shape after truncation: {test_data[k].shape}")

    augmentations_class = AugmentationsClass(cfg)
    mask_r_kpc = (augmentations_class.obs_R >5.5)
    test_data['vcirc_kms'] = augmentations_class.obs_Vc[None,mask_r_kpc, None]
    print('Test data vcirc_kms shape after adding to test data: ', test_data['vcirc_kms'].shape)


    # print('Test data keys and shape: ', test_data.keys(), test_data[list(test_data.keys())[0]].shape)
    other_things = ['attention_mask', 'magnitudes', 'vlos_mask', 'vlos_error', 'vlos_mask', 'vcirc_kms', ]
    keys_to_drop = set(test_data.keys()) - set(param_names_global) - {sim_data} - set(inference_conditions) -  set(other_things)
    keys_to_drop = list(keys_to_drop) 

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
    summary_network_a = SetTransformer(
            # summary_dim=cfg.global_model.summary_dim,
            # embed_dims=(cfg.global_model.embed_dims, cfg.global_model.embed_dims),
            # num_heads=(
            #     cfg.global_model.num_heads,
            #     cfg.global_model.num_heads,
            # ),
            # mlp_depths=(cfg.global_model.mlp_depths, cfg.global_model.mlp_depths),
            # mlp_widths=(cfg.global_model.mlp_widths, cfg.global_model.mlp_widths),
            # dropout=cfg.global_model.dropout,
        )
    summary_network_b = bf.networks.TimeSeriesTransformer()
    head = keras.Sequential(
        [bf.networks.MLP(widths=[32, 32, 32]), keras.layers.Dense(units=55)]
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
    workflow_global.approximator = keras.saving.load_model(model_path)
    # Augmentation

    augmentations_class.key = jax.random.PRNGKey(42)
#     augmentations_multistream = []

#     augmentations_multistream.append(augmentations_class.convert_distance_to_parallax)

# # --- Observational selection (window → subsample → compact) ---
#     augmentations_multistream.append(augmentations_class.observational_window)
#     augmentations_multistream.append(augmentations_class.subsampling_to_observed_n_stars)
#     augmentations_multistream.append(augmentations_class.compact_to_attended)

#     augmentations_multistream.append(augmentations_class.sample_magnitudes)
#     augmentations_multistream.append(augmentations_class.sample_obs_error)
#     augmentations_multistream.append(augmentations_class.apply_obs_error)

#     augmentations_multistream.append(augmentations_class.mask_vlos)


#     augmentations_multistream.append(augmentations_class.add_noise_to_vcirc)
#     augmentations_multistream.append(augmentations_class.log10_vcirc)

#         # --- Feature concatenations (must be last) ---
#     augmentations_multistream.append(augmentations_class.concatentate_sigma_error_to_sim_data)
#     augmentations_multistream.append(augmentations_class.concatenate_magnitudes_to_sim_data)
#     augmentations_multistream.append(augmentations_class.concatenate_vlos_mask_to_sim_data)
#     augmentations_multistream.append(augmentations_class.concatenate_j_to_sim_data)
    

#     #trying to get the same random key as in hyperparameter tuning:
#     #now we load the multistream test set
#     data_dir_multistream = 'streams/data_multistream_gala_new/'
#     N_multistream = 333
#     test_data_multistream_path = os.path.join(cfg.base_dir, data_dir_multistream, f"simulation_multistream_{N_multistream}.npz")
#     test_data_multistream = dict(np.load(test_data_multistream_path, allow_pickle=True))
#     test_data_multistream_rotation_curve = dict(np.load(f'./data/plots/gala_rotcurv_multistream/{N_multistream}/rotation_curves.npz'))
#     test_data_multistream['vcirc_kms'] = test_data_multistream_rotation_curve['vcirc_kms'][:, :, None] #extra dimension (n_observation, len_r_kpc, 1)

#     test_data_multistream[cfg.sim_data] = test_data_multistream[cfg.sim_data].reshape(-1, test_data_multistream[cfg.sim_data].shape[-2], test_data_multistream[cfg.sim_data].shape[-1])
#     test_data_multistream['vcirc_kms'] = np.repeat(test_data_multistream['vcirc_kms'], 3, axis=0)
#     test_data_multistream['j'] = test_data_multistream['j'].reshape(-1, 1)
#     print('Test data sim shape before augmentation: ', test_data_multistream[cfg.sim_data].shape)
#     for aug in augmentations_multistream:
#         print(f"Applying augmentation: {aug.__name__}")
#         test_data_multistream = aug(test_data_multistream)
#     for k in cfg.parameters_global:
#         test_data_multistream[k] = np.repeat(test_data_multistream[k], 3, axis=0).reshape(-1, 1)
#     for k in test_data_multistream.keys():
#         test_data_multistream[k] = np.array(test_data_multistream[k])


    augmentations = []
    if "remove_los_velocity" in cfg.augmentations:
        augmentations.append(augmentations_class.remove_los_velocity)
    if "sample_obs_error" in cfg.augmentations:
        augmentations.append(augmentations_class.sample_obs_error)
    if "concatentate_sigma_error_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatentate_sigma_error_to_sim_data)
    if "concatenate_magnitudes_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_magnitudes_to_sim_data)
    if "concatenate_vlos_mask_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_vlos_mask_to_sim_data)
    if "concatenate_j_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_j_to_sim_data)
    if "log10_vcirc" in cfg.augmentations:
        augmentations.append(augmentations_class.log10_vcirc)

    #reshape the streams dimensions
    test_data[cfg.sim_data] = test_data[cfg.sim_data].reshape(-1, test_data[cfg.sim_data].shape[-2], test_data[cfg.sim_data].shape[-1])
    n_streams = len(cfg.target_streams.keys())
    test_data['vcirc_kms'] = np.repeat(test_data['vcirc_kms'], n_streams, axis=0)
    print('test data vcirc_kms shape after tiling: ', test_data['vcirc_kms'].shape)
    test_data['j'] = test_data['j'].reshape(-1, 1)
    print('Test data sim shape before augmentation: ', test_data[cfg.sim_data].shape)
    for aug in augmentations:
        print(f"Applying augmentation: {aug.__name__}")
        test_data = aug(test_data)
        # Manually override the vlos error immediately after sample_obs_error is applied
        if aug.__name__ == "sample_obs_error":
            # Match the exact shape of the newly created v_los sigma_errors (batch_size, n_particles)
            target_shape = test_data["sigma_errors"][:, :, -1].shape
            
            # Reshape the real data and masks to match
            vlos_mask_np = np.array(test_data["vlos_mask"]).reshape(target_shape)
            vlos_error_np = np.array(test_data["vlos_error"]).reshape(target_shape)
            
            # Use standard numpy where to safely overwrite
            test_data["sigma_errors"] = np.array(test_data["sigma_errors"])
            test_data["sigma_errors"][:, :, -1] = np.where(
                vlos_mask_np, 
                vlos_error_np, 
                test_data["sigma_errors"][:, :, -1]
            )
            print("Successfully executed manual override of real v_los errors.")

    test_data[cfg.sim_data] = test_data[cfg.sim_data].reshape(-1, len(cfg.target_streams.keys()), test_data[cfg.sim_data].shape[-2], test_data[cfg.sim_data].shape[-1])
    test_data['vcirc_kms'] = test_data['vcirc_kms'].reshape(-1, len(cfg.target_streams.keys()), test_data['vcirc_kms'].shape[-2], test_data['vcirc_kms'].shape[-1])
    test_data['j'] = test_data['j'].reshape(-1,len(cfg.target_streams.keys()), 1)
    print('Test data sim shape after augmentation: ', test_data[cfg.sim_data].shape)
    print('Test data keys: ', test_data.keys())
    for k in test_data.keys():
        print('##########')
        print(f"{k} shape: {test_data[k].shape}")
    with open(os.path.join(cfg.base_dir, cfg.data_dir, '.hydra', 'config.yaml'), "r") as f:
        test_sim_config = yaml.safe_load(f)

    def prior_global_score(x, time, cfg=cfg, test_sim_config=test_sim_config):
        
        score = {}
        
        for k in cfg.parameters_global:
            print(f"Computing prior score for {k} with type {test_sim_config['priors_global'][k]['type']} at time {time}")
            # print(f"Computing prior score for {k} with type {test_sim_config['priors_global'][k]['type']}")
            if test_sim_config['priors_global'][k]['type'] == 'uniform':
                score[k] = (1-time)*jnp.zeros_like(x[k])
            elif test_sim_config['priors_global'][k]['type'] == 'normal':
                print(f"Parameters for normal prior of {k}: mean={test_sim_config['priors_global'][k]['prior_parameters'][0]}, std={test_sim_config['priors_global'][k]['prior_parameters'][1]}")
                mean = test_sim_config['priors_global'][k]['prior_parameters'][0]
                std = test_sim_config['priors_global'][k]['prior_parameters'][1]
                score[k] = -(1-time)*(x[k] - mean) / std**2 
        return score

    logging.info("Starting Partial-Pooling (global) inference...")

    # Reshape all data to have a single batch dimension (n_sims * n_streams)
    n_sims = test_data[cfg.sim_data].shape[0]
    n_streams = test_data[cfg.sim_data].shape[1]
    
    flat_input_a = test_data[cfg.sim_data].reshape(
        n_sims * n_streams, 
        test_data[cfg.sim_data].shape[-2], 
        test_data[cfg.sim_data].shape[-1]
    )
    # Reshape vcirc_kms to match the flattened batch dimension
    flat_input_b = test_data["vcirc_kms"].reshape(
        n_sims * n_streams,
        test_data["vcirc_kms"].shape[-2],
        test_data["vcirc_kms"].shape[-1]
    )
    flat_j = test_data["j"].reshape(n_sims * n_streams, 1)
    
    # Reshape attention mask to be (batch_size, 1, n_stars)
    flat_mask = test_data["attention_mask"].reshape(
        n_sims * n_streams, 1, test_data["attention_mask"].shape[-1]
    )

    # 1. Create a raw conditions dictionary using pre-adapter keys
    raw_conditions = {
            cfg.sim_data: flat_input_a,
            "vcirc_kms": flat_input_b,
            "attention_mask": flat_mask,
            "j": flat_j,
        }

    # 2. Use _prepare_conditions to process data through Adapter + SummaryNetwork + Standardizers
    resolved_flat, _, _ = workflow_global.approximator._prepare_conditions(data=raw_conditions)
    
    # Convert fully prepared tensor to numpy array before reshaping
    resolved_flat = np.array(resolved_flat)

    # 3. Reshape standard-resolved conditions back to compositional shape (n_datasets, n_compositional_conditions, ...)
    final_summary_outputs = resolved_flat.reshape(
        n_sims, n_streams, resolved_flat.shape[-1]
    )
    print('Final summary outputs shape after reshape: ', final_summary_outputs.shape)

    # 4. Run compositional sampling using only summary_outputs (this perfectly bypasses the reshape bug)
    global_posterior = workflow_global.compositional_sample(
                        num_samples=cfg.n_samples,
                        # conditions=None,
                        summaries=final_summary_outputs,
                        method=cfg.method,
                        steps=cfg.steps,
                        compositional_bridge_d1=1/cfg.inverse_compositional_bridge_d1,
                        compute_prior_score=prior_global_score,
                        batch_size=cfg.batch_size,
                        )
    os.makedirs(name= os.path.join(cfg.base_dir, cfg.results_dir), exist_ok=True)
    ps = global_posterior.copy()
    # q_min = 0.5
    # q_max = 1.5
    # r_posterior = np.sqrt(ps['dirx_Triaxial_rotated_halo']**2 + ps['diry_Triaxial_rotated_halo']**2 + ps['dirz_Triaxial_rotated_halo']**2)
    # u_uniform_posterior = special.erf(r_posterior/np.sqrt(2)) - np.sqrt(2/np.pi)*r_posterior*np.exp(-(r_posterior**2)/2)
    # ps['$q_{NFW}$'] = q_min + (q_max-q_min)*u_uniform_posterior
    # param_names_global = cfg.parameters_global + ['$q_{NFW}$']
    # cfg.paramater_global_pretty = cfg.paramater_global_pretty + ['$q_{NFW}$']
    np.savez(os.path.join(cfg.base_dir, cfg.results_dir, 'global_posterior.npz'), **ps)
    ###############
    # PLOTS GLOBAL#
    ###############
    from astropy import units as u
    obs_R   = augmentations_class.obs_R[mask_r_kpc]
    obs_Vc  = augmentations_class.obs_Vc[mask_r_kpc]
    obs_sVc = augmentations_class.obs_sVc[mask_r_kpc]
    r_array = obs_R * u.kpc
    N_sample = 100
    N_obs = len(obs_R)
    params = ['rho_TwoPowerTriaxial_halo', 'gamma_TwoPowerTriaxial_halo', 
                'a_TwoPowerTriaxial_halo', 'q_TwoPowerTriaxial_halo', 
                'r_Disk', 'z_Disk', 'Sigma_Disk']
    all_vcirc  = np.zeros((N_sample, N_obs))   # km/s, stored for reuse
    for k in params:
        print('Shape posterior samples for ', k, ': ', ps[k].shape)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert JAX array to standard NumPy array for Agama
    obs_R_np = np.array(obs_R)
    points = np.column_stack((obs_R_np, obs_R_np*0, obs_R_np*0))

    # for i in tqdm(range(N_sample)):
    #     # Extract purely as standard Python floats!
    #     p = {k: float(ps[k][0, i, 0]) for k in params}
    #     print(f"Sample {i}: ", p)

    #     pot_params = [ dict(type='Spheroid', 
    #                         scaleRadius=75/1e3, 
    #                         densityNorm=9.6e10,
    #                         gamma=0, 
    #                         alpha=1, 
    #                         beta=1.8, 
    #                         cutoffStrength=2,
    #                         outerCutoffRadius=2.1,
    #                         axisRatioY=1.0,
    #                         axisRatioZ=0.5), #bulge
    #                     dict(type='Spheroid', 
    #                         scaleRadius=p['a_TwoPowerTriaxial_halo'], 
    #                         densityNorm=p['rho_TwoPowerTriaxial_halo'],
    #                         gamma=p['gamma_TwoPowerTriaxial_halo'], 
    #                         alpha=1, 
    #                         beta=3, #fixed 
    #                         cutoffStrength=2, #fixed
    #                         outerCutoffRadius=np.inf, #fixed
    #                         axisRatioY=1.0, #fixed
    #                         axisRatioZ=p['q_TwoPowerTriaxial_halo']), #halo
    #                     dict(type='Disk', 
    #                         scaleRadius=p['r_Disk'], 
    #                         scaleHeight=p['z_Disk'], 
    #                         surfaceDensity=p['Sigma_Disk'],
    #                         sersicIndex=1, #fixed
    #                         innerCutoffRadius = 0, #fixed
    #                         ) #Disk 
    #                 ]
    #     pot = agama.Potential(*pot_params) 
        
    #     v_circ_kms = np.sqrt(-obs_R_np*pot.force(points)[:,0])
    #     all_vcirc[i] = v_circ_kms

    # # ── Rotation-curve plot (MOVED OUTSIDE THE TQDM LOOP) ─────────────────────────
    # for i in range(N_sample):
    #     ax.plot(obs_R_np, all_vcirc[i], color='grey', alpha=0.5, lw=0.4)
    
    # # Calculate and plot the posterior mean and standard deviation
    # mean_vcirc = np.mean(all_vcirc, axis=0)
                
    # ax.errorbar(obs_R, obs_Vc, yerr=3*obs_sVc, fmt='o', color='red',
    #             ms=3, lw=1, capsize=2, label='Observed ±3σ', zorder=5)
    # ax.set_xlabel('Radius (kpc)')
    # ax.set_ylabel('Circular Velocity (km/s)')
    # fig.savefig(os.path.join(cfg.results_dir, f'global_rotation_curve.pdf'))
    # plt.close(fig)
        # ── Rotation-curve plot ───────────────────────────────────────────────────────
        

        # for i in range(N_sample):
        #     ax.plot(r_array, all_vcirc[i], color='grey', alpha=0.5, lw=0.4)
    
    # Calculate and plot the posterior mean and standard deviation
    # mean_vcirc = np.mean(all_vcirc, axis=0)
    # std_vcirc = np.std(all_vcirc, axis=0)
    
    # ax.plot(r_array.value, mean_vcirc, color='blue', lw=2, label='Posterior Mean')
    # ax.fill_between(
    #     r_array.value, 
    #     mean_vcirc - 3* std_vcirc, 
    #     mean_vcirc + 3* std_vcirc, 
    #     color='blue', 
    #     alpha=0.3, 
    #     label='Posterior ±3σ'
    # )
                
    # ax.errorbar(obs_R, obs_Vc, yerr=3*obs_sVc, fmt='o', color='red',
    #             ms=3, lw=1, capsize=2, label='Zhou et al. (2023) ±3σ', zorder=5)
    # ax.set_xlabel('Radius [kpc]')
    # ax.set_ylabel('Circular Velocity [km/s]')
    # ax.legend()
    
                
    # # ax.errorbar(obs_R, obs_Vc, yerr=obs_sVc, fmt='o', color='red',
    # #             ms=3, lw=1, capsize=2, label='Observed ±3σ', zorder=5)
    # # ax.set_xlabel('Radius (kpc)')
    # # ax.set_ylabel('Circular Velocity (km/s)')
    # fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, f'global_rotation_curve.pdf'))
    # plt.close(fig)


    # df_halo_props = pd.DataFrame({
    #     '$M_{200}$ [$M_\odot$]': M_200_samples,
    #     '$R_{200}$ [kpc]': r_200_samples
    # })
    
    # import corner
    # fig_halo = corner.corner(
    #     df_halo_props, 
    #     color='teal', 
    #     labels=df_halo_props.columns,
    #     smooth = 1.0, 
    #     hist_kwargs={'density': True},
    #     contour_kwargs={'linewidths': 1.5}
    # )
    
    # fig_halo.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'global_M200_r200.pdf'))
    # plt.close(fig_halo)


    print('shapes of posterior samples: ', {k: v.shape for k, v in ps.items()})
    import matplotlib.colors as mcolors
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, 4))
    # colors = [mcolors.to_hex(c) for c in colors]
    colors = [tuple(c[:3]) for c in colors]
    for k in ps.keys():
        ps[k] = ps[k].reshape(-1,)
    df = pd.DataFrame(ps) 
    print('Df columns before renaming: ', df.columns)
    df.columns = list(cfg.paramater_global_pretty)
    print('Df columns after renaming: ', df.columns)
    import corner
    import matplotlib.lines as mlines

    fig = corner.corner(
        df, 
        color=colors[0], 
        labels=df.columns,
        hist_kwargs={'density': True},
        contour_kwargs={'linewidths': 1.5}
    )
    legend_handles = [mlines.Line2D([], [], color=colors[0], label='Global')]

    #Single stream posteriors
    for i, stream_name in enumerate(cfg.target_streams.keys()):
        print(f"Starting inference for stream {stream_name}...")
        test_data_stream = {"input_a": test_data[cfg.sim_data][:, cfg.target_streams[stream_name], :, :], 
                            "j": test_data["j"][:, cfg.target_streams[stream_name], :],
                            "input_b": test_data["vcirc_kms"][:, cfg.target_streams[stream_name], :, :],
                            "attention_mask": test_data["attention_mask"][:, cfg.target_streams[stream_name], :],
                            }
        attention_mask_stream = test_data['attention_mask'][cfg.target_streams[stream_name], :, :].reshape(1, 1, -1)
        test_data_stream["summary_attention_mask"] = attention_mask_stream
        print('test data stream shapes: ', {k: v.shape for k, v in test_data_stream.items()})
        print('we should see also the magnitude and sigma concatenated, and vlos_mask if used')
        # attention_mask_stream = test_data['attention_mask'][cfg.target_streams[stream_name], :, :].reshape(1, -1)
        # attention_mask_stream = test_data['attention_mask'][cfg.target_streams[stream_name], :, :].reshape(1, 1, -1)
        print('attention mask stream shape: ', attention_mask_stream.shape)
        posterior_stream = workflow_global.sample(
                            num_samples=cfg.n_samples,
                            conditions=test_data_stream,
                            kwargs={'summary_attention_mask': attention_mask_stream}
                            )
        ps_stream = posterior_stream.copy()
    # ...existing code...

        np.savez(os.path.join(cfg.base_dir, cfg.results_dir, f'{stream_name}_posterior.npz'), **ps_stream)
        print(f'Saved posterior samples for stream {stream_name}')
        for k in ps_stream.keys():
            ps_stream[k] = ps_stream[k].reshape(-1,)
        df_stream = pd.DataFrame(ps_stream) 
        df_stream.columns = list(cfg.paramater_global_pretty)
        corner.corner(
            df_stream, 
            fig=fig, 
            color=colors[i+1], 
            hist_kwargs={'density': True},
            contour_kwargs={'linewidths': 1.5}
        )
        legend_handles.append(mlines.Line2D([], [], color=colors[i+1], label=stream_name))
    fig.legend(handles=legend_handles, loc='upper right', fontsize=18, bbox_to_anchor=(0.95, 0.95))
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, f'global_cornerplot.pdf'))
    print(f'Saved global corner plot with all streams in pathc: {os.path.join(cfg.base_dir, cfg.results_dir, "global_cornerplot.pdf")}')




if __name__ == "__main__":
    main()