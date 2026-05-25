from autocvd import autocvd
# autocvd(num_gpus = 1)
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.core.config_store import ConfigStore
import numpy as np

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"
import bayesflow as bf
import keras
from scipy import  special 
import jax


import logging
logging.getLogger('bayesflow').setLevel(logging.DEBUG)

from config.EvalConfig import EvalConfig
from utils.utils_train_jax import AugmentationsClass #we will need to use the augmentations on the test_set
from utils.utils_train_jax_new_rotationcurve_fixedvlosmask import AugmentationsClass #we will need to use the augmentations on the test_set
from utils.custom_summary_network import SetTransformer, FusionNetwork


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
    
    N_multistream = 1000
    test_data_multistream_path = os.path.join(cfg.base_dir, cfg.data_dir, f"simulation_multistream_{N_multistream}.npz")
    test_data_multistream = dict(np.load(test_data_multistream_path, allow_pickle=True))
    test_data_multistream_rotation_curve = dict(np.load(f'./data/plots/agama_rotcurv_multistream/{N_multistream}/rotation_curves.npz'))
    augmentations_class = AugmentationsClass(cfg)
    mask_r_kpc = (augmentations_class.obs_R >5.5)
    test_data_multistream['vcirc_kms'] = test_data_multistream_rotation_curve['vcirc_kms'][:, mask_r_kpc, None] #extra dimension (n_observation, len_r_kpc, 1)
    # Boolean mask: True where a simulation is NOT NaN (shape: n_simulations)
    valid_mask = ~np.isnan(test_data_multistream['sim_data_carthesian']).any(axis=(-1, -2, -3))

    # Filter every key in the dict along the simulation axis
    test_data_multistream = {k: v[valid_mask] for k, v in test_data_multistream.items()}

    print('Test data sim shape before augmentation: ', test_data_multistream[cfg.sim_data].shape)

    print('Loading test data from ', test_data_path)
    keys_to_drop = set(test_data_multistream.keys()) - set(param_names_global) - {sim_data} - set(inference_conditions)
    keys_to_drop = list(keys_to_drop) 
    # print('Test data keys and shape: ', test_data_multistream.keys(), test_data_multistream[list(test_data_multistream.keys())[0]].shape)
    other_things = ['attention_mask', 'magnitudes', 'vlos_mask', 'vlos_error', 'vlos_mask', 'vcirc_kms', ]
    keys_to_drop = set(test_data_multistream.keys()) - set(param_names_global) - {sim_data} - set(inference_conditions) -  set(other_things)
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
    workflow_global.approximator = keras.models.load_model(model_path)
    test_data_multistream = {k: test_data_multistream[k] for k in cfg.parameters_global + [cfg.sim_data, "j", "vcirc_kms"] }
    # Augmentation
    augmentations_class = AugmentationsClass(cfg)
    augmentations_class.key = jax.random.PRNGKey(42)
    augmentations = []

    
    if "cut_to_300_particles" in cfg.augmentations:
        augmentations.append(augmentations_class.cut_to_300_particles)
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

    test_data_multistream[cfg.sim_data] = test_data_multistream[cfg.sim_data].reshape(-1, test_data_multistream[cfg.sim_data].shape[-2], test_data_multistream[cfg.sim_data].shape[-1])
    test_data_multistream['vcirc_kms'] = np.repeat(test_data_multistream['vcirc_kms'], 3, axis=0)
    test_data_multistream['j'] = test_data_multistream['j'].reshape(-1, 1)
    print('Test data sim shape before augmentation: ', test_data_multistream[cfg.sim_data].shape)
    for aug in augmentations:
        print(f"Applying augmentation: {aug.__name__}")
        test_data_multistream = aug(test_data_multistream)
    for k in cfg.parameters_global:
        test_data_multistream[k] = np.repeat(test_data_multistream[k], 3, axis=0).reshape(-1, 1)
    for k in test_data_multistream.keys():
        test_data_multistream[k] = np.array(test_data_multistream[k])

    #now we load gaiastream
    test_data_path_gaia = '/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz'
    print('Loading test data from ', test_data_path_gaia)
    test_data_gaia = dict(np.load(test_data_path_gaia, allow_pickle=True))
    test_data_gaia = {k: test_data_gaia[k] for k in [cfg.sim_data, "j", "attention_mask", "magnitudes", 'vlos_error', 'vlos_mask'] }
    for k in [cfg.sim_data, "attention_mask", "magnitudes", 'vlos_error', 'vlos_mask']:
        print(f"{k} shape before truncation: {test_data_gaia[k].shape}")
        if len(test_data_gaia[k].shape) == 2:
            test_data_gaia[k] = test_data_gaia[k][:, :300]
            if k == "vlos_mask":
                test_data_gaia[k] = test_data_gaia[k][:, None, :]
        elif len(test_data_gaia[k].shape) == 3:
            test_data_gaia[k] = test_data_gaia[k][:, :, :300]
        elif len(test_data_gaia[k].shape) == 4:
            test_data_gaia[k] = test_data_gaia[k][:, :, :300]
        print(f"{k} shape after truncation: {test_data_gaia[k].shape}")

    test_data_gaia['vcirc_kms'] = augmentations_class.obs_Vc[None, :, None]
    print('Test data vcirc_kms shape after adding to test data: ', test_data_gaia['vcirc_kms'].shape)

    augmentations_gaia = []
    augmentations_gaia.append(augmentations_class.sample_obs_error)
    augmentations_gaia.append(augmentations_class.concatentate_sigma_error_to_sim_data)
    augmentations_gaia.append(augmentations_class.concatenate_magnitudes_to_sim_data)
    augmentations_gaia.append(augmentations_class.concatenate_vlos_mask_to_sim_data)
    augmentations_gaia.append(augmentations_class.concatenate_j_to_sim_data)
    augmentations_gaia.append(augmentations_class.log10_vcirc)

    test_data_gaia[cfg.sim_data] = test_data_gaia[cfg.sim_data].reshape(-1, test_data_gaia[cfg.sim_data].shape[-2], test_data_gaia[cfg.sim_data].shape[-1])
    n_streams = len(cfg.target_streams.keys())
    test_data_gaia['vcirc_kms'] = np.repeat(test_data_gaia['vcirc_kms'], n_streams, axis=0)
    print('test data vcirc_kms shape after tiling: ', test_data_gaia['vcirc_kms'].shape)
    test_data_gaia['j'] = test_data_gaia['j'].reshape(-1, 1)
    print('Test data sim shape before augmentation: ', test_data_gaia[cfg.sim_data].shape)
    for aug in augmentations_gaia:
        print(f"Applying augmentation: {aug.__name__}")
        test_data_gaia = aug(test_data_gaia)
        # Manually override the vlos error immediately after sample_obs_error is applied
        if aug.__name__ == "sample_obs_error":
            # Match the exact shape of the newly created v_los sigma_errors (batch_size, n_particles)
            target_shape = test_data_gaia["sigma_errors"][:, :, -1].shape
            
            # Reshape the real data and masks to match
            vlos_mask_np = np.array(test_data_gaia["vlos_mask"]).reshape(target_shape)
            vlos_error_np = np.array(test_data_gaia["vlos_error"]).reshape(target_shape)
            
            # Use standard numpy where to safely overwrite
            test_data_gaia["sigma_errors"] = np.array(test_data_gaia["sigma_errors"])
            test_data_gaia["sigma_errors"][:, :, -1] = np.where(
                vlos_mask_np, 
                vlos_error_np, 
                test_data_gaia["sigma_errors"][:, :, -1]
            )
            print("Successfully executed manual override of real v_los errors.")

    # test_data_gaia[cfg.sim_data] = test_data_gaia[cfg.sim_data].reshape(-1, len(cfg.target_streams.keys()), test_data_gaia[cfg.sim_data].shape[-2], test_data_gaia[cfg.sim_data].shape[-1])
    # test_data_gaia['vcirc_kms'] = test_data_gaia['vcirc_kms'].reshape(-1, len(cfg.target_streams.keys()), test_data_gaia['vcirc_kms'].shape[-2], test_data_gaia['vcirc_kms'].shape[-1])
    # test_data_gaia['j'] = test_data_gaia['j'].reshape(-1,len(cfg.target_streams.keys()), 1)
    
    #Using summary network
    from keras.ops import convert_to_numpy
    from bayesflow.metrics.functional import maximum_mean_discrepancy
    observed_samples = convert_to_numpy(workflow_global.approximator.summarize(test_data_gaia,
                                                                               kwargs={'attention_mask': test_data_gaia['attention_mask']}))
    reference_samples = convert_to_numpy(workflow_global.approximator.summarize(test_data_multistream,
                                                                               kwargs={'attention_mask': test_data_multistream['attention_mask'],})) 
    import random
    import torch
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # optional stricter determinism
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    
    distance_observed, distance_null = bf.diagnostics.bootstrap_comparison(observed_samples=observed_samples,
                                                                           reference_samples=reference_samples,
                                                                           comparison_fn=maximum_mean_discrepancy,
                                                                        #    num_null_samples=500,
                                                                           )
    # Convert to numpy to avoid Tensor vs ndarray comparison error
    if hasattr(distance_observed, 'numpy'):
        distance_observed = distance_observed.numpy()
    else:
        distance_observed = np.array(distance_observed)
    
    if hasattr(distance_null, 'numpy'):
        distance_null = distance_null.numpy()
    else:
        distance_null = np.array(distance_null)
    print(f"Distance observed: {distance_observed}, Distance null mean: {np.mean(distance_null)}, p-value: {(distance_null >= distance_observed).mean()}")
    print('Types of distances: ', type(distance_observed), type(distance_null))


    def mmd_hypothesis_test_numpy(mmd_null, mmd_observed, alpha_level=0.05, bw_factor=1.5):
        mmd_null = np.asarray(mmd_null, dtype=np.float64).reshape(-1)
        mmd_observed = float(np.asarray(mmd_observed).reshape(()))

        f = plt.figure(figsize=(10, 5))
        kde = sns.kdeplot(mmd_null, fill=False, linewidth=0, bw_adjust=bw_factor)
        sns.kdeplot(mmd_null, fill=True, alpha=0.12, color="#132a70", bw_adjust=bw_factor)

        plt.vlines(
            x=mmd_observed,
            ymin=0,
            ymax=plt.gca().get_ylim()[1],
            color="red",
            linewidth=3,
            label="Observed data",
        )

        mmd_critical = float(np.quantile(mmd_null, 1 - alpha_level))
        kde_x, kde_y = kde.lines[0].get_data()
        kde_x = np.asarray(kde_x, dtype=np.float64)
        kde_y = np.asarray(kde_y, dtype=np.float64)
        plt.fill_between(
            kde_x, kde_y,
            where=(kde_x >= mmd_critical),
            interpolate=True,
            color="orange",
            alpha=0.5,
            label=f"{int(alpha_level * 100)}% rejection area",
        )
        plt.vlines(x=mmd_critical, color="orange", linewidth=3, ymin=0, ymax=plt.gca().get_ylim()[1])

        sns.kdeplot(mmd_null, fill=False, linewidth=3, color="#132a70", label=r"$H_0$", bw_adjust=bw_factor)
        plt.xlabel("MMD", fontsize=20)
        plt.ylabel("Density", fontsize=20)
        # plt.yticks([])
        plt.tick_params(axis="both", which="major", labelsize=16)
        plt.legend(fontsize=14)
        sns.despine()
        return f

    # fig = bf.diagnostics.mmd_hypothesis_test(mmd_null=distance_null, mmd_observed=distance_observed,)
    fig = mmd_hypothesis_test_numpy(mmd_null=distance_null, mmd_observed=distance_observed,)
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'distance_observed_vs_null.pdf'), bbox_inches='tight')
    print('Plot saved to ', os.path.join(cfg.base_dir, cfg.results_dir, 'distance_observed_vs_null_new.pdf'))





    

    

if __name__ == "__main__":
    main()
