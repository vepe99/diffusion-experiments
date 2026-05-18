from autocvd import autocvd
autocvd(num_gpus = 1)
import os

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
import bayesflow as bf
import keras

from scipy import  special 
import jax.numpy as jnp 

import logging
logging.getLogger('bayesflow').setLevel(logging.DEBUG)

from config.EvalConfig import EvalConfig
from utils.utils_train_jax_new_rotationcurve_fixedvlosmask import AugmentationsClass #we will need to use the augmentations on the test_set
from utils.custom_summary_network import SetTransformer, FusionNetwork


cs = ConfigStore.instance()
cs.store(name="eval_config", node=EvalConfig)



def _patch_build_config_batch_size(obj, new_batch_size=2):
    """Recursively walk a deserialized config dict and replace the first
    element (batch size) of any 'input_shape' list inside a 'build_config'
    with `new_batch_size`, so that the model can be loaded on a smaller GPU."""
    if isinstance(obj, dict):
        if "build_config" in obj and isinstance(obj["build_config"], dict):
            bc = obj["build_config"]
            if "input_shape" in bc and isinstance(bc["input_shape"], list):
                shape = bc["input_shape"]
                if len(shape) >= 1 and isinstance(shape[0], int) and shape[0] > new_batch_size:
                    shape[0] = new_batch_size
        for v in obj.values():
            _patch_build_config_batch_size(v, new_batch_size)
    elif isinstance(obj, list):
        for item in obj:
            _patch_build_config_batch_size(item, new_batch_size)


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
    # fixed_model_path = model_path.replace('.keras', '_fixed.keras')
    # if not os.path.exists(fixed_model_path):
    #     with zipfile.ZipFile(model_path, 'r') as zin:
    #         with zipfile.ZipFile(fixed_model_path, 'w') as zout:
    #             for item in zin.infolist():
    #                 data = zin.read(item.filename)
    #                 if item.filename == 'config.json':
    #                     config_str = data.decode('utf-8')
    #                     config_str = config_str.replace(
    #                         '__bayesflow_type__ArrayImpl',
    #                         '__bayesflow_type__ndarray'
    #                     )
    #                     # Patch build_config to use batch_size=1 to avoid OOM during model loading
    #                     config_json = json.loads(config_str)
    #                     _patch_build_config_batch_size(config_json, new_batch_size=2)
    #                     config_str = json.dumps(config_json)
    #                     data = config_str.encode('utf-8')
    #                 zout.writestr(item, data)
    #     print(f"Created fixed model at {fixed_model_path}")
    # return fixed_model_path



@hydra.main(version_base=None, config_path="config", config_name="eval_config_new_rotationcurve_agama",)
def main(cfg: EvalConfig):

    print(cfg)
    print("##############")
    model_path = os.path.join(cfg.base_dir, cfg.model_dir, 'global_model.keras' )
    # Fix ArrayImpl serialization issue in the .keras fil
    model_path = fix_keras_model(model_path, )
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
    test_data_rotation_curve = dict(np.load(f'./data/plots/agama_rotcurv_multistream/{cfg.multistream_n_simulation}/rotation_curves.npz'))
    mask_r_kpc = (augmentations_class.obs_R >5.5)&(augmentations_class.obs_R<18.0)
    test_data['vcirc_kms'] = test_data_rotation_curve['vcirc_kms'][:, mask_r_kpc, None] #extra dimension (n_observation, len_r_kpc, 1)
    for k in test_data.keys():
        print(f"{k}: {test_data[k].shape}")
    print('Remove index of bad simulation')
    # Boolean mask: True where a simulation is NOT NaN (shape: n_simulations)
    valid_mask = ~np.isnan(test_data['sim_data_carthesian']).any(axis=(-1, -2, -3))

    # Filter every key in the dict along the simulation axis
    test_data = {k: v[valid_mask] for k, v in test_data.items()}
    n_simulation = len(valid_mask)
    
    keys_to_drop = set(test_data.keys()) - set(param_names_global) - {sim_data} - set(inference_conditions) - set(['vcirc_kms', 'r_kpc'])
    keys_to_drop = list(keys_to_drop) 
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
    # if cfg.noise_schedule is not None:
    #     inference_network = bf.networks.CompositionalDiffusionModel(
    #                                                     subnet_kwargs={
    #                                                     "widths": [model_config['global_model']['inference_mlp_width']] * model_config['global_model']['inference_mlp_depth'],
    #                                                     "time_embedding_dim": model_config['global_model']['inference_time_embedding_dim'],
    #                                                     },
    #                                                     schedule_kwargs = {**cfg.noise_schedule,},
    #                                                     )
    # else:
    #     #probably needs to fix it to the training noise schedule 
    #     inference_network = bf.networks.CompositionalDiffusionModel(subnet_kwargs={
    #                                                     "widths": [model_config['global_model']['inference_mlp_width']] * model_config['global_model']['inference_mlp_depth'],
    #                                                     "time_embedding_dim": model_config['global_model']['inference_time_embedding_dim'],
    #                                                     },)
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
    # workflow_global.approximator.save_weights(model_path.replace('.keras', '.weights.h5'))
    
    # rng_ = np.random.default_rng(42)
    # val_index = rng_.integers(low=0, high=len(test_data[cfg.sim_data]), size=333, )
    # test_data = {k: test_data[k][val_index] for k in cfg.parameters_global + [cfg.sim_data, "j"] }
    test_data = {k: test_data[k] for k in cfg.parameters_global + [cfg.sim_data, "j"] + ['vcirc_kms']}
    # Augmentation
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
    # Tile vcirc_kms to match the number of streams before augmentations
    n_streams = len(cfg.target_streams.keys())
    test_data['vcirc_kms'] = np.repeat(test_data['vcirc_kms'], n_streams, axis=0)
    print('test data vcirc_kms shape after tiling: ', test_data['vcirc_kms'].shape)

    test_data['j'] = test_data['j'].reshape(-1, 1)
    print('Test data sim shape before augmentation: ', test_data[cfg.sim_data].shape)
    for aug in augmentations:
        print(f"Applying augmentation: {aug.__name__}")
        test_data = aug(test_data)
    test_data[cfg.sim_data] = test_data[cfg.sim_data].reshape(-1, len(cfg.target_streams.keys()), test_data[cfg.sim_data].shape[-2], test_data[cfg.sim_data].shape[-1])
    test_data['vcirc_kms'] = test_data['vcirc_kms'].reshape(-1, len(cfg.target_streams.keys()), test_data['vcirc_kms'].shape[-2], test_data['vcirc_kms'].shape[-1])
    test_data['j'] = test_data['j'].reshape(-1,len(cfg.target_streams.keys()), 1)
    print('Test data sim shape after augmentation: ', test_data[cfg.sim_data].shape)
    print('Test data keys: ', test_data.keys())
    # test_data['attention_mask'] = test_data['attention_mask'].reshape(-1, test_data['attention_mask'].shape[-1])
    print('Test data attention mask shape: ', test_data['attention_mask'].shape)
    with open(os.path.join(cfg.base_dir, cfg.data_dir, '.hydra', 'config.yaml'), "r") as f:
        test_sim_config = yaml.safe_load(f)
    print('Test simulation config prior: ', test_sim_config['priors_global'])

    def prior_global_score(x, time, cfg=cfg, test_sim_config=test_sim_config):
        
        score = {}
        
        for k in cfg.parameters_global:
            # print(f"Computing prior score for {k} with type {test_sim_config['priors_global'][k]['type']}")
            if test_sim_config['priors_global'][k]['type'] == 'uniform':
                score[k] = (1-time)*jnp.zeros_like(x[k])
            elif test_sim_config['priors_global'][k]['type'] == 'normal':
                mean = test_sim_config['priors_global'][k]['prior_parameters'][0]
                std = test_sim_config['priors_global'][k]['prior_parameters'][1]
                score[k] = -(1-time)*(x[k] - mean) / std**2 
        return score

    logging.info("Starting Partial-Pooling (global) inference...")

    # Monkeypatch approach 
    # Reshape all data to have a single batch dimension (n_sims * n_streams)
    # n_sims = test_data[cfg.sim_data].shape[0]
    # n_streams = test_data[cfg.sim_data].shape[1]
    
    # input_a = test_data[cfg.sim_data].reshape(
    #     n_sims,
    #     n_streams, 
    #     test_data[cfg.sim_data].shape[-2], 
    #     test_data[cfg.sim_data].shape[-1]
    # )
    # # Reshape vcirc_kms to match the flattened batch dimension
    # input_b = test_data["vcirc_kms"].reshape(
    #     n_sims,
    #     n_streams,
    #     test_data["vcirc_kms"].shape[-2],
    #     test_data["vcirc_kms"].shape[-1]
    # )
    # #reshape the conditions
    # j_conditions = test_data["j"].reshape(n_sims, n_streams, 1)
    # # Reshape attention mask to be (batch_size, 1, n_stars)
    # attention_mask_reshaped = test_data["attention_mask"].reshape(
    #     n_sims, n_streams, 1, test_data["attention_mask"].shape[-1]
    # )

    # conditions = {
    #     "input_a": input_a,
    #     "input_b": input_b,
    #     "summary_attention_mask": attention_mask_reshaped,
    #     "inference_conditions": j_conditions,
    # }
    # # Monkeypatch to bypass BayesFlow reshape bug with concatenated inference_conditions
    # original_resolve = workflow_global.approximator.condition_builder.resolve
    # def patched_resolve(*args, **kwargs):
    #     resolved_conds, sum_outs = original_resolve(*args, **kwargs)
    #     # Return resolved_conds for summary_outputs as well, ensuring their shapes match exactly (33)
    #     return resolved_conds, resolved_conds
    # workflow_global.approximator.condition_builder.resolve = patched_resolve
    # global_posterior = workflow_global.compositional_sample(
    #                     num_samples=cfg.n_samples,
    #                     conditions=conditions,
    #                     method = cfg.method,
    #                     steps = cfg.steps,
    #                     compositional_bridge_d1= 1/cfg.inverse_compositional_bridge_d1,
    #                     compute_prior_score=prior_global_score,
    #                     batch_size = cfg.batch_size,
    #                     )


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
    
    # ...existing code...
    ps['M_t'] = 4 * np.pi * ps['rho_thin_disk'] * ps['hr_thin_disk']**2 * ps['hz_thin_disk']
    test_data['M_t'] = 4 * np.pi * test_data['rho_thin_disk'] * test_data['hr_thin_disk']**2 * test_data['hz_thin_disk']

    ps['M_k'] = 4 * np.pi * ps['rho_thick_disk'] * ps['hr_thick_disk']**2 * ps['hz_thick_disk']
    test_data['M_k'] = 4 * np.pi * test_data['rho_thick_disk'] * test_data['hr_thick_disk']**2 * test_data['hz_thick_disk']
    cfg.paramater_global_pretty = cfg.paramater_global_pretty + ['$M_t$', '$M_k$']
# ...existing code...
    param_names_global = param_names_global + ['$M_t$', '$M_k$']
    np.savez(os.path.join(cfg.base_dir, cfg.results_dir, 'posterior.npz'), **ps)

    for k in ps.keys():
        print(f"{k}: {ps[k].shape}")
    print(cfg.paramater_global_pretty )
        
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
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'global_recovery.pdf'))
    print('Saved global recovery plot')
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
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, f'global_cornerplot_datasetid_{dataset_id}.pdf'))
    print(f'Saved global corner plot for dataset id {dataset_id}')
    plt.show()
    #calibration plot
    fig = bf.diagnostics.calibration_ecdf(
        estimates=ps,
        targets=test_data,
        difference=True,
        variable_names=cfg.paramater_global_pretty
    )
    for ax in fig.get_axes():
        ax.grid(False)
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'global_calibration.pdf'))
    print('Saved global calibration plot')
    plt.show()
    #calibration plot stacked
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
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'global_calibration_stacked.pdf'))
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
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'global_calibration_stacked_no_diff.pdf'))
    #calibration plot without diff
    fig = bf.diagnostics.calibration_ecdf(
        estimates=ps,
        targets=test_data,
        difference=False,
        variable_names=cfg.paramater_global_pretty
    )
    for ax in fig.get_axes():
        ax.grid(False)
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'global_calibration_no_diff.pdf'))
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
    # fig_1.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'global_histograms_1.pdf'))
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
    # fig_2.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'global_histograms_2.pdf'))
    # plt.show()
    # print('Saved global histograms plot')

    # z_score contraction
    fig = bf.diagnostics.plots.z_score_contraction(
        estimates=ps, 
        targets=test_data,
        variable_names=cfg.paramater_global_pretty
        # variable_names = param_names_global
    )
    for ax in fig.get_axes():
        ax.grid(False)
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'global_z_score_contraction.pdf'))
    print('Saved global z-score contraction plot')
    plt.show()
    print('Finished evaluation with composition')
    print('Results calibration saved in ', os.path.join(cfg.base_dir, cfg.results_dir, 'global_calibration.pdf'))
    

    ###############
    # local model # 
    ###############

if __name__ == "__main__":
    main()