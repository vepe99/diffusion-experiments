from autocvd import autocvd

# autocvd(num_gpus = 1)
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
import jax.numpy as jnp

import logging
logging.getLogger('bayesflow').setLevel(logging.DEBUG)

from config.EvalConfig import EvalConfig
from utils.utils_train_jax_new import AugmentationsClass #we will need to use the augmentations on the test_set


cs = ConfigStore.instance()
cs.store(name="eval_config", node=EvalConfig)


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


def standardize_by_stream(batch, sim_data, stats):
    """
    batch[sim_data]: (N, 1000, 6)
    batch['j']:      (N, 1) or (N,)
    """
    observations = jnp.array(batch[sim_data])          # (N, 1000, 6)
    stream_ids   = jnp.array(batch['j']).astype(int).squeeze()  # (N,)

    # reshape stream_ids for broadcasting: (N, 1, 1)
    j = stream_ids[:, None, None]

    # per-stream stats, shaped (1, 1, 6) for broadcasting
    mean_0, std_0 = stats['mean_stream_0'][None, None, :], stats['std_stream_0'][None, None, :]
    mean_1, std_1 = stats['mean_stream_1'][None, None, :], stats['std_stream_1'][None, None, :]
    mean_2, std_2 = stats['mean_stream_2'][None, None, :], stats['std_stream_2'][None, None, :]

    # select mean and std based on stream id via nested where
    mean = jnp.where(j == 0, mean_0, jnp.where(j == 1, mean_1, mean_2))  # (N, 1, 6)
    std  = jnp.where(j == 0, std_0,  jnp.where(j == 1, std_1,  std_2))   # (N, 1, 6)

    # std and mean broadcast over (N, 1000, 6)
    observations = (observations - mean) / std

    batch[sim_data] = np.array(observations)
    return batch


@hydra.main(version_base=None, config_path="config", config_name="eval_config_local_new",)
def main(cfg: EvalConfig):

    print(cfg)
    print("##############")
    model_path = os.path.join(cfg.base_dir, cfg.model_dir, 'local_model.keras' )
    # Fix ArrayImpl serialization issue in the .keras fil
    model_path = fix_keras_model(model_path, )
    print('Loading model from ', model_path)
    print("##############")
    param_names_local = list(cfg.parameters_local)
    param_names_global = list(cfg.parameters_global)
    sim_data = str(cfg.sim_data)
    inference_conditions = str(cfg.inference_conditions[0])
    test_data_path = os.path.join(cfg.base_dir, cfg.data_dir, f'simulation_multistream_{cfg.multistream_n_simulation}.npz')
    stats = np.load(os.path.join(os.path.dirname(model_path), 'stream_stats.npz'))

    print('Loading test data from ', test_data_path)
    test_data = dict(np.load(test_data_path, allow_pickle=True))
    keys_to_drop = set(test_data.keys()) - set(param_names_local) - set(param_names_global) - {sim_data} - set(inference_conditions)
    keys_to_drop = list(keys_to_drop) 
    inference_conditions = param_names_global + [inference_conditions]

    adapter = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .drop(keys_to_drop)
        .concatenate(param_names_local, into="inference_variables")
        .rename(sim_data, "summary_variables")
        .concatenate(inference_conditions, into="inference_conditions")
    )
    with open(os.path.join(cfg.base_dir, cfg.model_dir, '.hydra', 'config.yaml'), "r") as f:
        model_config = yaml.safe_load(f)
    print(model_config)

    # model_config = {'local_model':
    #                 {   'inference_mlp_width': 4,
    #                     'inference_mlp_depth': cfg.local_model.inference_mlp_depth,
    #                     'inference_time_embedding_dim': cfg.local_model.inference_time_embedding_dim,
    #                     'summary_dim': cfg.local_model.summary_dim,
    #                     'num_heads': cfg.local_model.num_heads,
    #                     'embed_dims': cfg.local_model.embed_dims,
    #                     'mlp_depths': cfg.local_model.mlp_depths,
    #                     'mlp_widths': cfg.local_model.mlp_widths,
    #                     'dropout': cfg.local_model.dropout,
    #                 }
    #             }
    # model_config = cfg.local_model

    if cfg.noise_schedule is not None:
        inference_network = bf.networks.FlowMatching(
                                                    subnet_kwargs={
                                                    "widths": [model_config['local_model']['inference_mlp_width']] * model_config['local_model']['inference_mlp_depth'],
                                                    "time_embedding_dim": model_config['local_model']['inference_time_embedding_dim'],
                                                    },
                                                    schedule_kwargs = {**cfg.noise_schedule,},
                                                    )
    else:
        #probably needs to fix it to the training noise schedule 
        inference_network = bf.networks.FlowMatching(
            subnet_kwargs={
                "widths": [model_config['local_model']['inference_mlp_width']] * model_config['local_model']['inference_mlp_depth'],
                "time_embedding_dim": model_config['local_model']['inference_time_embedding_dim'],
            },
        )
    summary_network = bf.networks.SetTransformer(
        summary_dim=model_config['local_model']['summary_dim'],
        embed_dims=(model_config['local_model']['embed_dims'], model_config['local_model']['embed_dims']),
        num_heads=(model_config['local_model']['num_heads'], model_config['local_model']['num_heads'],),
        mlp_depths=(model_config['local_model']['mlp_depths'], model_config['local_model']['mlp_depths'],),
        mlp_widths=(model_config['local_model']['mlp_widths'], model_config['local_model']['mlp_widths'],),
        dropout=model_config['local_model']['dropout'],
    )
    workflow_local = bf.BasicWorkflow(
        adapter=adapter,
        summary_network=summary_network,
        inference_network=inference_network,
        # standardize=["inference_variables", "inference_conditions"],
        standardize=["inference_variables", "summary_variables", "inference_conditions"],
    )
    workflow_local.approximator = keras.models.load_model(model_path) #this override everything 

    # workflow_local.approximator.inference_network.integrate_kwargs.update({
    # #     'method': cfg.method,
    #     # 'steps': cfg.steps,
    #     "max_steps": cfg.max_steps,
    #     })
    test_data = {k: test_data[k] for k in cfg.parameters_local + cfg.parameters_global + [cfg.sim_data, "j"] }
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
    augmentations.append(lambda batch: standardize_by_stream(batch, sim_data=sim_data, stats=stats))  # re-standardize after flip_dirz

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
        test_data = aug(test_data)
    print('Test data sim shape after augmentation: ', test_data[cfg.sim_data].shape)
    print('Test data attention mask shape: ', test_data['attention_mask'].shape)
    print('###############')


    logging.info("Starting Partial-Pooling (local) inference...")
    conditions = {cfg.sim_data: test_data[cfg.sim_data],}
    conditions['j'] = test_data['j']
    # Add this: repeat global params to match the flattened stream dimension
    n_streams = len(cfg.target_streams)  # = 3
    for param in cfg.parameters_global:
        test_data[param] = np.repeat(test_data[param], n_streams, axis=0)
        conditions[param] = test_data[param] 
    local_posterior = workflow_local.sample(
                        num_samples=1000,
                        conditions=conditions, 
                        batch_size=cfg.batch_size,
                        kwargs={'attention_mask': test_data['attention_mask']}
                    )


    os.makedirs(name=os.path.join(cfg.base_dir, cfg.results_dir), exist_ok=True)

    with open("./config/prior_local.yaml", "r") as f:
        prior_local_dict = yaml.safe_load(f)
    for key in param_names_local:
        print('We are going to renormalize the parameter', key, 'for each stream separately using the prior parameters from prior_local.yaml')
        for name in cfg.target_streams.keys():
            mask_stream = (test_data["j"] == cfg.target_streams[name]).squeeze()  # (300,)
            mean_prior = prior_local_dict[name][key]['prior_parameters'][0]
            std_prior  = prior_local_dict[name][key]['prior_parameters'][1]
            local_posterior[key][mask_stream] = (local_posterior[key][mask_stream] * std_prior + mean_prior)
            print(f"Renormalized {key} for stream {name} using mean={mean_prior} and std={std_prior}")
            print(f"Min and max: {local_posterior[key][mask_stream].min():.4f}, {local_posterior[key][mask_stream].max():.4f}")
    ps = local_posterior.copy()
    # ps shape: (N_TEST, N_STREAMS, N_PARENT_SAMPLES, 1)
    np.savez(os.path.join(cfg.base_dir, cfg.results_dir, 'local_posterior.npz'), **ps)

    

    ###############
    # PLOTS LOCAL #
    ###############

    # Add this: reshape local params from (100, 3, ...) to (300, ...)
    for param in cfg.parameters_local:
        test_data[param] = test_data[param].reshape(-1, *test_data[param].shape[2:])
    
    for stream_name, j_idx in cfg.target_streams.items():
        print(f'\n===== Generating plots for {stream_name} (j={j_idx}) =====')

        obs_mask = (test_data['j'] == j_idx).squeeze()  # (300,) instead of (300, 1)

        test_data_stream = {k: v[obs_mask] for k, v in test_data.items()}
        ps_stream        = {k: v[obs_mask] for k, v in ps.items()}

        print(f'  test_data shapes: { {k: v.shape for k, v in test_data_stream.items()} }')
        print(f'  ps shapes:        { {k: v.shape for k, v in ps_stream.items()} }')

        # --- Recovery ---
        fig = bf.diagnostics.recovery(
            estimates=ps_stream,
            targets=test_data_stream,
            variable_names=cfg.parameter_local_pretty
        )
        for ax in fig.get_axes():
            ax.grid(False)
            for txt in ax.texts:
                txt.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.3'))
        fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, f'{stream_name}_recovery.pdf'))
        print(f'  Saved {stream_name}_recovery.pdf')
        plt.close(fig)

        # --- Calibration ECDF (difference=True) ---
        fig = bf.diagnostics.calibration_ecdf(
            estimates=ps_stream,
            targets=test_data_stream,
            difference=True,
            variable_names=cfg.parameter_local_pretty
        )
        for ax in fig.get_axes():
            ax.grid(False)
        fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, f'{stream_name}_calibration.pdf'))
        print(f'  Saved {stream_name}_calibration.pdf')
        plt.close(fig)

        # --- Calibration ECDF (difference=False) ---
        fig = bf.diagnostics.calibration_ecdf(
            estimates=ps_stream,
            targets=test_data_stream,
            difference=False,
            variable_names=cfg.parameter_local_pretty
        )
        for ax in fig.get_axes():
            ax.grid(False)
        fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, f'{stream_name}_calibration_no_diff.pdf'))
        print(f'  Saved {stream_name}_calibration_no_diff.pdf')
        plt.close(fig)

        # --- Calibration histograms (split into two groups) ---
        ps_keys = list(ps_stream.keys())

        fig_1 = bf.diagnostics.plots.calibration_histogram(
            estimates={k: ps_stream[k]       for k in ps_keys[:4]},
            targets  ={k: test_data_stream[k] for k in ps_keys[:4]},
            variable_names=cfg.parameter_local_pretty[:4]
        )
        for ax in fig_1.get_axes():
            ax.grid(False)
        fig_1.savefig(os.path.join(cfg.base_dir, cfg.results_dir, f'{stream_name}_histograms_1.pdf'))
        plt.close(fig_1)

        if len(ps_keys) > 4:
            fig_2 = bf.diagnostics.plots.calibration_histogram(
                estimates={k: ps_stream[k]       for k in ps_keys[4:]},
                targets  ={k: test_data_stream[k] for k in ps_keys[4:]},
                variable_names=cfg.parameter_local_pretty[4:]
            )
            for ax in fig_2.get_axes():
                ax.grid(False)
            fig_2.savefig(os.path.join(cfg.base_dir, cfg.results_dir, f'{stream_name}_histograms_2.pdf'))
            plt.close(fig_2)

        print(f'  Saved {stream_name}_histograms plots')

        # --- Z-score contraction ---
        fig = bf.diagnostics.plots.z_score_contraction(
            estimates=ps_stream,
            targets=test_data_stream,
            variable_names=cfg.parameter_local_pretty
        )
        for ax in fig.get_axes():
            ax.grid(False)
        fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, f'{stream_name}_z_score_contraction.pdf'))
        print(f'  Saved {stream_name}_z_score_contraction.pdf')
        plt.close(fig)


    ###############
    # local model # 
    ###############

if __name__ == "__main__":
    main()