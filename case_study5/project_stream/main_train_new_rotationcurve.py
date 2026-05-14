from autocvd import autocvd
autocvd(num_gpus=1, interval=1)
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
from astropy import units as u
from astropy.io import ascii

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

import bayesflow as bf
import keras

import matplotlib.pyplot as plt

import logging

logging.getLogger("bayesflow").setLevel(logging.DEBUG)

# from case_study5.project_stream.train_config import TrainConfig
from config.TrainConfig import TrainConfig
from utils.utils_train_jax_new_rotationcurve import (AugmentationsClass,
                                                     compute_standardization, 
                                                     apply_standardization,
                                                     save_stats)
from utils.custom_summary_network import SetTransformer, FusionNetwork





cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="train_config_new_rotationcurve",
)
def main(cfg: TrainConfig):
    print(cfg)
    model_path = os.path.join(
        cfg.base_dir,
        cfg.results_dir,
    )
    os.makedirs(model_path, exist_ok=True)
    param_names_global = list(cfg.parameters_global)
    sim_data = str(cfg.sim_data)
    inference_conditions = str(cfg.inference_conditions[0])  # jut 1
    train_data_path = os.path.join(
        cfg.base_dir, cfg.data_dir, f"training_data_local_{cfg.N_simulations}.npz"
    )
    print("Train data path:", train_data_path)
    training_data = dict(np.load(train_data_path, allow_pickle=True))
    training_data_rotation_curve = dict(np.load('./data/plots/gala_rotcurv/rotation_curves.npz'))
    
    training_data['vcirc_kms'] = training_data_rotation_curve['vcirc_kms'][:, :, None] #extra dimension (n_observation, len_r_kpc, 1)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    # for i in range(training_data['vcirc_kms'].shape[0]):
        # ax.plot(training_data_rotation_curve['r_kpc'], training_data['vcirc_kms'][i,:,0], color='gray', alpha=0.1)
    ax.hist(np.log10(training_data['vcirc_kms'].flatten()), bins=30, alpha=0.1)
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel('Circular Velocity (km/s)')
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, "rotation_curves_hist.png"), dpi=300)

    # exit()
    # training_data['r_kpc'] = np.tile(training_data_rotation_curve['r_kpc'], reps=(training_data['vcirc_kms'].shape[0],1))[:, :, None] #[:, :, None] #extra dimension (n_observation, len_r_kpc, 1)

    training_data = {k: v[:20_000] for k, v in training_data.items()}
    for k in training_data.keys():
        print(f"{k}: {training_data[k].shape}")

    #nan cleaning
    sim_data_array = training_data['sim_data_projected']  # shape: (N, ...)
    # Build a boolean mask: True where the simulation is NaN-free
    valid_mask = ~np.any(np.isnan(sim_data_array.reshape(sim_data_array.shape[0], -1)), axis=1)

    n_removed = (~valid_mask).sum()
    print(f"Removing {n_removed}/{len(valid_mask)} simulations containing NaN values.")

    # Apply the mask to all arrays
    clean_data = {key: training_data[key][valid_mask] for key in param_names_global + [inference_conditions] + ["vcirc_kms"]}
    clean_data['sim_data_projected'] = sim_data_array[valid_mask]
    training_data = clean_data

    print("Training data keys", training_data.keys())
    keys_to_drop = (
        set(training_data.keys())
        - set(param_names_global)
        - {sim_data}
        - {"vcirc_kms"}
        - set(inference_conditions)
    )
    keys_to_drop = list(keys_to_drop)

    adapter = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .drop(keys_to_drop)
        .rename(inference_conditions, "inference_conditions")
        .concatenate(param_names_global, into="inference_variables")
        .rename(sim_data, "input_a")
        .rename('attention_mask', 'summary_attention_mask')
        # .concatenate(['vcirc_kms', 'r_kpc'], into="input_b")
        .rename("vcirc_kms", "input_b")
        .group(
        # ["input_a", "input_b", "attention_mask"], into="summary_variables")  
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
        standardize=["inference_variables", "summary_variables"],
        checkpoint_filepath=model_path,
        checkpoint_name="checkpoint_global_model.keras",
    )

    augmentations_class = AugmentationsClass(cfg)
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
    
    history = workflow_global.fit_offline(
        training_data,
        epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        verbose=cfg.verbose,
        augmentations=augmentations,
    )
    workflow_global.approximator.save(os.path.join(model_path, "global_model.keras"))
    # workflow_global.approximator.save_weights(model_path.replace('.keras', '.weights.h5'))
    loss_plot = bf.diagnostics.plots.loss(
        history,
    )
    loss_plot.savefig(os.path.join(model_path, "loss_plot.pdf"))


if __name__ == "__main__":
    main()
