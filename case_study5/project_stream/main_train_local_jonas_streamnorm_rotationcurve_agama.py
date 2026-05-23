from autocvd import autocvd

autocvd(num_gpus=1, interval=1)
import os
import yaml

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.core.config_store import ConfigStore
import numpy as np

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

import bayesflow as bf
import keras
import jax.numpy as jnp

import logging

logging.getLogger("bayesflow").setLevel(logging.DEBUG)

# from case_study5.project_stream.train_config import TrainConfig
from config.TrainConfig import TrainConfig
from utils.utils_train_jax_new_rotationcurve_fixedvlosmask import AugmentationsClass
from utils.custom_summary_network import SetTransformer, FusionNetwork

cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)

def compute_and_save_stream_stats(training_data, sim_data, model_path):
    """
    training_data[sim_data]: shape (N, 1000, 6)
    training_data['j']:      shape (N,) with stream ids in {0, 1, 2}
    
    mean/std per stream: shape (6,)
    """
    observations = training_data[sim_data]        # (N, 1000, 6)
    # stream_ids   = training_data['j'].astype(int) # (N,)
    stream_ids   = training_data['j'].astype(int).squeeze()   # (N,)


    stats = {}
    for j in np.unique(stream_ids):
        mask   = stream_ids == j
        obs_j  = observations[mask]               # (N_j, 1000, 6)
        mean_j = obs_j.mean(axis=(0, 1))          # (6,)
        std_j  = obs_j.std(axis=(0, 1))           # (6,)
        std_j  = np.where(std_j == 0, 1.0, std_j)
        stats[f'mean_stream_{j}'] = mean_j
        stats[f'std_stream_{j}']  = std_j
        print(f"Stream {j}: mean shape={mean_j.shape}, std shape={std_j.shape}")
        print(f"  mean={mean_j}")
        print(f"  std={std_j}")
    
    # ---- vcirc_kms: mean/std per radial bin ----
    x = np.log10(training_data["vcirc_kms"])                  # (N, 34, 1)
    stats["vcirc_kms"] = {
        "mean_log10vcirc_kms": x.mean(axis=0),                     # (34, 1)
        "std_log10vcirc_kms":  x.std(axis=0).clip(min=1e-8),       # (34, 1)
    }

    save_path = os.path.join(model_path, 'stream_stats.npz')
    np.savez(save_path, **stats)
    print(f"Stream stats saved to {save_path}")
    return stats


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

    #apply normalization to vcirc_kms, the vcirc_kms should already be in log10 
    x = jnp.array(batch["vcirc_kms"])  # (N, 34, 1)
    mean_vcirc = jnp.array(stats["vcirc_kms"]["mean_log10vcirc_kms"])  # (34, 1)
    std_vcirc  = jnp.array(stats["vcirc_kms"]["std_log10vcirc_kms"])   # (34, 1)
    x = (x - mean_vcirc) / std_vcirc
    batch["vcirc_kms"] = np.array(x)

    return batch

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="train_config_local_new_rotationcurve_agama",
)
def main(cfg: TrainConfig):
    print(cfg)
    model_path = os.path.join(
        cfg.base_dir,
        cfg.results_dir,
    )
    os.makedirs(model_path, exist_ok=True)
    param_names_local = list(cfg.parameters_local)
    param_names_global = list(cfg.parameters_global)
    sim_data = str(cfg.sim_data)
    inference_conditions = str(cfg.inference_conditions[0])  # jut 1
    train_data_path = os.path.join(
        cfg.base_dir, cfg.data_dir, f"training_data_local_{cfg.N_simulations}.npz"
    )
    print("Train data path:", train_data_path)
    training_data = dict(np.load(train_data_path, allow_pickle=True))
    training_data_rotation_curve = dict(np.load('./data/plots/agama_rotcurv/rotation_curves.npz'))
    
    training_data['vcirc_kms'] = training_data_rotation_curve['vcirc_kms'][:, :, None] #extra dimension (n_observation, len_r_kpc, 1)
    augmentations_class = AugmentationsClass(cfg)
    mask_r_kpc = (augmentations_class.obs_R >5.5)
    training_data['vcirc_kms'] = training_data_rotation_curve['vcirc_kms'][:, mask_r_kpc, None] #extra dimension (n_observation, len_r_kpc, 1)
    # validation_data = {k: v[60_000:80_000] for k, v in training_data.items()}
    # print("Validation data keys", validation_data.keys())
    # training_data = {k: v[:290_000] for k, v in training_data.items()}
    #nan cleaning
    sim_data_array = training_data['sim_data_projected']  # shape: (N, ...)
    # Build a boolean mask: True where the simulation is NaN-free
    valid_mask = ~np.any(np.isnan(sim_data_array.reshape(sim_data_array.shape[0], -1)), axis=1)

    n_removed = (~valid_mask).sum()
    print(f"Removing {n_removed}/{len(valid_mask)} simulations containing NaN values.")

    # Apply the mask to all arrays
    clean_data = {key: training_data[key][valid_mask] for key in param_names_global + param_names_local + [inference_conditions] + ["vcirc_kms"]}
    clean_data['sim_data_projected'] = sim_data_array[valid_mask]
    training_data = clean_data
    stats = compute_and_save_stream_stats(training_data, sim_data, model_path)

    print("Training data keys", training_data.keys())
    
    with open("./config/prior_local.yaml", "r") as f:
        prior_local_dict = yaml.safe_load(f)

    for key in param_names_local:
        print('We are going to renormalize the parameter', key, 'for each stream separately using the prior parameters from prior_local.yaml')
        for name in cfg.target_streams.keys():
            mask_stream = training_data["j"] == cfg.target_streams[name]
            mean_prior = prior_local_dict[name][key]['prior_parameters'][0]
            std_prior = prior_local_dict[name][key]['prior_parameters'][1]
            training_data[key][mask_stream] = (training_data[key][mask_stream] - mean_prior) / std_prior
            print(f"Renormalized {key} for stream {name} using mean={mean_prior} and std={std_prior}")
            print('Min and max of the renormalized parameter for this stream:', training_data[key][mask_stream].min(), training_data[key][mask_stream].max())

    keys_to_drop = (
        set(training_data.keys())
        - set(param_names_local)
        - set(param_names_global)
        - {sim_data}
        - {"vcirc_kms"}
        - set(inference_conditions)
    )
    keys_to_drop = list(keys_to_drop)
    inference_conditions = param_names_global + [inference_conditions]

    adapter = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .drop(keys_to_drop)
        .concatenate(param_names_local, into="inference_variables")
        .concatenate(inference_conditions, into="inference_conditions")
        .rename(sim_data, "input_a")
        .rename('attention_mask', 'summary_attention_mask')
        .rename("vcirc_kms", "input_b")
        .group(
        ["input_a", "input_b",], into="summary_variables")   
    )
    summary_network_a = SetTransformer(
            summary_dim = 32,
            embed_dims = (64, 64),
            num_heads = (4, 4),
            num_seeds = 6,
            dropout=cfg.global_model.dropout,
        )
    summary_network_b = bf.networks.TimeSeriesTransformer(
        summary_dim = 32,
        embed_dims = (64, 64,),
        num_heads = (4, 4,),

    )
    head = keras.Sequential(
        [bf.networks.MLP(widths=[64, 64]), keras.layers.Dense(units=32)]
    )

    summary_network = FusionNetwork(
        backbones={"input_a": summary_network_a, "input_b": summary_network_b},
        head=head,
    )

    workflow_local = bf.CompositionalWorkflow(
        adapter=adapter,
        summary_network=summary_network,
        inference_network=bf.networks.DiffusionModel(),
        standardize=["inference_variables", "inference_conditions"],
        checkpoint_filepath=model_path,
        checkpoint_name="checkpoint_local_model.keras",
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
    augmentations.append(lambda batch: standardize_by_stream(batch, sim_data=sim_data, stats=stats))  # re-standardize after flip_dirz

    if "concatentate_sigma_error_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatentate_sigma_error_to_sim_data)
    if "concatenate_magnitudes_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_magnitudes_to_sim_data)
    if "concatenate_vlos_mask_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_vlos_mask_to_sim_data)
    if "concatenate_j_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_j_to_sim_data)


    history = workflow_local.fit_offline(
        training_data,
        epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        verbose=cfg.verbose,
        augmentations=augmentations,
    )
    workflow_local.approximator.save(os.path.join(model_path, "local_model.keras"))
    # workflow_local.approximator.save_weights(model_path.replace('.keras', '.weights.h5'))
    loss_plot = bf.diagnostics.plots.loss(
        history,
    )
    loss_plot.savefig(os.path.join(model_path, "loss_plot.pdf"))


if __name__ == "__main__":
    main()
