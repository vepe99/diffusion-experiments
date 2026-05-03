from autocvd import autocvd
autocvd(num_gpus=1, interval=1)
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.core.config_store import ConfigStore
import numpy as np

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

import bayesflow as bf


import logging

logging.getLogger("bayesflow").setLevel(logging.DEBUG)

# from case_study5.project_stream.train_config import TrainConfig
from config.TrainConfig import TrainConfig
from utils.utils_train_jax_new import AugmentationsClass

cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="train_config_new",
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
    
    
    training_data = {k: v[:60_000] for k, v in training_data.items()}

    #nan cleaning
    sim_data_array = training_data['sim_data_projected']  # shape: (N, ...)
    # Build a boolean mask: True where the simulation is NaN-free
    valid_mask = ~np.any(np.isnan(sim_data_array.reshape(sim_data_array.shape[0], -1)), axis=1)

    n_removed = (~valid_mask).sum()
    print(f"Removing {n_removed}/{len(valid_mask)} simulations containing NaN values.")

    # Apply the mask to all arrays
    clean_data = {key: training_data[key][valid_mask] for key in param_names_global + [inference_conditions]}
    clean_data['sim_data_projected'] = sim_data_array[valid_mask]
    training_data = clean_data

    print("Training data keys", training_data.keys())
    keys_to_drop = (
        set(training_data.keys())
        - set(param_names_global)
        - {sim_data}
        - set(inference_conditions)
    )
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
    workflow_global = bf.BasicWorkflow(
        adapter=adapter,
        summary_network=bf.networks.SetTransformer(
            summary_dim=cfg.global_model.summary_dim,
            embed_dims=(cfg.global_model.embed_dims, cfg.global_model.embed_dims),
            num_heads=(
                cfg.global_model.num_heads,
                cfg.global_model.num_heads,
            ),
            mlp_depths=(cfg.global_model.mlp_depths, cfg.global_model.mlp_depths),
            mlp_widths=(cfg.global_model.mlp_widths, cfg.global_model.mlp_widths),
            dropout=cfg.global_model.dropout,
        ),
        inference_network=bf.networks.CompositionalDiffusionModel(
            subnet_kwargs={
                "widths": [cfg.global_model.inference_mlp_width]
                * cfg.global_model.inference_mlp_depth,
                "time_embedding_dim": cfg.global_model.inference_time_embedding_dim,
            }
        ),
        standardize=["inference_variables", "summary_variables"],
        checkpoint_filepath=model_path,
        checkpoint_name="checkpoint_global_model.keras",
    )

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

    # --- Feature concatenations (must be last) ---
    if "concatentate_sigma_error_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatentate_sigma_error_to_sim_data)
    if "concatenate_magnitudes_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_magnitudes_to_sim_data)
    if "concatenate_vlos_mask_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_vlos_mask_to_sim_data)
    if "concatenate_j_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_j_to_sim_data)


    if cfg.test:
        import matplotlib.pyplot as plt

        train_batch = {k: v[: cfg.batch_size] for k, v in training_data.items()}
        for aug in augmentations:
            train_batch = aug(train_batch)
        sim_data = train_batch[cfg.sim_data]
        print("sim_data shape:", sim_data.shape)
        mask = train_batch["attention_mask"][:, 0, :].astype(
            bool
        )  # (batch_size, n_particles)
        magnitudes = train_batch["magnitudes"]  # (batch_size, n_particles)
        print("magnitudes shape:", magnitudes.shape)
        print("observational error shape:", train_batch["obs_errors"].shape)
        print("sim_data shape post augmentation:", sim_data.shape)
        # plot the masked array
        fig_scatter = plt.figure()
        ax1 = fig_scatter.add_subplot(1, 3, 1)
        ax2 = fig_scatter.add_subplot(1, 3, 2)
        ax3 = fig_scatter.add_subplot(1, 3, 3)
        for i in range(cfg.batch_size):
            if train_batch["j"][i] == 0:
                ax1.scatter(sim_data[i, mask[i], 0], sim_data[i, mask[i], 1], alpha=0.5)
            elif train_batch["j"][i] == 1:
                ax2.scatter(sim_data[i, mask[i], 0], sim_data[i, mask[i], 1], alpha=0.5)
            elif train_batch["j"][i] == 2:
                ax3.scatter(sim_data[i, mask[i], 0], sim_data[i, mask[i], 1], alpha=0.5)
        ax1.set_title(augmentations_class.idx_to_stream[0])
        ax2.set_title(augmentations_class.idx_to_stream[1])
        ax3.set_title(augmentations_class.idx_to_stream[2])
        fig_scatter.savefig(os.path.join(model_path, "augmentation_test.pdf"))
        plt.show()
        # plot the count of stars
        fig_histogram = plt.figure(figsize=(15, 5))
        ax1 = fig_histogram.add_subplot(1, 3, 1)
        ax2 = fig_histogram.add_subplot(1, 3, 2)
        ax3 = fig_histogram.add_subplot(1, 3, 3)
        particles_length_per_stream = {
            stream: [] for stream in cfg.target_streams.keys()
        }
        for i in range(cfg.batch_size):
            if train_batch["j"][i] == 0:
                particles_length_per_stream[
                    augmentations_class.idx_to_stream[0]
                ].append(mask[i].sum())
            elif train_batch["j"][i] == 1:
                particles_length_per_stream[
                    augmentations_class.idx_to_stream[1]
                ].append(mask[i].sum())
            elif train_batch["j"][i] == 2:
                particles_length_per_stream[
                    augmentations_class.idx_to_stream[2]
                ].append(mask[i].sum())
        ax1.hist(
            particles_length_per_stream[augmentations_class.idx_to_stream[0]], bins=20
        )
        ax2.hist(
            particles_length_per_stream[augmentations_class.idx_to_stream[1]], bins=20
        )
        ax3.hist(
            particles_length_per_stream[augmentations_class.idx_to_stream[2]], bins=20
        )
        ax1.set_title(augmentations_class.idx_to_stream[0])
        ax2.set_title(augmentations_class.idx_to_stream[1])
        ax3.set_title(augmentations_class.idx_to_stream[2])
        fig_histogram.savefig(os.path.join(model_path, "augmentation_histogram.pdf"))
        plt.show()
        # plot the distribution of magnitudes
        fig_magnitudes = plt.figure()
        ax1 = fig_magnitudes.add_subplot(1, 3, 1)
        ax2 = fig_magnitudes.add_subplot(1, 3, 2)
        ax3 = fig_magnitudes.add_subplot(1, 3, 3)
        particles_magnitudes_per_stream = {
            stream: [] for stream in cfg.target_streams.keys()
        }
        for i in range(cfg.batch_size):
            if train_batch["j"][i] == 0:
                particles_magnitudes_per_stream[
                    augmentations_class.idx_to_stream[0]
                ].extend(magnitudes[i][mask[i]])
            elif train_batch["j"][i] == 1:
                particles_magnitudes_per_stream[
                    augmentations_class.idx_to_stream[1]
                ].extend(magnitudes[i][mask[i]])
            elif train_batch["j"][i] == 2:
                particles_magnitudes_per_stream[
                    augmentations_class.idx_to_stream[2]
                ].extend(magnitudes[i][mask[i]])
        ax1.hist(
            particles_magnitudes_per_stream[augmentations_class.idx_to_stream[0]],
            bins=20,
        )
        ax2.hist(
            particles_magnitudes_per_stream[augmentations_class.idx_to_stream[1]],
            bins=20,
        )
        ax3.hist(
            particles_magnitudes_per_stream[augmentations_class.idx_to_stream[2]],
            bins=20,
        )
        ax1.set_title(augmentations_class.idx_to_stream[0])
        ax2.set_title(augmentations_class.idx_to_stream[1])
        ax3.set_title(augmentations_class.idx_to_stream[2])
        fig_magnitudes.savefig(os.path.join(model_path, "augmentation_magnitudes.pdf"))
        plt.show()

        train_batch = {k: v[: cfg.batch_size] for k, v in training_data.items()}

        # ---- Capture sim_data BEFORE mask_vlos for comparison ----
        # Run augmentations up to (but not including) mask_vlos
        sim_data_before_mask_vlos = None
        sigma_errors_before_mask_vlos = None
        for aug in augmentations:
            if aug == augmentations_class.mask_vlos:
                # Snapshot before mask_vlos
                sim_data_before_mask_vlos = np.array(train_batch[cfg.sim_data][:, :, -1])
                sigma_errors_before_mask_vlos = np.array(train_batch["sigma_errors"][:, :, -1])
            train_batch = aug(train_batch)

        sim_data = train_batch[cfg.sim_data]
        print("sim_data shape:", sim_data.shape)
        mask = train_batch["attention_mask"][:, 0, :].astype(
            bool
        )  # (batch_size, n_particles)
        magnitudes = train_batch["magnitudes"]  # (batch_size, n_particles)
        print("magnitudes shape:", magnitudes.shape)
        print("observational error shape:", train_batch["obs_errors"].shape)
        print("sim_data shape post augmentation:", sim_data.shape)

        # ---- Visualize v_los before/after mask_vlos ----
        if sim_data_before_mask_vlos is not None and "vlos_mask" in train_batch:
            vlos_mask = np.array(train_batch["vlos_mask"][:, 0, :]).astype(bool)  # (batch_size, n_particles)
            # After masking, v_los is in the 6th column (index 5) of the original sim_data
            # but sim_data may have been concatenated with sigma, magnitudes, etc.
            # So we grab it from the stored snapshot and the post-mask state
            sim_data_np = np.array(train_batch[cfg.sim_data])

            # The v_los after mask_vlos is at column index 5 (before concatenations added more columns)
            # We need to figure out the correct index. Since mask_vlos runs before concatenations,
            # the 6th column of sim_data at that point was index 5.
            # After concatenations, sim_data has more columns, but we saved the snapshot.
            # Let's just use the stored before/after at the last dim before concatenation.

            # We'll re-run to get the exact post-mask v_los
            # Simpler: re-run augmentations but stop right after mask_vlos
            train_batch_vlos = {k: v[: cfg.batch_size] for k, v in training_data.items()}
            sim_data_after_mask_vlos = None
            for aug in augmentations:
                train_batch_vlos = aug(train_batch_vlos)
                if aug == augmentations_class.mask_vlos:
                    sim_data_after_mask_vlos = np.array(train_batch_vlos[cfg.sim_data][:, :, -1])
                    vlos_mask_plot = np.array(train_batch_vlos["vlos_mask"][:, 0, :]).astype(bool)
                    mask_plot = np.array(train_batch_vlos["attention_mask"][:, 0, :]).astype(bool)
                    j_plot = np.array(train_batch_vlos["j"])
                    break

            if sim_data_after_mask_vlos is not None:
                fig_vlos, axes = plt.subplots(2, 3, figsize=(18, 10))
                stream_names = [augmentations_class.idx_to_stream[i] for i in range(augmentations_class.n_streams)]

                for col, (stream_idx, stream_name) in enumerate(enumerate(stream_names)):
                    ax_before = axes[0, col]
                    ax_after = axes[1, col]

                    # Collect v_los values for this stream
                    vlos_before_kept = []
                    vlos_before_masked = []
                    vlos_after_kept = []
                    vlos_after_masked = []

                    for i in range(cfg.batch_size):
                        if j_plot[i, 0] == stream_idx:
                            attended = mask_plot[i]
                            kept = vlos_mask_plot[i] & attended
                            not_kept = ~vlos_mask_plot[i] & attended

                            vlos_before_kept.extend(sim_data_before_mask_vlos[i, kept].tolist())
                            vlos_before_masked.extend(sim_data_before_mask_vlos[i, not_kept].tolist())
                            vlos_after_kept.extend(sim_data_after_mask_vlos[i, kept].tolist())
                            vlos_after_masked.extend(sim_data_after_mask_vlos[i, not_kept].tolist())

                    # Before mask_vlos
                    if vlos_before_kept:
                        ax_before.hist(vlos_before_kept, bins=50, alpha=0.6, label=f"kept (n={len(vlos_before_kept)})", color="tab:blue", density=True)
                    if vlos_before_masked:
                        ax_before.hist(vlos_before_masked, bins=50, alpha=0.6, label=f"to-be-masked (n={len(vlos_before_masked)})", color="tab:orange", density=True)
                    ax_before.set_title(f"{stream_name} — BEFORE mask_vlos")
                    ax_before.set_xlabel("$v_{los}$")
                    ax_before.legend(fontsize=8)

                    # After mask_vlos
                    if vlos_after_kept:
                        ax_after.hist(vlos_after_kept, bins=50, alpha=0.6, label=f"kept (n={len(vlos_after_kept)})", color="tab:blue", density=True)
                    if vlos_after_masked:
                        ax_after.hist(vlos_after_masked, bins=50, alpha=0.6, label=f"replaced (n={len(vlos_after_masked)})", color="tab:red")
                    ax_after.set_title(f"{stream_name} — AFTER mask_vlos")
                    ax_after.set_xlabel("$v_{los}$")
                    ax_after.legend(fontsize=8)

                fig_vlos.suptitle("v_los distribution before/after mask_vlos augmentation", fontsize=14)
                fig_vlos.tight_layout()
                fig_vlos.savefig(os.path.join(model_path, "augmentation_vlos_mask.pdf"))
                plt.show()

                # ---- Visualize sigma_errors for v_los before/after ----
                train_batch_sigma = {k: v[: cfg.batch_size] for k, v in training_data.items()}
                sigma_before = None
                sigma_after = None
                for aug in augmentations:
                    if aug == augmentations_class.mask_vlos:
                        sigma_before = np.array(train_batch_sigma["sigma_errors"][:, :, -1])
                    train_batch_sigma = aug(train_batch_sigma)
                    if aug == augmentations_class.mask_vlos:
                        sigma_after = np.array(train_batch_sigma["sigma_errors"][:, :, -1])
                        vlos_mask_sigma = np.array(train_batch_sigma["vlos_mask"][:, 0, :]).astype(bool)
                        mask_sigma = np.array(train_batch_sigma["attention_mask"][:, 0, :]).astype(bool)
                        j_sigma = np.array(train_batch_sigma["j"])
                        break

                if sigma_before is not None and sigma_after is not None:
                    fig_sigma, axes_s = plt.subplots(2, 3, figsize=(18, 10))
                    for col, (stream_idx, stream_name) in enumerate(enumerate(stream_names)):
                        ax_sb = axes_s[0, col]
                        ax_sa = axes_s[1, col]

                        sig_before_kept = []
                        sig_before_masked = []
                        sig_after_kept = []
                        sig_after_masked = []

                        for i in range(cfg.batch_size):
                            if j_sigma[i, 0] == stream_idx:
                                attended = mask_sigma[i]
                                kept = vlos_mask_sigma[i] & attended
                                not_kept = ~vlos_mask_sigma[i] & attended

                                sig_before_kept.extend(sigma_before[i, kept].tolist())
                                sig_before_masked.extend(sigma_before[i, not_kept].tolist())
                                sig_after_kept.extend(sigma_after[i, kept].tolist())
                                sig_after_masked.extend(sigma_after[i, not_kept].tolist())

                        if sig_before_kept:
                            ax_sb.hist(sig_before_kept, bins=50, alpha=0.6, label=f"kept (n={len(sig_before_kept)})", color="tab:blue", density=True)
                        if sig_before_masked:
                            ax_sb.hist(sig_before_masked, bins=50, alpha=0.6, label=f"to-be-masked (n={len(sig_before_masked)})", color="tab:orange", density=True)
                        ax_sb.set_title(f"{stream_name} — σ BEFORE mask_vlos")
                        ax_sb.set_xlabel("$\\sigma_{v_{los}}$")
                        ax_sb.legend(fontsize=8)

                        if sig_after_kept:
                            ax_sa.hist(sig_after_kept, bins=50, alpha=0.6, label=f"kept (n={len(sig_after_kept)})", color="tab:blue", density=True)
                        if sig_after_masked:
                            ax_sa.hist(sig_after_masked, bins=50, alpha=0.6, label=f"replaced (n={len(sig_after_masked)})", color="tab:red", density=True)
                        ax_sa.set_title(f"{stream_name} — σ AFTER mask_vlos")
                        ax_sa.set_xlabel("$\\sigma_{v_{los}}$")
                        ax_sa.legend(fontsize=8)

                    fig_sigma.suptitle("σ_vlos distribution before/after mask_vlos augmentation", fontsize=14)
                    fig_sigma.tight_layout()
                    fig_sigma.savefig(os.path.join(model_path, "augmentation_sigma_vlos_mask.pdf"))
                    plt.show()

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
