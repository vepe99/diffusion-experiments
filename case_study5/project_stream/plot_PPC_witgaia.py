# from astropy.io import ascii
# import matplotlib
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde
# import galstreams
# from utils.utils_train_jax  import AugmentationsClass #we will need to use the augmentations on the test_set


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""    


# from config.EvalConfig import EvalConfig
# import hydra
# from hydra.core.config_store import ConfigStore
# cs = ConfigStore.instance()
# cs.store(name="eval_config", node=EvalConfig)



# @hydra.main(version_base=None, config_path="config", config_name="eval_config",)
# def main(cfg: EvalConfig):
#     data = np.load("./data/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz")
#     sim_data = data['sim_data_projected']
#     j = data['j']
#     name_to_plot = ['Pal5', 'NGC3201', 'M68']
#     path_data = "/export/data/vgiusepp/diffusion_experiments_test_new/diffusion-experiments/case_study5/project_stream/data/streams/"
#     sampling_type = 'data_multistream_gala_posterior_predictive_check/model_54_60k_1000epochs_local1_quantile/'
#     name_file = 'ppc_10samples_q16-84.npz'

#     posterior_sample = np.load(f'{path_data}{sampling_type}{name_file}')
#     posteriorpredictive_sample = posterior_sample['sim_data_projected'] 

#      #augumentation function for training 
#     augmentations_class = AugmentationsClass(cfg)
#     augmentations = []
#     if "cut_to_300_particles" in cfg.augmentations:
#         augmentations.append(augmentations_class.cut_to_300_particles)
#     if "remove_los_velocity" in cfg.augmentations:
#         augmentations.append(augmentations_class.remove_los_velocity)
#     if "convert_distance_to_parallax" in cfg.augmentations:
#         augmentations.append(augmentations_class.convert_distance_to_parallax)
#     if "sample_magnitudes" in cfg.augmentations:
#         augmentations.append(augmentations_class.sample_magnitudes)
#     if "sample_obs_error" in cfg.augmentations:
#         augmentations.append(augmentations_class.sample_obs_error)
#     if "apply_obs_error" in cfg.augmentations:  
#         augmentations.append(augmentations_class.apply_obs_error)
#     if "observational_window" in cfg.augmentations:
#         augmentations.append(augmentations_class.observational_window)
#     if "observed_n_stars" in cfg.augmentations:
#         augmentations.append(augmentations_class.subsampling_to_observed_n_stars)
#     if "mask_vlos" in cfg.augmentations:
#         augmentations.append(augmentations_class.mask_vlos)


#     print(f"Loaded sim_data shape: {sim_data.shape}")
#     print(f"Loaded j shape: {j.shape}")
#     attention_mask = data['attention_mask'].astype(bool)
#     posterior_sample_shape = posteriorpredictive_sample.shape
#     print(f"Loaded posterior sample shape: {posterior_sample_shape}")
#     n_posterior = posterior_sample_shape[0]  # 10
#     n_streams   = posterior_sample_shape[1]  # 3

#     # j is (1, 3, 1) — tile to (10, 3, 1) then flatten to (30, 1)
#     j_for_ppc = np.tile(j[0:1], (n_posterior, 1, 1)).reshape(-1, 1)  # (30, 1)

#     posterior_sample = posteriorpredictive_sample.reshape(
#         -1, posterior_sample_shape[2], posterior_sample_shape[3]
#     )  # (30, 1002, 6)

#     dictionary_sample = {
#         'sim_data_projected': posterior_sample,
#         'j': j_for_ppc,
#     }
    
#     for aug in augmentations:
#         dictionary_sample = aug(dictionary_sample)
#     aug_data = dictionary_sample['sim_data_projected']  # (30, 300, 6)
#     posterior_sample = aug_data.reshape(
#         n_posterior, n_streams, aug_data.shape[-2], aug_data.shape[-1]
#     )  # → (10, 3, 300, 6)
#     fig = plt.figure(figsize=(7, 5))
#     for i in range(sim_data.shape[1]):
#         plt.scatter(sim_data[0, i, attention_mask[i, 0, :], 0], sim_data[0, i, attention_mask[i, 0, :], 1], s=1, label=f"{name_to_plot[i]}")
#         plt.scatter(posteriorpredictive_sample[:, i,:,  0], posteriorpredictive_sample[:, i, :, 1], s=1, label=f"{name_to_plot[i]} Posterior Predictive", alpha=0.5)
#     plt.legend()
#     plt.xlabel('$\\alpha$ ', fontsize=20)
#     plt.ylabel('$\\delta$ ', fontsize=20)
#     # plt.rasterized(True)
#     plt.savefig(os.path.join(path_data, sampling_type, 'PPC_10sample.pdf'), dpi=300)


# if __name__ == "__main__":
#     main()

from astropy.io import ascii
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import galstreams
import corner
from utils.utils_train_jax_new_rotationcurve import AugmentationsClass

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from config.EvalConfig import EvalConfig
import hydra
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(name="eval_config", node=EvalConfig)

PHASE_SPACE_LABELS = [
    r'$\alpha$ [deg]',
    r'$\delta$ [deg]',
    r'$\pi$ [mas]',
    r'$\mu_\alpha$ [mas/yr]',
    r'$\mu_\delta$ [mas/yr]',
    r'$v_{\rm los}$ [km/s]',
]

@hydra.main(version_base=None, config_path="config", config_name="eval_config_new_rotationcurve",)
def main(cfg: EvalConfig):
    data = np.load("./data/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz")
    sim_data = data['sim_data_projected']         # (1, 3, 1000, 6)
    j = data['j']                                  # (1, 3, 1)
    attention_mask = data['attention_mask'].astype(bool)  # (3, 1, N_stars)

    name_to_plot = ['Pal5', 'NGC3201', 'M68']
    path_data = "/export/data/vgiusepp/diffusion_experiments_test_new/diffusion-experiments/case_study5/project_stream/data/streams/"
    sampling_type = 'data_multistream_gala_posterior_predictive_check/rotationacurve/model_9_test_local200epochs/'
    name_file = 'ppc_10samples_q16-84.npz'

    posterior_sample = np.load(f'{path_data}{sampling_type}{name_file}')
    posteriorpredictive_sample = posterior_sample['sim_data_projected']  # (10, 3, 1002, 6)

    # Build augmentation pipeline
    augmentations_class = AugmentationsClass(cfg)
    augmentations = []
    for key, fn in [
        ("cut_to_300_particles",       augmentations_class.cut_to_300_particles),
        ("remove_los_velocity",        augmentations_class.remove_los_velocity),
        ("convert_distance_to_parallax", augmentations_class.convert_distance_to_parallax),
        ("observational_window",       augmentations_class.observational_window),
        ("observed_n_stars",           augmentations_class.subsampling_to_observed_n_stars),
        ("compact_to_attended",        augmentations_class.compact_to_attended),
        ("sample_magnitudes",          augmentations_class.sample_magnitudes),
        ("sample_obs_error",           augmentations_class.sample_obs_error),
        ("apply_obs_error",            augmentations_class.apply_obs_error),
        ("mask_vlos",                  augmentations_class.mask_vlos),
    ]:
        if key in cfg.augmentations:
            augmentations.append(fn)

    print(f"Loaded sim_data shape:            {sim_data.shape}")
    print(f"Loaded j shape:                   {j.shape}")
    print(f"Loaded posterior sample shape:    {posteriorpredictive_sample.shape}")

    posterior_sample_shape = posteriorpredictive_sample.shape
    n_posterior = posterior_sample_shape[0]  # 10
    n_streams   = posterior_sample_shape[1]  # 3

    # Tile j: (1, 3, 1) → (10, 3, 1) → (30, 1)
    j_for_ppc = np.tile(j[0:1], (n_posterior, 1, 1)).reshape(-1, 1)

    # Flatten posterior for augmentation: (10, 3, 1002, 6) → (30, 1002, 6)
    posterior_flat = posteriorpredictive_sample.reshape(
        -1, posterior_sample_shape[2], posterior_sample_shape[3]
    )

    dictionary_sample = {
        'sim_data_projected': posterior_flat,
        'j': j_for_ppc,
    }

    for aug in augmentations:
        dictionary_sample = aug(dictionary_sample)

    aug_data = np.array(dictionary_sample['sim_data_projected'])  # (30, N_aug, 6)

    # Apply attention mask from augmentation pipeline if present, else use all particles
    if 'attention_mask' in dictionary_sample:
        aug_mask = np.array(dictionary_sample['attention_mask']).astype(bool)  # (30, N_aug)
    else:
        aug_mask = np.ones(aug_data.shape[:2], dtype=bool)  # (30, N_aug)

    # Reshape back: (30, N_aug, 6) → (10, 3, N_aug, 6)
    n_aug_particles = aug_data.shape[-2]
    posterior_sample_aug = aug_data.reshape(n_posterior, n_streams, n_aug_particles, 6)
    aug_mask_reshaped   = aug_mask.reshape(n_posterior, n_streams, n_aug_particles)  # (10, 3, N_aug)
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, 4))

    # ── Corner plots: one per stream ──────────────────────────────────────────
    for i, stream_name in enumerate(name_to_plot):

        # Observed Gaia data for this stream, filtered by attention mask
        obs = sim_data[0, i, attention_mask[i, 0, :], :]  # (N_obs, 6)                         # remove masked v_los (stored as 0)


        # Posterior predictive for this stream, each sample masked individually
        ppc_list = []
        for s in range(n_posterior):
            mask_s = aug_mask_reshaped[s, i, :]           # (N_aug,)
            ppc_list.append(posterior_sample_aug[s, i, mask_s, :])  # (N_valid, 6)
        ppc_all = np.concatenate(ppc_list, axis=0)        # (N_ppc_total, 6)

        n_dims = 5
        fig, axes = plt.subplots(n_dims, n_dims, figsize=(14, 14))
        fig.suptitle(stream_name, fontsize=25, y=1.01)
       

        for row in range(n_dims):
            for col in range(n_dims):
                ax = axes[row, col]

                if col > row:
                    ax.set_visible(False)
                    continue

                if row == col:
                    # Diagonal: 1-D histograms
                    ax.hist(ppc_all[:, col], bins=40, density=True,
                            color=colors[i+1], alpha=0.5, label='PPC')
                    ax.hist(obs[:, col], bins=40, density=True,
                            color='k', alpha=0.7, histtype='step',
                            linewidth=1.5, label='Gaia')
                    ax.set_yticks([])
                else:
                    # Off-diagonal: 2-D scatter
                    ax.scatter(ppc_all[:, col], ppc_all[:, row],
                               s=0.3, alpha=0.3, color=colors[i+1], rasterized=True)
                    ax.scatter(obs[:, col], obs[:, row],
                               s=1.5, alpha=0.8, color='k', rasterized=True)

                # Axis labels only on the edges
                if row == n_dims - 1:
                    ax.set_xlabel(PHASE_SPACE_LABELS[col], fontsize=15)
                    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
                    ax.tick_params(axis="both", labelsize=15)
                else:
                    ax.set_xticklabels([])
                if col == 0 and row != 0:
                    ax.set_ylabel(PHASE_SPACE_LABELS[row], fontsize=15)
                    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
                    ax.tick_params(axis="both", labelsize=15)
                else:
                    ax.set_yticklabels([])

        # Legend on the top-left diagonal panel
        axes[0, 0].legend(fontsize=20, loc='upper right')

        plt.tight_layout()
        out_path = os.path.join(path_data, sampling_type, f'PPC_corner_{stream_name}.pdf')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved corner plot → {out_path}")


if __name__ == "__main__":
    main()