from autocvd import autocvd
# autocvd(num_gpus = 1)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.core.config_store import ConfigStore
import numpy as np

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

import bayesflow as bf


import logging
logging.getLogger('bayesflow').setLevel(logging.DEBUG)

# from case_study5.project_stream.train_config import TrainConfig
from train_config import TrainConfig
from utils_train import AugmentationsClass
cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)


@hydra.main(version_base=None, config_path="config", config_name="train_config",)
def main(cfg: TrainConfig):
    print(cfg)
    model_path = os.path.join(cfg.base_dir, cfg.results_dir, )
    os.makedirs(model_path, exist_ok=True)
    param_names_global = list(cfg.parameters_global)
    sim_data = str(cfg.sim_data)
    inference_conditions = str(cfg.inference_conditions[0]) #jut 1
    
    adapter = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        #.drop(keys not in param_names_global + [sim_data] + inference_conditions)
        .concatenate(param_names_global, into="inference_variables")
        .rename(sim_data, "summary_variables")
        .rename(inference_conditions, "inference_conditions")
    )
    workflow_global = bf.BasicWorkflow(
        adapter=adapter,
        summary_network=bf.networks.SetTransformer(summary_dim=cfg.global_model.summary_dim, 
                                                   embed_dims=(cfg.global_model.summary_dim, cfg.global_model.summary_dim), 
                                                   num_heads=(cfg.global_model.num_heads, cfg.global_model.num_heads,),
                                                   dropout=cfg.global_model.dropout),
        inference_network=bf.networks.CompositionalDiffusionModel(
                                                        subnet_kwargs={
                                                        "widths": [cfg.global_model.inference_mlp_width] * cfg.global_model.inference_mlp_depth,
                                                        "time_embedding_dim": cfg.global_model.inference_time_embedding_dim,
                                                        }),
        standardize=["inference_variables", "summary_variables"]
    )
    train_data_path = os.path.join(cfg.base_dir, cfg.data_dir, f"training_data_{cfg.N_simulations}.npz")
    print("Train data path:", train_data_path)
    training_data = dict(np.load(train_data_path, allow_pickle=True))
    # training_data = {k: v[:30_000] for k, v in training_data.items()}
    print("Training data keys", training_data.keys())

    augmentations_class = AugmentationsClass(cfg)
    augmentations = []

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
    if "observed_n_stars" in cfg.augmentations:
        augmentations.append(augmentations_class.subsampling_to_observed_n_stars)

    
    if cfg.test:
        import matplotlib.pyplot as plt 
        train_batch = {k: v[:cfg.batch_size] for k, v in training_data.items()}
        for aug in augmentations:
            train_batch = aug(train_batch)
        sim_data = train_batch[cfg.sim_data]
        print("sim_data shape:", sim_data.shape)
        mask = train_batch['attention_mask'][:, 0, :].astype(bool)  # (batch_size, n_particles)
        magnitudes = train_batch['magnitudes']  # (batch_size, n_particles)
        print("magnitudes shape:", magnitudes.shape)
        print('observational error shape:', train_batch['obs_errors'].shape)
        print('sim_data shape post augmentation:', sim_data.shape)
        #plot the masked array
        fig_scatter = plt.figure()
        ax1 = fig_scatter.add_subplot(1, 3, 1)
        ax2 = fig_scatter.add_subplot(1, 3, 2)
        ax3 = fig_scatter.add_subplot(1, 3, 3)
        for i in range(cfg.batch_size):
            if train_batch['j'][i] == 0:
                ax1.scatter(sim_data[i, mask[i], 0],
                            sim_data[i, mask[i], 1], alpha=0.5)
            elif train_batch['j'][i] == 1:
                ax2.scatter(sim_data[i, mask[i], 0],
                            sim_data[i, mask[i], 1], alpha=0.5)
            elif train_batch['j'][i] == 2:
                ax3.scatter(sim_data[i, mask[i], 0],
                            sim_data[i, mask[i], 1], alpha=0.5)
        ax1.set_title(augmentations_class.idx_to_stream[0])
        ax2.set_title(augmentations_class.idx_to_stream[1])
        ax3.set_title(augmentations_class.idx_to_stream[2])
        fig_scatter.savefig(os.path.join(model_path, 'augmentation_test.pdf'))
        plt.show()
        #plot the count of stars
        fig_histogram = plt.figure(figsize=(15, 5))
        ax1 = fig_histogram.add_subplot(1, 3, 1)
        ax2 = fig_histogram.add_subplot(1, 3, 2)
        ax3 = fig_histogram.add_subplot(1, 3, 3)
        particles_length_per_stream = {stream: [] for stream in cfg.target_streams.keys()}
        for i in range(cfg.batch_size):
            if train_batch['j'][i] == 0:
                particles_length_per_stream[augmentations_class.idx_to_stream[0]].append(mask[i].sum())
            elif train_batch['j'][i] == 1:
                particles_length_per_stream[augmentations_class.idx_to_stream[1]].append(mask[i].sum())
            elif train_batch['j'][i] == 2:
                particles_length_per_stream[augmentations_class.idx_to_stream[2]].append(mask[i].sum())
        ax1.hist(particles_length_per_stream[augmentations_class.idx_to_stream[0]], bins=20)
        ax2.hist(particles_length_per_stream[augmentations_class.idx_to_stream[1]], bins=20)
        ax3.hist(particles_length_per_stream[augmentations_class.idx_to_stream[2]], bins=20)
        ax1.set_title(augmentations_class.idx_to_stream[0])
        ax2.set_title(augmentations_class.idx_to_stream[1])
        ax3.set_title(augmentations_class.idx_to_stream[2])
        fig_histogram.savefig(os.path.join(model_path, 'augmentation_histogram.pdf'))
        plt.show()
        #plot the distribution of magnitudes
        fig_magnitudes = plt.figure()
        ax1 = fig_magnitudes.add_subplot(1, 3, 1)
        ax2 = fig_magnitudes.add_subplot(1, 3, 2)
        ax3 = fig_magnitudes.add_subplot(1, 3, 3)
        particles_magnitudes_per_stream = {stream: [] for stream in cfg.target_streams.keys()}
        for i in range(cfg.batch_size):
            if train_batch['j'][i] == 0:
                particles_magnitudes_per_stream[augmentations_class.idx_to_stream[0]].extend(magnitudes[i][mask[i]])
            elif train_batch['j'][i] == 1:
                particles_magnitudes_per_stream[augmentations_class.idx_to_stream[1]].extend(magnitudes[i][mask[i]])
            elif train_batch['j'][i] == 2:
                particles_magnitudes_per_stream[augmentations_class.idx_to_stream[2]].extend(magnitudes[i][mask[i]])
        ax1.hist(particles_magnitudes_per_stream[augmentations_class.idx_to_stream[0]], bins=20)
        ax2.hist(particles_magnitudes_per_stream[augmentations_class.idx_to_stream[1]], bins=20)
        ax3.hist(particles_magnitudes_per_stream[augmentations_class.idx_to_stream[2]], bins=20)
        ax1.set_title(augmentations_class.idx_to_stream[0])
        ax2.set_title(augmentations_class.idx_to_stream[1])
        ax3.set_title(augmentations_class.idx_to_stream[2])
        fig_magnitudes.savefig(os.path.join(model_path, 'augmentation_magnitudes.pdf'))
        plt.show()

    history = workflow_global.fit_offline(
        training_data,
        epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        verbose=cfg.verbose,
        augmentations=augmentations,
    )
    workflow_global.approximator.save(os.path.join(model_path, 'global_model.keras'))
    loss_plot = bf.diagnostics.plots.loss(history,);
    loss_plot.savefig(os.path.join(model_path, 'loss_plot.pdf'))


if __name__ == "__main__":
    main()