from autocvd import autocvd
autocvd(num_gpus = 1)
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
import keras
import bayesflow as bf
from scipy import  special 


import logging
logging.getLogger('bayesflow').setLevel(logging.DEBUG)

from config.EvalConfig import EvalConfig
from utils.utils_train_jax import AugmentationsClass #we will need to use the augmentations on the test_set


cs = ConfigStore.instance()
cs.store(name="eval_config", node=EvalConfig)

@hydra.main(version_base=None, config_path="config", config_name="eval_config",)
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
    test_data = dict(np.load(test_data_path, allow_pickle=True))
    keys_to_drop = set(test_data.keys()) - set(param_names_global) - {sim_data} - set(inference_conditions)
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
    with open(os.path.join(cfg.base_dir, cfg.model_dir, '.hydra', 'config.yaml'), "r") as f:
        model_config = yaml.safe_load(f)
    print(model_config)
    workflow_global = bf.BasicWorkflow(
        adapter=adapter,
        summary_network=bf.networks.SetTransformer(summary_dim=model_config['global_model']['summary_dim'], 
                                                   num_heads=(model_config['global_model']['num_heads'],model_config['global_model']['num_heads'],),
                                                   embed_dims = (model_config['global_model']['embed_dims'], model_config['global_model']['embed_dims'],),
                                                   mlp_depths=(model_config['global_model']['mlp_depths'], model_config['global_model']['mlp_depths']),
                                                   mlp_widths=(model_config['global_model']['mlp_widths'], model_config['global_model']['mlp_widths']),
                                                   dropout=0.1),
        inference_network=bf.networks.CompositionalDiffusionModel(subnet_kwargs={
                                                        "widths": [model_config['global_model']['inference_mlp_width']] * model_config['global_model']['inference_mlp_depth'],
                                                        "time_embedding_dim": model_config['global_model']['inference_time_embedding_dim'],
                                                        }),
        standardize=["inference_variables", "summary_variables"]
    )
    workflow_global.approximator = keras.models.load_model(model_path)
    test_data = {k: test_data[k] for k in cfg.parameters_global + [cfg.sim_data, "j"] }
    # Augmentation
    augmentations_class = AugmentationsClass(cfg)
    augmentations = []
    if "cut_to_300_particles" in cfg.augmentations:
        augmentations.append(augmentations_class.cut_to_300_particles)
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
    if "mask_vlos" in cfg.augmentations:
        augmentations.append(augmentations_class.mask_vlos)
    if "flip_dirz" in cfg.augmentations:
        augmentations.append(augmentations_class.flip_dirz)
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
    for k in cfg.parameters_global:
        test_data[k] = np.repeat(test_data[k], 3, axis=0).reshape(-1, 1)
    for k in test_data.keys():
        print('##########')
        print(f"{k} shape: {test_data[k].shape}")

    observed_data_path = '/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz'
    print('Loading observed data from ', observed_data_path)
    obs_data = dict(np.load(observed_data_path, allow_pickle=True))
    print('observed data')
    for k in obs_data.keys():
        print('##########')
        print(f"{k} shape: {obs_data[k].shape}")
    # obs_data = {k: obs_data[k] for k in [cfg.sim_data, "j", "attention_mask", "magnitudes", "vlos_mask"] }
    # print('Test data keys and shape: ', test_data.keys(), test_data[list(test_data.keys())[0]].shape)
    # other_things = ['attention_mask', 'magnitudes', 'vlos_mask']
    other_things = ['attention_mask', 'magnitudes']
    keys_to_drop = set(obs_data.keys()) - set(param_names_global) - {sim_data} - set(inference_conditions) -  set(other_things)
    keys_to_drop = list(keys_to_drop) 
    augmentations_class = AugmentationsClass(cfg)
    augmentations = []
    if "remove_los_velocity" in cfg.augmentations:
        augmentations.append(augmentations_class.remove_los_velocity)
    if "sample_obs_error" in cfg.augmentations:
        augmentations.append(augmentations_class.sample_obs_error)
    if "observational_window" in cfg.augmentations:
        augmentations.append(augmentations_class.observational_window)
    if "concatentate_sigma_error_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatentate_sigma_error_to_sim_data)
    if "concatenate_magnitudes_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_magnitudes_to_sim_data)
    if "concatenate_vlos_mask_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_vlos_mask_to_sim_data)
    if "concatenate_j_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_j_to_sim_data)
    #reshape the streams dimensions
    obs_data[cfg.sim_data] = obs_data[cfg.sim_data].reshape(-1, obs_data[cfg.sim_data].shape[-2], obs_data[cfg.sim_data].shape[-1])
    obs_data['j'] = obs_data['j'].reshape(-1, 1)
    if 'vlos_mask' in obs_data:
        obs_data['vlos_mask'] = obs_data['vlos_mask'].reshape(-1, 1, obs_data[cfg.sim_data].shape[-2])
    for k in [cfg.sim_data, "attention_mask", "magnitudes", "vlos_mask"]:
        print(f"{k} shape before truncation: {obs_data[k].shape}")
        if len(obs_data[k].shape) == 2:
            obs_data[k] = obs_data[k][:, :300]
        elif (k == cfg.sim_data):
            obs_data[k] = obs_data[k][:, :300, :]
        elif (k == "attention_mask")|(k == "vlos_mask"):
            obs_data[k] = obs_data[k][:, :, :300]
        print(f"{k} shape after truncation: {obs_data[k].shape}")
    print('Observed data sim shape before augmentation: ', obs_data[cfg.sim_data].shape)
    for aug in augmentations:
        obs_data = aug(obs_data)
    for k in obs_data.keys():
        print('##########')
        print(f"{k} shape: {obs_data[k].shape}")
    
    #Using summary network
    from keras.ops import convert_to_numpy
    from bayesflow.metrics.functional import maximum_mean_discrepancy
    observed_samples = convert_to_numpy(workflow_global.approximator.summarize(obs_data,
                                                                               kwargs={'attention_mask': obs_data['attention_mask']}))
    reference_samples = convert_to_numpy(workflow_global.approximator.summarize(test_data,
                                                                               kwargs={'attention_mask': test_data['attention_mask'],})) 
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
        plt.ylabel("")
        plt.yticks([])
        plt.tick_params(axis="both", which="major", labelsize=16)
        plt.legend(fontsize=14)
        sns.despine()
        return f

    # fig = bf.diagnostics.mmd_hypothesis_test(mmd_null=distance_null, mmd_observed=distance_observed,)
    fig = mmd_hypothesis_test_numpy(mmd_null=distance_null, mmd_observed=distance_observed,)
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'distance_observed_vs_null.pdf'), bbox_inches='tight')
    print('Plot saved to ', os.path.join(cfg.base_dir, cfg.results_dir, 'distance_observed_vs_null.pdf'))





    

    

if __name__ == "__main__":
    main()
