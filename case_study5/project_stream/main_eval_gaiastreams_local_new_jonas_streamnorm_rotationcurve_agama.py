from autocvd import autocvd
autocvd(num_gpus=1)
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
import jax.numpy as jnp

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"
import bayesflow as bf
import keras

import logging
logging.getLogger('bayesflow').setLevel(logging.DEBUG)

from config.EvalConfig import EvalConfig
from utils.utils_train_jax_new_rotationcurve_fixedvlosmask import AugmentationsClass
from utils.custom_summary_network import SetTransformer, FusionNetwork


cs = ConfigStore.instance()
cs.store(name="eval_config", node=EvalConfig)


def fix_keras_model(model_path):
    import zipfile
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
    #apply normalization to vcirc_kms, the vcirc_kms should already be in log10 
    x = jnp.array(batch["vcirc_kms"])  # (N, 34, 1)
    mean_vcirc = jnp.array(stats["vcirc_kms"].item()["mean_log10vcirc_kms"])  # (34, 1)
    std_vcirc  = jnp.array(stats["vcirc_kms"].item()["std_log10vcirc_kms"])   # (34, 1)
    x = (x - mean_vcirc) / std_vcirc
    batch["vcirc_kms"] = jnp.array(x)
    return batch



@hydra.main(version_base=None, config_path="config", config_name="eval_config_gaia_local_new_rotationcurve_agama")
def main(cfg: EvalConfig):

    print(cfg)
    print("##############")

    # ── Model ──────────────────────────────────────────────────────────────
    model_path = os.path.join(cfg.base_dir, cfg.model_dir, 'local_model.keras')
    model_path = fix_keras_model(model_path)
    print('Loading model from ', model_path)
    print("##############")

    param_names_local  = list(cfg.parameters_local)
    param_names_global = list(cfg.parameters_global)
    sim_data           = str(cfg.sim_data)
    inference_conditions_key = str(cfg.inference_conditions[0])   # 'j'
    inference_conditions     = param_names_global + [inference_conditions_key]

    with open(os.path.join(cfg.base_dir, cfg.model_dir, '.hydra', 'config.yaml'), 'r') as f:
        model_config = yaml.safe_load(f)
    print(model_config)


    # ── Load Gaia observed data ────────────────────────────────────────────
    test_data_path = '/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz'
    print('Loading Gaia data from ', test_data_path)
    test_data = dict(np.load(test_data_path, allow_pickle=True))
    test_data = {k: test_data[k] for k in [cfg.sim_data, "j", "attention_mask", "magnitudes", 'vlos_error', 'vlos_mask']}

    # Truncate particle axis to 300
    for k in [cfg.sim_data, 'attention_mask', 'magnitudes', 'vlos_error', 'vlos_mask']:
        print(f"{k} shape before truncation: {test_data[k].shape}")
        if test_data[k].ndim == 2:
            test_data[k] = test_data[k][:, :300]
            if k == "vlos_mask":
                test_data[k] = test_data[k][:, None, :]
        elif test_data[k].ndim == 3:
            test_data[k] = test_data[k][:, :, :300]
        elif test_data[k].ndim == 4:
            test_data[k] = test_data[k][:, :, :300]
        print(f"{k} shape after truncation:  {test_data[k].shape}")

    augmentations_class = AugmentationsClass(cfg)
    mask_r_kpc = (augmentations_class.obs_R >5.5)
    test_data['vcirc_kms'] = augmentations_class.obs_Vc[None, mask_r_kpc, None]

    stats = np.load(os.path.join(os.path.dirname(model_path), 'stream_stats.npz'), allow_pickle=True)

    # ── Adapter ────────────────────────────────────────────────────────────
    keys_to_drop = (
        set(test_data.keys()) 
        - set(param_names_local) 
        - set(param_names_global) 
        - {sim_data} 
        - {"vcirc_kms"}
        - set(inference_conditions)
    )
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
    workflow_local.approximator = keras.models.load_model(model_path) #this override everything 



    # ── Augmentations ──────────────────────────────────────────────────────
    augmentations = []
    augmentations.append(augmentations_class.sample_obs_error)
    augmentations.append(augmentations_class.log10_vcirc)
    augmentations.append(lambda batch: standardize_by_stream(batch, sim_data=sim_data, stats=stats))  # re-standardize after flip_dirz
    augmentations.append(augmentations_class.concatentate_sigma_error_to_sim_data)
    augmentations.append(augmentations_class.concatenate_magnitudes_to_sim_data)
    augmentations.append(augmentations_class.concatenate_vlos_mask_to_sim_data)
    augmentations.append(augmentations_class.concatenate_j_to_sim_data)
    
    

    # Flatten stream dimension before augmentation: (1, N_STREAMS, N_STARS, D) -> (N_STREAMS, N_STARS, D)
    test_data[cfg.sim_data] = test_data[cfg.sim_data].reshape(
        -1, test_data[cfg.sim_data].shape[-2], test_data[cfg.sim_data].shape[-1]
    )
    test_data['vcirc_kms'] = np.repeat(test_data['vcirc_kms'], 3, axis=0)
    test_data['j'] = test_data['j'].reshape(-1, 1)
    print('Gaia sim shape before augmentation: ', test_data[cfg.sim_data].shape)

    for aug in augmentations:
        print(f"Applying augmentation: {aug.__name__}")
        test_data = aug(test_data)

    # Restore stream dimension: (N_STREAMS, N_STARS, D) -> (1, N_STREAMS, N_STARS, D)
    n_streams = len(cfg.target_streams)
    test_data[cfg.sim_data] = test_data[cfg.sim_data].reshape(
        1, n_streams, test_data[cfg.sim_data].shape[-2], test_data[cfg.sim_data].shape[-1]
    )
    test_data['j']              = test_data['j'].reshape(1, n_streams, 1)
    test_data['attention_mask'] = test_data['attention_mask'].reshape(
        1, n_streams, *test_data['attention_mask'].shape[1:]
    )
    test_data['vcirc_kms']      = test_data['vcirc_kms'].reshape(
        1, n_streams, test_data['vcirc_kms'].shape[-2], test_data['vcirc_kms'].shape[-1]
    )
    print('Gaia sim shape after augmentation:  ', test_data[cfg.sim_data].shape)
    print('j shape:                            ', test_data['j'].shape)
    print('attention_mask shape:               ', test_data['attention_mask'].shape)

    # ── Load global posterior ──────────────────────────────────────────────
    global_posterior_path = os.path.join(
        cfg.base_dir,
        # 'plots/gala6D/new_hyper/model54_60k_1000epochs/global_posterior.npz'
        # 'hyperparameter_tuning/gala/300k/model_21/gaiastreams/global_posterior.npz'
        # 'hyperparameter_tuning/gala/new_aug_bigheads/model_121/gaiastreams/global_posterior.npz'
        # 'hyperparameter_tuning/gala/rotationcurve/model_9_test/gaiastreams/global_posterior.npz'
        'hyperparameter_tuning/agama/rotationcurve/model_5/gaiastreams/global_posterior.npz'
    )
    print('Loading global posterior from ', global_posterior_path)
    global_posterior = dict(np.load(global_posterior_path, allow_pickle=True))
    print('Global posterior keys: ', list(global_posterior.keys()))
    for k, v in global_posterior.items():
        print(f'  {k}: {v.shape}')

    # global_posterior[param] expected shape: (1, N_PARENT_SAMPLES, 1)
    # If it comes out as (N_PARENT_SAMPLES,) or (N_PARENT_SAMPLES, 1), reshape accordingly
    for param in param_names_global:
        arr = np.asarray(global_posterior[param])
        if arr.ndim == 1:
            arr = arr.reshape(1, -1, 1)
        elif arr.ndim == 2:
            arr = arr.reshape(1, arr.shape[0], 1)
        global_posterior[param] = arr
        print(f'  {param} reshaped to: {arr.shape}')

    # ── Build conditions ───────────────────────────────────────────────────
    # conditions shape: (1, N_STREAMS, ...)  — single "real" observation
    conditions = {
        "input_a": test_data[cfg.sim_data],         # (300, 300, 15)
        "input_b": test_data["vcirc_kms"],            # (300, 34, 1)  <-- missing
        "summary_attention_mask": test_data["attention_mask"],      # (300, 1, 300)
        "j": test_data["j"],                            # (300, 1)
    }

    # ancestral_conditions: (1, N_PARENT_SAMPLES, 1) per global param
    ancestral_conds = {param: global_posterior[param] for param in param_names_global}

    logging.info("Starting local ancestral sampling on Gaia data...")

    local_posterior = workflow_local.ancestral_sample(
        # num_samples=cfg.n_samples,
        conditions=conditions,
        ancestral_conditions=ancestral_conds,
        batch_size=cfg.batch_size,
        kwargs={'attention_mask': test_data['attention_mask']},
    )

    # ── Save posterior ─────────────────────────────────────────────────────
    os.makedirs(os.path.join(cfg.base_dir, cfg.results_dir), exist_ok=True)

    with open("./config/prior_local.yaml", "r") as f:
        prior_local_dict = yaml.safe_load(f)
    for key in param_names_local:
        print('We are going to renormalize the parameter', key, 'for each stream separately using the prior parameters from prior_local.yaml')
        for name, j_idx in cfg.target_streams.items():
            mean_prior = prior_local_dict[name][key]['prior_parameters'][0]
            std_prior  = prior_local_dict[name][key]['prior_parameters'][1]
            # local_posterior[key] shape: (N_TEST, N_STREAMS, N_SAMPLES, 1)
            # index stream axis directly with j_idx
            local_posterior[key][:, j_idx, :, :] = (
                local_posterior[key][:, j_idx, :, :] * std_prior + mean_prior
            )
            print(f"Renormalized {key} for stream {name} (j={j_idx}) using mean={mean_prior}, std={std_prior}")
            print(f"Min and max: {local_posterior[key][:, j_idx].min():.4f}, {local_posterior[key][:, j_idx].max():.4f}")
    ps = local_posterior.copy()
    
    np.savez(os.path.join(cfg.base_dir, cfg.results_dir, 'gaia_local_posterior.npz'), **ps)
    print('Saved gaia_local_posterior.npz')
    for k, v in ps.items():
        print(f'  {k}: {np.asarray(v).shape}')

    # ── Corner plots per stream ────────────────────────────────────────────
    # ps[param] shape: (1, N_STREAMS, N_PARENT_SAMPLES, N_LOCAL_SAMPLES, 1)
    # or (1, N_STREAMS, N_SAMPLES, D) depending on BayesFlow version — squeeze safely

    for stream_name, j_idx in cfg.target_streams.items():
        print(f'\n===== Corner plot for {stream_name} (j={j_idx}) =====')

        stream_samples = {}
        for param, pretty in zip(param_names_local, cfg.parameter_local_pretty):
            arr = np.asarray(ps[param])          # flatten everything except last dim
            # Select the stream axis (axis=1 after the dataset axis=0)
            arr_stream = arr[0, j_idx]           # shape: (N_SAMPLES, ...) or (N_PARENT_SAMPLES, N_LOCAL_SAMPLES, 1)
            stream_samples[pretty] = arr_stream.reshape(-1)   # flatten to 1D for ChainConsumer

        df = pd.DataFrame(stream_samples)
        print(f'  DataFrame shape: {df.shape}')

        c = ChainConsumer()
        c.add_chain(Chain(samples=df, name=stream_name))
        c.set_override(ChainConfig(shade=True))

        fig = c.plotter.plot()
        out_path = os.path.join(cfg.base_dir, cfg.results_dir, f'{stream_name}_local_cornerplot.pdf')
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {out_path}')


if __name__ == "__main__":
    main()