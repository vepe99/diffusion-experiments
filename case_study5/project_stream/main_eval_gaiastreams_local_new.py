from autocvd import autocvd
autocvd(num_gpus=1)
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
from chainconsumer import Chain, ChainConsumer, ChainConfig
import pandas as pd

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"
import keras
import bayesflow as bf

import logging
logging.getLogger('bayesflow').setLevel(logging.DEBUG)

from config.EvalConfig import EvalConfig
from utils.utils_train_jax_new import AugmentationsClass

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


@hydra.main(version_base=None, config_path="config", config_name="eval_config_gaia_local_new")
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

    # ── Adapter ────────────────────────────────────────────────────────────
    adapter = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .concatenate(param_names_local, into="inference_variables")
        .rename(sim_data, "summary_variables")
        .concatenate(inference_conditions, into="inference_conditions")
    )

    # ── Inference network (local model keys) ───────────────────────────────
    if cfg.noise_schedule is not None:
        inference_network = bf.networks.CompositionalDiffusionModel(
            subnet_kwargs={
                "widths": [model_config['local_model']['inference_mlp_width']] * model_config['local_model']['inference_mlp_depth'],
                "time_embedding_dim": model_config['local_model']['inference_time_embedding_dim'],
            },
            schedule_kwargs={**cfg.noise_schedule},
        )
    else:
        inference_network = bf.networks.CompositionalDiffusionModel(
            subnet_kwargs={
                "widths": [model_config['local_model']['inference_mlp_width']] * model_config['local_model']['inference_mlp_depth'],
                "time_embedding_dim": model_config['local_model']['inference_time_embedding_dim'],
            },
        )

    workflow_local = bf.BasicWorkflow(
        adapter=adapter,
        summary_network=bf.networks.SetTransformer(
            summary_dim=model_config['local_model']['summary_dim'],
            num_heads=(model_config['local_model']['num_heads'], model_config['local_model']['num_heads']),
            embed_dims=(model_config['local_model']['embed_dims'], model_config['local_model']['embed_dims']),
            mlp_depths=(model_config['local_model']['mlp_depths'], model_config['local_model']['mlp_depths']),
            mlp_widths=(model_config['local_model']['mlp_widths'], model_config['local_model']['mlp_widths']),
            dropout=0.1,
        ),
        inference_network=inference_network,
        standardize=["inference_variables", "summary_variables", "inference_conditions"],
    )
    workflow_local.approximator = keras.models.load_model(model_path)
    workflow_local.approximator.inference_network.integrate_kwargs.update({
        'method':    cfg.method,
        'steps':     cfg.steps,
        'max_steps': cfg.max_steps,
    })

    # ── Load Gaia observed data ────────────────────────────────────────────
    test_data_path = '/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz'
    print('Loading Gaia data from ', test_data_path)
    test_data = dict(np.load(test_data_path, allow_pickle=True))
    test_data = {k: test_data[k] for k in [cfg.sim_data, 'j', 'attention_mask', 'magnitudes']}

    # Truncate particle axis to 300
    for k in [cfg.sim_data, 'attention_mask', 'magnitudes']:
        print(f"{k} shape before truncation: {test_data[k].shape}")
        if test_data[k].ndim == 2:
            test_data[k] = test_data[k][:, :300]
        elif test_data[k].ndim == 3:
            test_data[k] = test_data[k][:, :, :300]
        elif test_data[k].ndim == 4:
            test_data[k] = test_data[k][:, :, :300]
        print(f"{k} shape after truncation:  {test_data[k].shape}")

    # ── Augmentations ──────────────────────────────────────────────────────
    augmentations_class = AugmentationsClass(cfg)
    augmentations = []
    for name, fn in [
        ("remove_los_velocity",             augmentations_class.remove_los_velocity),
        ("sample_obs_error",                augmentations_class.sample_obs_error),
        ("observational_window",            augmentations_class.observational_window),
        ("mask_vlos",                       augmentations_class.mask_vlos),
        ("concatentate_sigma_error_to_sim_data", augmentations_class.concatentate_sigma_error_to_sim_data),
        ("concatenate_magnitudes_to_sim_data",   augmentations_class.concatenate_magnitudes_to_sim_data),
        ("concatenate_vlos_mask_to_sim_data",    augmentations_class.concatenate_vlos_mask_to_sim_data),
        ("concatenate_j_to_sim_data",            augmentations_class.concatenate_j_to_sim_data),
    ]:
        if name in cfg.augmentations:
            augmentations.append(fn)

    # Flatten stream dimension before augmentation: (1, N_STREAMS, N_STARS, D) -> (N_STREAMS, N_STARS, D)
    test_data[cfg.sim_data] = test_data[cfg.sim_data].reshape(
        -1, test_data[cfg.sim_data].shape[-2], test_data[cfg.sim_data].shape[-1]
    )
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
    print('Gaia sim shape after augmentation:  ', test_data[cfg.sim_data].shape)
    print('j shape:                            ', test_data['j'].shape)
    print('attention_mask shape:               ', test_data['attention_mask'].shape)

    # ── Load global posterior ──────────────────────────────────────────────
    global_posterior_path = os.path.join(
        cfg.base_dir,
        'plots/gala6D_aug/new_hyper/model54_60k_1000epochs/global_posterior.npz'
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
        cfg.sim_data: test_data[cfg.sim_data],   # (1, N_STREAMS, N_STARS, D)
        'j':          test_data['j'],            # (1, N_STREAMS, 1)
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