from autocvd import autocvd

autocvd(num_gpus = 1)
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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
    if cfg.noise_schedule is not None:
        inference_network = bf.networks.CompositionalDiffusionModel(
                                                        subnet_kwargs={
                                                        "widths": [model_config['local_model']['inference_mlp_width']] * model_config['local_model']['inference_mlp_depth'],
                                                        "time_embedding_dim": model_config['local_model']['inference_time_embedding_dim'],
                                                        },
                                                        schedule_kwargs = {**cfg.noise_schedule,},
                                                        )
    else:
        #probably needs to fix it to the training noise schedule 
        inference_network = bf.networks.CompositionalDiffusionModel(subnet_kwargs={
                                                        "widths": [model_config['local_model']['inference_mlp_width']] * model_config['local_model']['inference_mlp_depth'],
                                                        "time_embedding_dim": model_config['local_model']['inference_time_embedding_dim'],
                                                        },)
    workflow_local = bf.BasicWorkflow(
        adapter=adapter,
        summary_network=bf.networks.SetTransformer(summary_dim=model_config['local_model']['summary_dim'], 
                                                   num_heads=(model_config['local_model']['num_heads'],model_config['local_model']['num_heads'],),
                                                   embed_dims = (model_config['local_model']['embed_dims'], model_config['local_model']['embed_dims'],),
                                                   mlp_depths=(model_config['local_model']['mlp_depths'], model_config['local_model']['mlp_depths']),
                                                   mlp_widths=(model_config['local_model']['mlp_widths'], model_config['local_model']['mlp_widths']),
                                                   dropout=0.1),
        inference_network=inference_network,
        standardize=["inference_variables", "summary_variables", "inference_conditions"],
    )
    workflow_local.approximator = keras.models.load_model(model_path)
    workflow_local.approximator.inference_network.integrate_kwargs.update({
        'method': cfg.method,
        'steps': cfg.steps,
        "max_steps": cfg.max_steps,
        })
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
    test_data['j'] = test_data['j'].reshape(-1, 1)
    print('Test data sim shape before augmentation: ', test_data[cfg.sim_data].shape)
    for aug in augmentations:
        test_data = aug(test_data)
    print('Test data sim shape after augmentation: ', test_data[cfg.sim_data].shape)
    print('Test data attention mask shape: ', test_data['attention_mask'].shape)
    print('###############')
    print('Repeating for each posterior sample the sim data and also the attention mask')

    #WE NEED TO GET ALSO THE SAMPLES FROM THE GLOBAL PRIOR
    global_posterior = dict(np.load('/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/plots/gala6D_aug/new_hyper/model54_60k_1000epochs/100test/global_posterior.npz', allow_pickle=True))
    print('keys global posterior: ', global_posterior.keys())

    logging.info("Starting Partial-Pooling (local) inference...")

    # --- Build conditions with correct shapes for ancestral_sample ---
    n_test    = cfg.multistream_n_simulation          # e.g. 100 test cases
    n_streams = len(cfg.target_streams)               # e.g. 3 streams

    # sim_data: (N_TEST * N_STREAMS, N_PARTICLES, D) -> (N_TEST, N_STREAMS, N_PARTICLES, D)
    sim_data_4d = test_data[cfg.sim_data].reshape(
        n_test, n_streams, *test_data[cfg.sim_data].shape[1:]
    )

    # j: (N_TEST * N_STREAMS, 1) -> (N_TEST, N_STREAMS, 1)
    j_3d = test_data['j'].reshape(n_test, n_streams, 1)


    # attention_mask: (N_TEST * N_STREAMS, 1, N_PARTICLES) -> (N_TEST, N_STREAMS, 1, N_PARTICLES)
    attn_mask_4d = test_data['attention_mask'].reshape(
        n_test, n_streams, *test_data['attention_mask'].shape[1:]
    )
    # attn_mask_4d = test_data['attention_mask']

    conditions = {
        cfg.sim_data: sim_data_4d,   # (N_TEST, N_STREAMS, N_PARTICLES, D)
        'j':          j_3d,          # (N_TEST, N_STREAMS, 1)
    }

    # ancestral_conditions must be (N_TEST, N_PARENT_SAMPLES, 1) — do NOT repeat by n_streams
    # global_posterior[param] is already (N_TEST, N_PARENT_SAMPLES, 1), use as-is
    ancestral_conds = {
        param: global_posterior[param]   # (N_TEST, N_PARENT_SAMPLES, 1)
        for param in cfg.parameters_global
    }

    # Also add global params to conditions so the adapter can see them
    for param in cfg.parameters_global:
        # repeat each param across streams: (N_TEST, 1, 1) -> broadcast or explicit repeat
        conditions[param] = np.repeat(
            global_posterior[param][:, :1, :],   # take first sample as placeholder shape
            n_streams, axis=1
        )  # shape: (N_TEST, N_STREAMS, 1) — the ancestral_sample will handle the actual conditioning


    def ancestral_sample_batched(workflow, conditions, ancestral_conds, attn_mask_4d, cfg, n_test_per_batch=5):
        """
        Manually batch over the n_datasets (test cases) axis to avoid OOM,
        since _prepare_ancestral_conditions converts everything to GPU at once.
        """
        n_test = cfg.multistream_n_simulation
        all_samples = None

        for i in tqdm(range(0, n_test, n_test_per_batch)):
            # Slice along n_datasets axis for all inputs
            batch_conditions = {k: v[i:i + n_test_per_batch] for k, v in conditions.items()}
            batch_ancestral  = {k: v[i:i + n_test_per_batch] for k, v in ancestral_conds.items()}
            batch_attn       = attn_mask_4d[i:i + n_test_per_batch]

            batch_samples = workflow.ancestral_sample(
                conditions=batch_conditions,
                ancestral_conditions=batch_ancestral,
                kwargs={'attention_mask': batch_attn},
            )

            if all_samples is None:
                all_samples = batch_samples
            else:
                for k in all_samples:
                    all_samples[k] = np.concatenate([all_samples[k], batch_samples[k]], axis=0)

        return all_samples


    local_posterior = ancestral_sample_batched(
        workflow=workflow_local,
        conditions=conditions,
        ancestral_conds=ancestral_conds,
        attn_mask_4d=attn_mask_4d,
        cfg=cfg,
        n_test_per_batch=cfg.batch_size,   # tune this down if still OOM, up for speed
    )

    for k in local_posterior.keys():
        print(f'Local posterior {k} shape: ', local_posterior[k].shape)



    os.makedirs(name=os.path.join(cfg.base_dir, cfg.results_dir), exist_ok=True)
    ps = local_posterior.copy()
    # ps shape: (N_TEST, N_STREAMS, N_PARENT_SAMPLES, 1)
    np.savez(os.path.join(cfg.base_dir, cfg.results_dir, 'local_posterior.npz'), **ps)

    # Build ground-truth targets: (N_TEST * N_STREAMS, 1) -> (N_TEST, N_STREAMS, 1)
    test_params_local = {}
    for p in param_names_local:
        arr = test_data[p]
        if arr.ndim == 3:
            test_params_local[p] = arr.reshape(n_test, n_streams, -1)
        elif arr.ndim == 2:
            test_params_local[p] = arr.reshape(n_test, n_streams, 1)
        else:
            test_params_local[p] = arr.reshape(n_test, n_streams, 1)
    # test_params_local[p] shape: (N_TEST, N_STREAMS, 1)

    
    ###############
    # PLOTS LOCAL #
    ###############
    test_params_local = {}
    for p in param_names_local:
        arr = np.asarray(test_data[p])  # force numpy, not JAX array
        print(f'  {p} raw shape: {arr.shape}')
        if arr.ndim == 3:
            # (N_TEST, N_STREAMS, D) -> (N_TEST * N_STREAMS, D)
            test_params_local[p] = arr.reshape(n_test * n_streams, -1)
        elif arr.ndim == 2 and arr.shape[0] == n_test and arr.shape[1] == n_streams:
            # (N_TEST, N_STREAMS) -> (N_TEST * N_STREAMS, 1)
            test_params_local[p] = arr.reshape(n_test * n_streams, 1)
        elif arr.ndim == 2:
            # already (N_TEST * N_STREAMS, D)
            test_params_local[p] = arr
        elif arr.ndim == 1 and arr.shape[0] == n_test * n_streams:
            # (N_TEST * N_STREAMS,) -> (N_TEST * N_STREAMS, 1)
            test_params_local[p] = arr.reshape(-1, 1)
        elif arr.ndim == 1 and arr.shape[0] == n_test:
            # (N_TEST,) -> repeat for each stream -> (N_TEST * N_STREAMS, 1)
            test_params_local[p] = np.repeat(arr, n_streams).reshape(-1, 1)
        else:
            raise ValueError(f'Unexpected shape for {p}: {arr.shape}')
    test_data = test_params_local

    ###############
    # PLOTS LOCAL #
    ###############

    # Flatten ps: (N_TEST, N_STREAMS, N_SAMPLES, D) -> (N_TEST * N_STREAMS, N_SAMPLES, D)
    ps_flat = {k: np.asarray(v).reshape(n_test * n_streams, *v.shape[2:]) for k, v in ps.items()}

    # j per observation: force numpy to avoid JAX boolean indexing issues
    j_per_obs = np.asarray(np.load(test_data_path, allow_pickle=True)['j']).reshape(-1)
    print(f'j_per_obs shape: {j_per_obs.shape}')
    print(f'test_data sample shape: {next(iter(test_data.values())).shape}')
    print(f'ps_flat sample shape: {next(iter(ps_flat.values())).shape}')

    for stream_name, j_idx in cfg.target_streams.items():
        print(f'\n===== Generating plots for {stream_name} (j={j_idx}) =====')

        obs_mask = (j_per_obs == j_idx)  # (N_TEST * N_STREAMS,) numpy boolean

        test_data_stream = {k: v[obs_mask] for k, v in test_data.items()}
        ps_stream        = {k: v[obs_mask] for k, v in ps_flat.items()}

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