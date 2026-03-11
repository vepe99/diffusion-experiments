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
    os.environ["KERAS_BACKEND"] = "torch"
import keras
import bayesflow as bf
from scipy import  special 


import logging
logging.getLogger('bayesflow').setLevel(logging.DEBUG)

from config.EvalConfig import EvalConfig
from utils.utils_train_jax import AugmentationsClass #we will need to use the augmentations on the test_set


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



@hydra.main(version_base=None, config_path="config", config_name="eval_config_local",)
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
        .rename(inference_conditions, "inference_conditions")
    )
    with open(os.path.join(cfg.base_dir, cfg.model_dir, '.hydra', 'config.yaml'), "r") as f:
        model_config = yaml.safe_load(f)
    print(model_config)
    if cfg.noise_schedule is not None:
        inference_network = bf.networks.DiffusionModel(
                                                        subnet_kwargs={
                                                        "widths": [model_config['local_model']['inference_mlp_width']] * model_config['local_model']['inference_mlp_depth'],
                                                        "time_embedding_dim": model_config['local_model']['inference_time_embedding_dim'],
                                                        },
                                                        schedule_kwargs = {**cfg.noise_schedule,},
                                                        )
    else:
        #probably needs to fix it to the training noise schedule 
        inference_network = bf.networks.DiffusionModel(subnet_kwargs={
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
    test_data = {k: test_data[k] for k in cfg.parameters_local + cfg.parameters_global + [cfg.sim_data, "j"] }
    # Augmentation
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
    if "mask_vlos" in cfg.augmentations:
        augmentations.append(augmentations_class.mask_vlos)
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
    # test_data[cfg.sim_data] = np.repeat(test_data[cfg.sim_data][:, None, :, :], cfg.n_samples, axis=1).reshape(cfg.multistream_n_simulation*cfg.n_samples*len(cfg.target_streams), test_data[cfg.sim_data].shape[-2], test_data[cfg.sim_data].shape[-1])
    # test_data['attention_mask'] = np.repeat(test_data['attention_mask'][:, None, :], cfg.n_samples, axis=1).reshape(cfg.multistream_n_simulation*cfg.n_samples*len(cfg.target_streams), 1, -1)
    # print('Test data sim shape after repeating: ', test_data[cfg.sim_data].shape)
    # print('Test data attention mask shape after repeating: ', test_data['attention_mask'].shape)

    #WE NEED TO GET ALSO THE SAMPLES FROM THE GLOBAL PRIOR
    global_posterior = dict(np.load('/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/plots/plots_streamax_concatenation_500_hyper_333tests/global_posterior.npz', allow_pickle=True))
    print('keys global posterior: ', global_posterior.keys())
    def expand_global_posterior(arr):
        """
        Expand global posterior samples for local inference.
        
        arr: shape (N_TEST, N_SAMPLES, ...) from global posterior
        Returns: shape (N_TEST * N_SUBJECTS * N_SAMPLES, ...)
        
        For each test case, we have N_SUBJECTS streams, and N_SAMPLES posterior samples.
        Each stream gets the same posterior samples for the global parameters.
        """
        arr = np.asarray(arr)
        # arr shape: (N_TEST, N_SAMPLES, ...)
        
        # Expand for subjects: (N_TEST, 1, N_SAMPLES, ...) -> (N_TEST, N_SUBJECTS, N_SAMPLES, ...)
        arr = np.repeat(arr[:, None, :, ...], len(cfg.target_streams), axis=1)
        
        # Flatten: (N_TEST * N_SUBJECTS * N_SAMPLES, ...)
        return arr.reshape(cfg.multistream_n_simulation * len(cfg.target_streams) * cfg.n_samples, *arr.shape[3:])
    
    def expand_local_test_param(arr):
        """
        Expand local test parameters for local inference.
        
        arr: shape (N_TEST * N_SUBJECTS, 1) — already flattened
        Returns: shape (N_TEST * N_SUBJECTS * N_SAMPLES, 1)
        
        Each (test, subject) pair gets repeated N_SAMPLES times.
        """
        arr = np.asarray(arr)
        
        # Ensure 2D: (N_TEST * N_SUBJECTS, 1)
        if arr.ndim == 1:
            arr = arr[:, None]
        
        # Repeat for samples: (N_TEST * N_SUBJECTS, 1) -> (N_TEST * N_SUBJECTS, N_SAMPLES, 1)
        arr = np.repeat(arr[:, None, :], cfg.n_samples, axis=1)
        
        # Flatten: (N_TEST * N_SUBJECTS * N_SAMPLES, 1)
        return arr.reshape(-1, arr.shape[-1])

    #conditions 
    conditions = {cfg.sim_data: test_data[cfg.sim_data],}
    conditions['j'] = expand_local_test_param(test_data['j'])
    # conditions['j'] = test_data['j']
    for param in cfg.parameters_global:
        conditions[param] = expand_global_posterior(global_posterior[param])
        print(f'Condition {param} shape after expansion: ', conditions[param].shape)
    

    logging.info("Starting Partial-Pooling (local) inference...")
    # local_posterior_flat = workflow_local.sample(
    #                     num_samples=cfg.n_samples,
    #                     conditions=conditions,
    #                     batch_size = cfg.batch_size,
    #                     kwargs={'attention_mask': test_data['attention_mask']},
    #                     )

    # def sample_in_batches(data, workflow, num_samples, batch_size, sampler_settings=None) -> dict:
    #     posterior_samples = None
    #     for i in tqdm(range(0, len(data[cfg.sim_data]), batch_size)):
    #         batch_data = {k: v[i:i + batch_size] for k, v in data.items()}
    #         if sampler_settings is None:
    #             batch_samples = workflow.sample(conditions=batch_data, 
    #                                             num_samples=num_samples, 
    #                                             kwargs={'attention_mask': test_data['attention_mask']})
    #         else:
    #             batch_samples = workflow.sample(conditions=batch_data, num_samples=num_samples,
    #                                              kwargs={'attention_mask': test_data['attention_mask']}, 
    #                                              **sampler_settings)
    #         if i == 0:
    #             posterior_samples = batch_samples
    #         else:
    #             for key in posterior_samples.keys():
    #                 posterior_samples[key] = np.vstack([posterior_samples[key], batch_samples[key]])
    #     return posterior_samples
    def sample_in_batches(data, workflow, num_samples, batch_size, sampler_settings=None) -> dict:
        """
        Batch over expanded conditions (N_TEST * N_SUBJECTS * N_SAMPLES).
        For each batch, look up the corresponding sim_data and attention_mask
        by mapping back to the observation index (i // N_SAMPLES).
        """
        posterior_samples = None
        n_total = len(data['j'])  # N_TEST * N_SUBJECTS * N_SAMPLES
        sim_data_full = test_data[cfg.sim_data]       # (N_TEST * N_SUBJECTS, N_PARTICLES, D)
        attn_mask_full = test_data['attention_mask']   # (N_TEST * N_SUBJECTS, 1, N_PARTICLES)

        if sampler_settings is None:
            pass
        else:
            workflow.approximator.inference_network.integrate_kwargs.update({
                'method': sampler_settings['method'],
                'steps': sampler_settings['steps'],
                "max_steps": sampler_settings['max_steps'],
                })
        
        for i in tqdm(range(0, n_total, batch_size)):
            batch_conds = {k: v[i:i + batch_size] for k, v in data.items()}
            actual_batch_size = batch_conds['j'].shape[0]
            
            # Map each expanded index back to the observation index
            obs_indices = np.arange(i, i + actual_batch_size) // cfg.n_samples
            
            # Gather the corresponding sim_data and attention_mask
            batch_conds[cfg.sim_data] = sim_data_full[obs_indices]
            batch_attn = attn_mask_full[obs_indices]
            
            batch_samples = workflow.sample(conditions=batch_conds, num_samples=num_samples,
                                                kwargs={'attention_mask': batch_attn}, 
                                                )
            if posterior_samples is None:
                posterior_samples = batch_samples
            else:
                for key in posterior_samples.keys():
                    posterior_samples[key] = np.vstack([posterior_samples[key], batch_samples[key]])
        return posterior_samples
    
    local_posterior_flat = sample_in_batches(
        workflow=workflow_local,
        data=conditions,
        num_samples=1,
        batch_size=cfg.batch_size*10,
        sampler_settings=dict(method="tsit5", steps=50, max_steps=50)
        )

    local_posterior = {}
    for k in local_posterior_flat.keys():
        arr = local_posterior_flat[k][:, 0]  # only one sample per condition
        # arr shape: (N_TEST * N_SUBJECTS * N_SAMPLES, ...)
        arr = arr.reshape(cfg.multistream_n_simulation * len(cfg.target_streams), cfg.n_samples, *arr.shape[1:])
        local_posterior[k] = arr
    os.makedirs(name= os.path.join(cfg.base_dir, cfg.results_dir), exist_ok=True)
    ps = local_posterior.copy()
    np.savez(os.path.join(cfg.base_dir, cfg.results_dir, 'local_posterior.npz'), **ps)

    # ps = dict(np.load(os.path.join(cfg.base_dir, cfg.results_dir, 'local_posterior.npz'), allow_pickle=True))
    # ps_clean = {}
    # for k in cfg.parameters_local:
    #     ps_clean[k] = ps[k]
    # ps = ps_clean


    test_params_local = {}
    for p in param_names_local:
        # test_data[p] shape: (N_TEST, N_SUBJECTS, 1) or similar
        arr = test_data[p]
        if arr.ndim == 3:
            # (N_TEST, N_SUBJECTS, 1) -> (N_TEST * N_SUBJECTS, 1)
            test_params_local[p] = arr.reshape(cfg.multistream_n_simulation * len(cfg.target_streams), -1)
        elif arr.ndim == 2:
            # (N_TEST, N_SUBJECTS) -> (N_TEST * N_SUBJECTS, 1)
            test_params_local[p] = arr.reshape(cfg.multistream_n_simulation * len(cfg.target_streams), 1)
        else:
            test_params_local[p] = arr.reshape(-1, 1)
    test_data = test_params_local


    ###############
    # PLOTS LOCAL#
    ###############

    # We need the j index to filter by stream
    # test_data['j'] was overwritten by test_params_local, so we need to get it from the original
    # j was stored before augmentation as (N_TEST * N_SUBJECTS, 1)
    j_flat = expand_local_test_param(np.load(test_data_path, allow_pickle=True)['j'].reshape(-1, 1))
    # j_flat: (N_TEST * N_SUBJECTS * N_SAMPLES, 1) — but we only need per-observation j
    # For test_data filtering: (N_TEST * N_SUBJECTS,)
    j_per_obs = np.load(test_data_path, allow_pickle=True)['j'].reshape(-1)  # (N_TEST * N_SUBJECTS,)

    for stream_name, j_idx in cfg.target_streams.items():
        print(f'\n===== Generating plots for {stream_name} (j={j_idx}) =====')
        
        # Filter test_data (targets): shape (N_TEST * N_SUBJECTS, 1) -> select where j == j_idx
        obs_mask = (j_per_obs == j_idx)
        test_data_stream = {k: v[obs_mask] for k, v in test_data.items()}
        
        # Filter ps (estimates): shape (N_TEST * N_SUBJECTS, N_SAMPLES, D) -> select where j == j_idx
        ps_stream = {k: v[obs_mask] for k, v in ps.items()}
        
        print(f'  test_data keys and shapes: { {k: v.shape for k, v in test_data_stream.items()} }')
        print(f'  ps keys and shapes: { {k: v.shape for k, v in ps_stream.items()} }')

        # Recovery plot
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

        # Corner plot
        # dataset_id = np.array([0])
        # fig = bf.diagnostics.plots.pairs_posterior(
        #     estimates=ps_stream,
        #     targets=test_data_stream,
        #     dataset_id=dataset_id,
        #     variable_names=cfg.parameter_local_pretty,
        # )
        # fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, f'{stream_name}_pairs_posterior_datasetid_{dataset_id[0]}.pdf'))
        # print(f'  Saved {stream_name}_pairs_posterior_datasetid_{dataset_id[0]}.pdf')
        # plt.close(fig)

        # Calibration ECDF (difference=True)
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

        # Calibration ECDF (difference=False)
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

        # Calibration histograms (split into two groups)
        ps_keys = list(ps_stream.keys())
        ps_stream_1 = {k: ps_stream[k] for k in ps_keys[:4]}
        test_data_stream_1 = {k: test_data_stream[k] for k in ps_keys[:4]}
        fig_1 = bf.diagnostics.plots.calibration_histogram(
            estimates=ps_stream_1,
            targets=test_data_stream_1,
            variable_names=cfg.parameter_local_pretty[:4]
        )
        for ax in fig_1.get_axes():
            ax.grid(False)
        fig_1.savefig(os.path.join(cfg.base_dir, cfg.results_dir, f'{stream_name}_histograms_1.pdf'))
        plt.close(fig_1)

        if len(ps_keys) > 4:
            ps_stream_2 = {k: ps_stream[k] for k in ps_keys[4:]}
            test_data_stream_2 = {k: test_data_stream[k] for k in ps_keys[4:]}
            fig_2 = bf.diagnostics.plots.calibration_histogram(
                estimates=ps_stream_2,
                targets=test_data_stream_2,
                variable_names=cfg.parameter_local_pretty[4:]
            )
            for ax in fig_2.get_axes():
                ax.grid(False)
            fig_2.savefig(os.path.join(cfg.base_dir, cfg.results_dir, f'{stream_name}_histograms_2.pdf'))
            plt.close(fig_2)

        print(f'  Saved {stream_name}_histograms plots')

        # Z-score contraction
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