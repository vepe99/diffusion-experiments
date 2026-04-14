from autocvd import autocvd
autocvd(num_gpus = 1)
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
from tqdm import tqdm

import numpy as np

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"
import gc

import bayesflow as bf
from bayesflow.diagnostics import metrics as bf_metrics
import yaml
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import logging
logging.getLogger('bayesflow').setLevel(logging.DEBUG)

from utils.utils_train_jax_new import AugmentationsClass
import jax

def clear_gpu_memory():
    """Helper function to aggressively clear GPU memory."""
    gc.collect()


def objective(trial, cfg, test_data):
    clear_gpu_memory()
    results_dir = f'./data/hyperparameter_tuning/gala/local/new_aug_jonas/model_{trial.number}/'
    os.makedirs(results_dir, exist_ok=True)

    summary_dim = trial.suggest_int("SetTransformer_summary_dim", 32, 128)
    embed_dims = trial.suggest_int("SetTransformer_embed_dims", 32, 128)
    num_heads = trial.suggest_int("SetTransformer_num_heads", 1, 3)
    mlp_depths = trial.suggest_int("SetTransformer_mlp_depths", 2, 6) 
    mlp_widths = trial.suggest_int("SetTransformer_mlp_widths", 32, 128)

    inference_mlp_depth = trial.suggest_int("inference_mlp_depth", 2, 8)
    inference_mlp_width = trial.suggest_int("inference_mlp_width", 32, 256)
    time_embedding_dim = trial.suggest_int("inference_time_embedding_dim", 16, 64, step=2)

    # --- Validate hyperparameters before touching Keras ---
    if embed_dims % num_heads != 0:
        logging.warning(
            f"Trial {trial.number} pruned: embed_dims={embed_dims} not divisible by num_heads={num_heads}"
        )
        raise optuna.TrialPruned("embed_dims must be divisible by num_heads")

    param_names_global = list(cfg.parameters_global)
    param_names_local = list(cfg.parameters_local)
    sim_data = 'sim_data_projected'
    inference_conditions = param_names_global + ['j']
    keys_to_drop = list(
        set(training_data.keys())
        - set(param_names_local)
        - set(param_names_global)
        - {sim_data}
        - set(inference_conditions)
    )

    try:
        adapter = (
            bf.adapters.Adapter()
            .to_array()
            .convert_dtype("float64", "float32")
            .drop(keys_to_drop)
            .concatenate(param_names_local, into="inference_variables")
            .rename(sim_data, "summary_variables")
            .concatenate(inference_conditions, into="inference_conditions")
        )

        workflow_local = bf.BasicWorkflow(
            adapter=adapter,
            summary_network=bf.networks.SetTransformer(
                summary_dim=summary_dim,
                embed_dims=(embed_dims, embed_dims),
                num_heads=(num_heads, num_heads),
                mlp_depths=(mlp_depths, mlp_depths),
                mlp_widths=(mlp_widths, mlp_widths),
                dropout=0.1,
            ),
            inference_network=bf.networks.CompositionalDiffusionModel(
                subnet_kwargs={
                    "widths": [inference_mlp_width] * inference_mlp_depth,
                    "time_embedding_dim": time_embedding_dim,
                }
            ),
            standardize=["inference_variables", "summary_variables", "inference_conditions"],
        )

        # --- Training with batch-size retry ---
        batch_size_training = 1024
        for attempt in range(2):
            try:
                history = workflow_local.fit_offline(
                    training_data,
                    epochs=1000,
                    batch_size=batch_size_training,
                    verbose=2,
                    augmentations=augmentations,
                )
                break  # success
            except Exception as e:
                err_str = str(e).lower()
                is_oom = "out of memory" in err_str or "cuda" in err_str or "resource exhausted" in err_str
                if is_oom and attempt == 0:
                    logging.warning(f"Trial {trial.number} OOM during training, halving batch size and retrying...")
                    batch_size_training //= 2
                    clear_gpu_memory()
                else:
                    raise

        #sampling
        logging.info("Starting Partial-Pooling (local) inference...")
        conditions = {cfg.sim_data: test_data[cfg.sim_data],}
        conditions['j'] = test_data['j']
        for param in cfg.parameters_global:
            conditions[param] = test_data[param]

        # --- Sampling with batch-size retry ---
        batch_size_sampling = 250
        for attempt in range(3):
            try:
                local_posterior = workflow_local.sample(
                        num_samples=500,
                        conditions=conditions, 
                        batch_size=cfg.batch_size,
                        kwargs={'attention_mask': test_data['attention_mask']}
                    )
                break
            except Exception as e:
                err_str = str(e).lower()
                is_oom = "out of memory" in err_str or "cuda" in err_str or "resource exhausted" in err_str
                if is_oom and attempt == 0:
                    logging.warning(f"Trial {trial.number} OOM during sampling, halving batch size and retrying...")
                    batch_size_sampling //= 2
                    clear_gpu_memory()
                else:
                    raise
        
        #jonas denormalization suggestion
        for key in param_names_local:
            print('We are going to renormalize the parameter', key, 'for each stream separately using the prior parameters from prior_local.yaml')
            for name in cfg.target_streams.keys():
                mask_stream = (test_data["j"] == cfg.target_streams[name]).squeeze()  # (300,)
                mean_prior = prior_local_dict[name][key]['prior_parameters'][0]
                std_prior  = prior_local_dict[name][key]['prior_parameters'][1]
                local_posterior[key][mask_stream] = (local_posterior[key][mask_stream] * std_prior + mean_prior)
                print(f"Renormalized {key} for stream {name} using mean={mean_prior} and std={std_prior}")
        


        root_mean_squared_error = bf_metrics.root_mean_squared_error(
            estimates=local_posterior,
            targets=test_data,
            variable_keys=param_names_local,
            variable_names=param_names_local,
        )
        calibration_errors = bf_metrics.calibration_error(
            estimates=local_posterior,
            targets=test_data,
            variable_keys=param_names_local,
            variable_names=param_names_local,
        )

        workflow_local.approximator.save(os.path.join(results_dir, "local_model.keras"))
        #calibration plot with diff
        for stream_name, j_idx in cfg.target_streams.items():
            print(f'\n===== Generating plots for {stream_name} (j={j_idx}) =====')

            obs_mask = (test_data['j'] == j_idx).squeeze()  # (300,) instead of (300, 1)

            test_data_stream = {k: v[obs_mask] for k, v in test_data.items()}
            ps_stream        = {k: v[obs_mask] for k, v in local_posterior.items()}

            print(f'  test_data shapes: { {k: v.shape for k, v in test_data_stream.items()} }')
            print(f'  ps shapes:        { {k: v.shape for k, v in ps_stream.items()} }')
            # --- Calibration ECDF (difference=True) ---
            fig = bf.diagnostics.calibration_ecdf(
                estimates=ps_stream,
                targets=test_data_stream,
                difference=True,
                variable_names=cfg.parameter_local_pretty
            )
            for ax in fig.get_axes():
                ax.grid(False)
            fig.savefig(os.path.join(results_dir, f'{stream_name}_calibration.pdf'))
            print(f'  Saved {stream_name}_calibration.pdf')

        return root_mean_squared_error['values'].mean(), calibration_errors['values'].mean()

    except optuna.TrialPruned:
        raise  # let Optuna handle it cleanly

    except Exception as e:
        err_str = str(e).lower()
        is_oom = "out of memory" in err_str or "cuda" in err_str or "resource exhausted" in err_str

        logging.error(f"Trial {trial.number} failed: {type(e).__name__}: {e}")

        for var_name in ("workflow_global", "local_posterior", "history"):
            try:
                del locals()[var_name]
            except KeyError:
                pass
        clear_gpu_memory()

        if is_oom:
            raise optuna.TrialPruned(f"OOM: {e}")
        else:
            raise optuna.TrialPruned(f"{type(e).__name__}: {e}")
    

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from config.TrainConfig import TrainConfig
from optuna.storages import JournalStorage, JournalFileStorage


if __name__ == "__main__":
    # Initialize Hydra config without the decorator
    cs = ConfigStore.instance()
    cs.store(name="train_config_schema", node=TrainConfig)

    # 2. Point to the directory containing train_config.yaml
    config_path = '/export/data/vgiusepp/diffusion_experiments_test_new/diffusion-experiments/case_study5/project_stream/config/'

    with initialize_config_dir(version_base=None, config_dir=config_path):
        # 3. Compose: loads train_config.yaml, validated against TrainConfig schema
        cfg = compose(config_name="train_config_local_new")
    
    base_dir =  '/export/data/vgiusepp/diffusion_experiments_test_new/diffusion-experiments/case_study5/project_stream/data/'
    data_dir = 'streams/data_gala/'
    N_simulations = 300_000


    train_data_path = os.path.join(base_dir, data_dir, f"training_data_local_{N_simulations}.npz")
    training_data = dict(np.load(train_data_path, allow_pickle=True))
    training_data = {k: training_data[k][:290_000] for k in training_data.keys()}

    #jonas suggestion
    with open("./config/prior_local.yaml", "r") as f:
        prior_local_dict = yaml.safe_load(f)
    
    param_names_local = list(cfg.parameters_local)

    for key in param_names_local:
        print('We are going to renormalize the parameter', key, 'for each stream separately using the prior parameters from prior_local.yaml')
        for name in cfg.target_streams.keys():
            mask_stream = training_data["j"] == cfg.target_streams[name]
            mean_prior = prior_local_dict[name][key]['prior_parameters'][0]
            std_prior = prior_local_dict[name][key]['prior_parameters'][1]
            training_data[key][mask_stream] = (training_data[key][mask_stream] - mean_prior) / std_prior
            print(f"Renormalized {key} for stream {name} using mean={mean_prior} and std={std_prior}")
            print('Min and max of the renormalized parameter for this stream:', training_data[key][mask_stream].min(), training_data[key][mask_stream].max())


    augmentations_class = AugmentationsClass(cfg)
    augmentations_class.key = jax.random.PRNGKey(0)
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


    test_data = dict(np.load(train_data_path, allow_pickle=True))
    test_data = {k: test_data[k][-1_000:] for k in test_data.keys()}
    for aug in augmentations:
        test_data = aug(test_data)
    for k in test_data.keys():
        test_data[k] = np.array(test_data[k])

    augmentations_class.key = jax.random.PRNGKey(42)
        
    print("Loaded config:", cfg)
    study_name = 'study_DiffusionMode_local'  # Unique identifier of the study.
    storage_name = JournalStorage(JournalFileStorage("./data/hyperparameter_tuning/gala/local/new_aug_jonas/optuna_diffusionmodel_gala_local_cutNGC3201.log"))
    study = optuna.create_study(study_name=study_name, storage=storage_name, directions=['minimize', 'minimize'], load_if_exists=True)
    study.optimize(
        lambda trial: objective(trial, cfg, test_data),
        callbacks=[MaxTrialsCallback(200, states=(TrialState.COMPLETE, TrialState.FAIL))],
    )