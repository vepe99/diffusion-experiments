from autocvd import autocvd
autocvd(num_gpus = 1, interval=1)
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  
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
import jax.numpy as jnp

def clear_gpu_memory():
    """Helper function to aggressively clear GPU memory."""
    gc.collect()

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

    # save_path = os.path.join(model_path, 'stream_stats.npz')
    # np.savez(save_path, **stats)
    # print(f"Stream stats saved to {save_path}")
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
    return batch



def objective(trial, cfg):
    augmentations_class.key = jax.random.PRNGKey(42)
    # Clear memory at the start of each trial
    clear_gpu_memory()
    results_dir = f'./data/hyperparameter_tuning/streamax_new/streamax_new/100k/model_{trial.number}/'
    os.makedirs(results_dir, exist_ok=True)

    
    summary_dim = trial.suggest_int("SetTransformer_summary_dim", 32, 128)
    num_heads = trial.suggest_int("SetTransformer_num_heads", 1, 8)
    embed_dim_multiplier = trial.suggest_int("SetTransformer_embed_dim_multiplier", 4, 16)
    embed_dims = embed_dim_multiplier * num_heads  # always divisible, range ~16-128
    mlp_depths = trial.suggest_int("SetTransformer_mlp_depths", 2, 6) 
    mlp_widths = trial.suggest_int("SetTransformer_mlp_widths", 32, 128)

    inference_mlp_depth = trial.suggest_int("inference_mlp_depth", 2, 8)
    inference_mlp_width = trial.suggest_int("inference_mlp_width", 32, 256, step=16)
    time_embedding_dim = trial.suggest_int("inference_time_embedding_dim", 16, 64, step=2)

    # --- Validate hyperparameters before building the model ---
    if embed_dims % num_heads != 0:
        logging.warning(
            f"Trial {trial.number} pruned: embed_dims={embed_dims} not divisible by num_heads={num_heads}"
        )
        raise optuna.TrialPruned("embed_dims must be divisible by num_heads")


    param_names_global = list(cfg.parameters_global)
    sim_data = 'sim_data_projected'
    inference_conditions = 'j' #just 1
    keys_to_drop = (
        set(training_data.keys())
        - set(param_names_global)
        - {sim_data}
        - set(inference_conditions)
    )
    keys_to_drop = list(keys_to_drop)
        
    try:
        adapter = (
            bf.adapters.Adapter()
            .to_array()
            .drop(keys_to_drop)
            .convert_dtype("float64", "float32")
            .concatenate(param_names_global, into="inference_variables")
            .rename(sim_data, "summary_variables")
            .rename(inference_conditions, "inference_conditions")
        )

        workflow_global = bf.BasicWorkflow(
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
            standardize=["inference_variables", "summary_variables"],
        )

        # --- Training with batch-size retry ---
        batch_size_training = 1024
        for attempt in range(2):
            try:
                history = workflow_global.fit_offline(
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
                    raise  # non-OOM or second failure — re-raise

        # --- Sampling with batch-size retry ---
        batch_size_sampling = 250
        for attempt in range(3):
            try:
                global_posterior = workflow_global.sample(
                    num_samples=500,
                    conditions={
                        cfg.sim_data: test_data[cfg.sim_data],
                        "j": test_data["j"],
                    },
                    batch_size=batch_size_sampling,
                    kwargs={"attention_mask": test_data["attention_mask"]},
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

        root_mean_squared_error = bf_metrics.root_mean_squared_error(
            estimates=global_posterior,
            targets=test_data,
            variable_keys=param_names_global,
            variable_names=param_names_global,
        )
        calibration_errors = bf_metrics.calibration_error(
            estimates=global_posterior,
            targets=test_data,
            variable_keys=param_names_global,
            variable_names=param_names_global,
        )
        workflow_global.approximator.save(os.path.join(results_dir, "global_model.keras"))
        workflow_global.approximator.save_weights(os.path.join(results_dir, "global_model.weights.h5"))
        #calibration plot with diff
        fig = bf.diagnostics.calibration_ecdf(
            estimates=global_posterior,
            targets=test_data,
            difference=True,
            variable_names=cfg.paramater_global_pretty,
        )
        for ax in fig.get_axes():
            ax.grid(False)
        fig.savefig(os.path.join(results_dir, 'calibration.pdf'))
        return root_mean_squared_error["values"].mean(), calibration_errors["values"].mean()

    except optuna.TrialPruned:
        raise  # let Optuna handle it cleanly

    except Exception as e:
        err_str = str(e).lower()
        is_oom = "out of memory" in err_str or "cuda" in err_str or "resource exhausted" in err_str

        logging.error(f"Trial {trial.number} failed: {type(e).__name__}: {e}")

        # Cleanup
        for var_name in ("workflow_global", "global_posterior", "history"):
            try:
                del locals()[var_name]
            except KeyError:
                pass
        clear_gpu_memory()

        if is_oom:
            raise optuna.TrialPruned(f"OOM: {e}")
        else:
            # Non-OOM failures (e.g. bad hyperparam combos) get pruned too,
            # so the study continues rather than crashing entirely.
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
        cfg = compose(config_name="train_config")
    
    base_dir =  '/export/data/vgiusepp/diffusion_experiments_test_new/diffusion-experiments/case_study5/project_stream/data/'
    data_dir = 'streams/data_streamax_new/'
    N_simulations = 100_000


    train_data_path = os.path.join(base_dir, data_dir, f"training_data_local_{N_simulations}.npz")
    training_data = dict(np.load(train_data_path, allow_pickle=True))
    training_data = {k: training_data[k][:90_000] for k in training_data.keys()}


    augmentations_class = AugmentationsClass(cfg)
    augmentations_class.key = jax.random.PRNGKey(42)
    augmentations = []

    
    if "cut_to_300_particles" in cfg.augmentations:
        augmentations.append(augmentations_class.cut_to_300_particles)
    if "remove_los_velocity" in cfg.augmentations: #remove this if you want to train with vlos and errors
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
    if "observational_window_random" in cfg.augmentations:
        augmentations.append(augmentations_class.observational_window_random)
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


    test_data = dict(np.load(train_data_path, allow_pickle=True))
    test_data = {k: test_data[k][-10_000:] for k in test_data.keys()}
    for aug in augmentations:
        print(f"Applying augmentation {aug.__name__} to test data...")
        test_data = aug(test_data)
    for k in test_data.keys():
        test_data[k] = np.array(test_data[k])

    augmentations_class.key = jax.random.PRNGKey(42)
        
    print("Loaded config:", cfg)
    study_name = 'study_DiffusionModel'  # Unique identifier of the study.
    storage_name = JournalStorage(JournalFileStorage("./data/hyperparameter_tuning/streamax_new/100k/optuna_diffusionmodel_gala_cutNGC3201.log"))
    study = optuna.create_study(study_name=study_name, storage=storage_name, directions=['minimize', 'minimize'], load_if_exists=True)
    study.optimize(
        lambda trial: objective(trial, cfg),
        callbacks=[MaxTrialsCallback(300, states=(TrialState.COMPLETE, TrialState.FAIL))],
    )