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

from utils.utils_train_jax import AugmentationsClass
import jax

def clear_gpu_memory():
    """Helper function to aggressively clear GPU memory."""
    gc.collect()


def objective(trial, cfg, test_data):
    clear_gpu_memory()

    summary_dim = trial.suggest_int("SetTransformer_summary_dim", 16, 128)
    embed_dims = trial.suggest_int("SetTransformer_embed_dims", 16, 128)
    num_heads = trial.suggest_int("SetTransformer_num_heads", 1, 3)
    mlp_depths = trial.suggest_int("SetTransformer_mlp_depths", 2, 6)
    mlp_widths = trial.suggest_int("SetTransformer_mlp_widths", 16, 128)
    inference_mlp_depth = trial.suggest_int("inference_mlp_depth", 2, 6)
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
            inference_network=bf.networks.DiffusionModel(
                subnet_kwargs={
                    "widths": [inference_mlp_width] * inference_mlp_depth,
                    "time_embedding_dim": time_embedding_dim,
                }
            ),
            standardize=["inference_variables", "summary_variables", "inference_conditions"],
        )

        # --- Training with batch-size retry ---
        batch_size_training = 1000
        for attempt in range(2):
            try:
                history = workflow_global.fit_offline(
                    training_data,
                    epochs=1000,
                    batch_size=batch_size_training,
                    verbose=2,
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

        # --- Build conditions for ancestral sampling ---
        global_posterior = dict(np.load(
            '/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/plots/'
            'gala6D/new_hyper/model54_60k_1000epochs/333test/global_posterior.npz',
            allow_pickle=True
        ))

        n_test = 333
        n_streams = len(cfg.target_streams)

        sim_data_4d = test_data[cfg.sim_data].reshape(
            n_test, n_streams, *test_data[cfg.sim_data].shape[1:]
        )
        j_3d = test_data['j'].reshape(n_test, n_streams, 1)
        attn_mask_4d = test_data['attention_mask']

        conditions = {
            cfg.sim_data: sim_data_4d,
            'j': j_3d,
        }
        ancestral_conds = {
            param: global_posterior[param] for param in cfg.parameters_global
        }
        for param in cfg.parameters_global:
            conditions[param] = np.repeat(
                global_posterior[param][:, :1, :], n_streams, axis=1
            )

        def ancestral_sample_batched(workflow, conditions, ancestral_conds, attn_mask_4d, cfg, n_test_per_batch=5):
            n_test = 333
            all_samples = None
            for i in tqdm(range(0, n_test, n_test_per_batch)):
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

        # --- Sampling with n_test_per_batch retry ---
        # NOTE: was previously broken — batch_size_sampling was never defined before the except block
        n_test_per_batch = 30
        for attempt in range(2):
            try:
                local_posterior = ancestral_sample_batched(
                    workflow=workflow_global,
                    conditions=conditions,
                    ancestral_conds=ancestral_conds,
                    attn_mask_4d=attn_mask_4d,
                    cfg=cfg,
                    n_test_per_batch=n_test_per_batch,
                )
                break  # success
            except Exception as e:
                err_str = str(e).lower()
                is_oom = "out of memory" in err_str or "cuda" in err_str or "resource exhausted" in err_str
                if is_oom and attempt == 0:
                    logging.warning(f"Trial {trial.number} OOM during sampling, halving n_test_per_batch and retrying...")
                    n_test_per_batch //= 2
                    clear_gpu_memory()
                else:
                    raise

        # --- Reshape posteriors and compute metrics ---
        ps = local_posterior.copy()
        test_params_local = {}
        for p in param_names_local:
            arr = np.asarray(test_data[p])
            print(f'  {p} raw shape: {arr.shape}')
            if arr.ndim == 3:
                test_params_local[p] = arr.reshape(n_test * n_streams, -1)
            elif arr.ndim == 2 and arr.shape[0] == n_test and arr.shape[1] == n_streams:
                test_params_local[p] = arr.reshape(n_test * n_streams, 1)
            elif arr.ndim == 2:
                test_params_local[p] = arr
            elif arr.ndim == 1 and arr.shape[0] == n_test * n_streams:
                test_params_local[p] = arr.reshape(-1, 1)
            elif arr.ndim == 1 and arr.shape[0] == n_test:
                test_params_local[p] = np.repeat(arr, n_streams).reshape(-1, 1)
            else:
                raise ValueError(f'Unexpected shape for {p}: {arr.shape}')

        test_data_flat = test_params_local
        ps_flat = {k: np.asarray(v).reshape(n_test * n_streams, *v.shape[2:]) for k, v in ps.items()}

        root_mean_squared_error = bf_metrics.root_mean_squared_error(
            estimates=ps_flat,
            targets=test_data_flat,
            variable_keys=param_names_local,
            variable_names=param_names_local,
        )
        calibration_errors = bf_metrics.calibration_error(
            estimates=ps_flat,
            targets=test_data_flat,
            variable_keys=param_names_local,
            variable_names=param_names_local,
        )
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
    config_path = '/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/config/'

    with initialize_config_dir(version_base=None, config_dir=config_path):
        # 3. Compose: loads train_config.yaml, validated against TrainConfig schema
        cfg = compose(config_name="eval_config_local")
    
    base_dir =  '/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/'
    data_dir = 'streams/data_gala/'
    N_simulations = 300_000


    train_data_path = os.path.join(base_dir, data_dir, f"training_data_local_{N_simulations}.npz")
    training_data = dict(np.load(train_data_path, allow_pickle=True))
    training_data = {k: training_data[k][:60_000] for k in training_data.keys()}

    augmentations_class = AugmentationsClass(cfg)
    augmentations = []

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

    

    for aug in augmentations:
        training_data = aug(training_data)
    
    print('Training data shapes after augmentations:')
    for k, v in training_data.items():
        training_data[k] = np.array(training_data[k])
        print(f'  {k}: {v.shape}')
    

    # test_data = dict(np.load('.', allow_pickle=True))
    # test_data = {k: test_data[k][-1_000:] for k in test_data.keys()}
    test_data = dict(np.load(os.path.join(base_dir, 'streams/data_multistream_gala/simulation_multistream_333.npz')))
    test_data[cfg.sim_data] = test_data[cfg.sim_data].reshape(-1, test_data[cfg.sim_data].shape[-2], test_data[cfg.sim_data].shape[-1])
    test_data['j'] = test_data['j'].reshape(-1, 1)
    for aug in augmentations:
        test_data = aug(test_data)
        
    print("Loaded config:", cfg)
    study_name = 'study_DiffusionMode_local'  # Unique identifier of the study.
    storage_name = JournalStorage(JournalFileStorage("./data/hyperparameter_tuning/gala/local/optuna_diffusionmodel_galax_local_cutNGC3201.log"))
    study = optuna.create_study(study_name=study_name, storage=storage_name, directions=['minimize', 'minimize'], load_if_exists=True)
    study.optimize(
        lambda trial: objective(trial, cfg, test_data),
        callbacks=[MaxTrialsCallback(100, states=(TrialState.COMPLETE, TrialState.FAIL))],
    )