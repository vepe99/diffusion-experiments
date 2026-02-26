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

from utils_train import AugmentationsClass
import jax

def clear_gpu_memory():
    """Helper function to aggressively clear GPU memory."""
    gc.collect()


def objective(trial, cfg):
    # Clear memory at the start of each trial
    clear_gpu_memory()

    try:
        summary_dim = trial.suggest_int("SetTransformer_summary_dim", 20, 64)
        embed_dims = trial.suggest_int("SetTransformer_embed_dims", 48, 128)
        num_heads = trial.suggest_int("SetTransformer_num_heads", 1, 4)
        mlp_depths = trial.suggest_int("SetTransformer_mlp_depths", 2, 6) 
        mlp_widths = trial.suggest_int("SetTransformer_mlp_widths", 32, 256)

        inference_mlp_depth = trial.suggest_int("inference_mlp_depth", 5, 8)
        inference_mlp_width = trial.suggest_int("inference_mlp_width", 64, 512)
        time_embedding_dim = trial.suggest_int("inference_time_embedding_dim", 16, 64, step=2)

        param_names_global = list(cfg.parameters_global)
        sim_data = 'sim_data_projected'
        inference_conditions = 'j' #just 1
        
        adapter = (
            bf.adapters.Adapter()
            .to_array()
            .convert_dtype("float64", "float32")
            .concatenate(param_names_global, into="inference_variables")
            .rename(sim_data, "summary_variables")
            .rename(inference_conditions, "inference_conditions")
        )
        workflow_global = bf.BasicWorkflow(
            adapter=adapter,
            summary_network=bf.networks.SetTransformer(summary_dim=summary_dim, 
                                                       embed_dims=(embed_dims, embed_dims), 
                                                       num_heads=(num_heads, num_heads),
                                                       mlp_depths=(mlp_depths, mlp_depths),
                                                       mlp_widths=(mlp_widths, mlp_widths),
                                                       dropout=0.1),
            inference_network=bf.networks.CompositionalDiffusionModel(
                                                        subnet_kwargs={
                                                        "widths": [inference_mlp_width] * inference_mlp_depth,
                                                        "time_embedding_dim": time_embedding_dim,
                                                        }),
            standardize=["inference_variables", "summary_variables"]
            )
       
        
        batch_size_training = 512
        try:
            history = workflow_global.fit_offline(
                training_data,
                epochs=100,
                batch_size=batch_size_training,
                verbose=2,
            )
        except Exception as e:
            logging.error(f"Training failed with error: {e}")
            logging.error('Half batch for training and retrying...')
            batch_size_training = int(batch_size_training/2)
            history = workflow_global.fit_offline(
                training_data,
                epochs=100,
                batch_size=batch_size_training,
                verbose=2,
            )

        workflow_global.approximator.inference_network.integrate_kwargs.update({
            'method': "two_step_adaptive",
            'steps': "adaptive",
            "max_steps": 1000,
            })
        
        batch_size_sampling = 100
        try: 
            gloabl_posterior = workflow_global.sample(
                                num_samples=1000,
                                conditions={cfg.sim_data: test_data[cfg.sim_data], 
                                            "j": test_data["j"] },
                                batch_size=batch_size_sampling,
                                kwargs={'attention_mask': test_data['attention_mask']}
                            )
        except Exception as e:
            logging.error(f"Sampling failed with error: {e}")
            logging.error('Half batch for sampling and retrying...')
            batch_size_sampling = int(batch_size_sampling/2)
            gloabl_posterior = workflow_global.sample(
                                num_samples=1000,
                                conditions={cfg.sim_data: test_data[cfg.sim_data], 
                                            "j": test_data["j"] },
                                batch_size=batch_size_sampling,
                                kwargs={'attention_mask': test_data['attention_mask']}
                            )
            
        root_mean_squared_error = bf_metrics.root_mean_squared_error(
                estimates=gloabl_posterior,
                targets=test_data,
                variable_keys=param_names_global,
                variable_names=param_names_global,
            )
        
        calibration_errors = bf_metrics.calibration_error(
                estimates=gloabl_posterior,
                targets=test_data,
                variable_keys=param_names_global,
                variable_names=param_names_global,
            )
        average_rms = root_mean_squared_error['values'].mean()
        average_calibration = calibration_errors['values'].mean()
        return average_rms, average_calibration


    except Exception as e:
        # Check if it's a CUDA OOM error
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            logging.warning(f"Trial {trial.number} failed due to CUDA OOM: {e}")
            
            # Aggressive cleanup
            try:
                del workflow_global
            except NameError:
                pass
            try:
                del global_posteriors
            except NameError:
                pass
            clear_gpu_memory()
            
            # Raise TrialPruned to skip this trial and continue with the next
            raise optuna.TrialPruned(f"CUDA out of memory: {e}")
        else:
            # Re-raise if it's a different RuntimeError
            raise
    
    except Exception as e:
        # Catch any other unexpected errors
        logging.error(f"Trial {trial.number} failed with unexpected error: {e}")
        
        # Cleanup
        try:
            del workflow_global
        except NameError:
            pass
        try:
            del global_posteriors
        except NameError:
            pass
        clear_gpu_memory()
        
        # Optionally prune or re-raise
        raise optuna.TrialPruned(f"Unexpected error: {e}")
    

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from train_config import TrainConfig
from optuna.storages import JournalStorage, JournalFileStorage


if __name__ == "__main__":
    # Initialize Hydra config without the decorator
    cs = ConfigStore.instance()
    cs.store(name="train_config_schema", node=TrainConfig)

    # 2. Point to the directory containing train_config.yaml
    config_path = '/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/config/'

    with initialize_config_dir(version_base=None, config_dir=config_path):
        # 3. Compose: loads train_config.yaml, validated against TrainConfig schema
        cfg = compose(config_name="train_config")
    
    base_dir =  '/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/'
    data_dir = 'streams/data_streamax/'
    N_simulations = 1_000_000


    train_data_path = os.path.join(base_dir, data_dir, f"training_data_{N_simulations}.npz")
    training_data = dict(np.load(train_data_path, allow_pickle=True))
    training_data = {k: training_data[k][:100_000] for k in training_data.keys()}

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
    if "flip_dirz" in cfg.augmentations:
        augmentations.append(augmentations_class.flip_dirz)

    

    for aug in augmentations:
        training_data = aug(training_data)
    
    print('Training data shapes after augmentations:')
    for k, v in training_data.items():
        print(f'  {k}: {v.shape}')
    

    test_data_path = os.path.join(base_dir, 'streams/data_streamax/', f"validation_data_1000.npz")
    test_data = dict(np.load(test_data_path, allow_pickle=True))
    for aug in augmentations:
        test_data = aug(test_data)
        
    print("Loaded config:", cfg)
    study_name = 'study_DiffusionModel'  # Unique identifier of the study.
    storage_name = JournalStorage(JournalFileStorage("./data/hyperparameter_tuning/optuna_diffusionmodel.log"))
    study = optuna.create_study(study_name=study_name, storage=storage_name, directions=['minimize', 'minimize'], load_if_exists=True)
    study.optimize(
        lambda trial: objective(trial, cfg),
        callbacks=[MaxTrialsCallback(100, states=(TrialState.COMPLETE, TrialState.FAIL))],
    )