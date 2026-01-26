from autocvd import autocvd
autocvd(num_gpus = 1)

import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

backend = "torch"
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = backend

import numpy as np
import bayesflow as bf
from bayesflow.diagnostics import metrics as bf_metrics


import gc
import torch




import logging
logging.getLogger('bayesflow').setLevel(logging.DEBUG)

from case_study5.settings import EPOCHS, BATCH_SIZE, N_TRAINING_BATCHES, N_TRIALS, N_SUBJECTS, N_TEST, N_SAMPLES, BASE, METHOD, STEPS, MAX_STEP, N_SAMPLES_LOCAL, sample_in_batches

def prior_global_score(x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        m_nfw = x["m_nfw"]
        r_s = x["r_s"]
        q1 = x["q1"]
        q2 = x["q2"]

        score = {
            "m_nfw": np.zeros_like(m_nfw),
            "r_s": np.zeros_like(r_s),
            "q1": np.zeros_like(q1),
            "q2": np.zeros_like(q2),
        }

        return score


def objective(trial):
    
    summary_dim = trial.suggest_int("SetTransformer_summary_dim", 8, 64)
    embed_dims = trial.suggest_int("SetTransformer_embed_dims", 8, 64)
    num_heads = trial.suggest_int("SetTransformer_num_heads", 2, 8)
    # mlp_depths = trial.suggest_int("SetTransformer_mlp_depths", 2, 4) 
    # mlp_widths = trial.suggest_int("SetTransformer_mlp_widths", 32, 256)

    inference_mlp_depth = trial.suggest_int("inference_mlp_depth", 2, 6)
    inference_mlp_width = trial.suggest_int("inference_mlp_width", 64, 512)
    time_embedding_dim = trial.suggest_int("inference_time_embedding_dim", 16, 64, step=2)


    workflow_global = bf.BasicWorkflow(
        adapter=adapter,
        summary_network=bf.networks.SetTransformer(summary_dim=summary_dim, 
                                                   embed_dims=(embed_dims, embed_dims), 
                                                   num_heads=(num_heads, num_heads),
                                                #    mlp_depths=(mlp_depths, mlp_depths),
                                                #    mlp_widths=(mlp_widths, mlp_widths),
                                                   dropout=0.1),
        inference_network=bf.networks.CompositionalDiffusionModel(
                                                    subnet_kwargs={
                                                    "widths": [inference_mlp_width] * inference_mlp_depth,
                                                    "activation": "mish",               #DEFAULT
                                                    "kernel_initializer": "he_normal",  #DEFAULT
                                                    "residual": True,                   #DEFAULT
                                                    "dropout": 0.05,                    #DEFAULT
                                                    "spectral_normalization": False,    #DEFAULT
                                                    "time_embedding_dim": time_embedding_dim,
                                                    "merge": "concat",                  #DEFAULT
                                                    "norm": "layer",                    #DEFAULT
                                                    }),
        standardize=["inference_variables", "summary_variables"]
        )
    
    history = workflow_global.fit_offline(
        training_data,
        epochs=50,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    # # Split test data into 4 quarters
    n_test_total = test_data['sim_data'].shape[0]
    quarter_size = n_test_total // 8

    global_posteriors = []

    for i in range(8):
        start_idx = i * quarter_size
        # For the last quarter, include any remaining samples
        end_idx = (i + 1) * quarter_size if i < 7 else n_test_total
        
        test_data_batch = {
            'sim_data': test_data['sim_data'][start_idx:end_idx],
            'j': test_data['j'][start_idx:end_idx]
        }
        posterior_batch = workflow_global.compositional_sample(
            num_samples=N_SAMPLES,
            conditions={'sim_data': test_data_batch['sim_data'], 
                        "j": test_data_batch["j"] },
            compute_prior_score=prior_global_score,
            compositional_bridge_d1=1/N_SUBJECTS,
            mini_batch_size=2,
            method=METHOD,
            steps=STEPS,
            max_steps=MAX_STEP
        )
        global_posteriors.append(posterior_batch)
        
        # Free GPU memory after each batch
        del test_data_batch
        gc.collect()
        torch.cuda.empty_cache()

    # Concatenate posterior samples from all quarters
    global_posterior = {
        k: np.concatenate([np.array(gp[k]) for gp in global_posteriors], axis=0)
        for k in global_posteriors[0].keys()
    }
    ps = global_posterior.copy()
    ps['m_nfw'] = (ps['m_nfw'] * training_data_mean_and_std['std_m_nfw'] + training_data_mean_and_std['mean_m_nfw'])
    ps['r_s'] = (ps['r_s'] * training_data_mean_and_std['std_r_s'] + training_data_mean_and_std['mean_r_s'])
    ps['q1'] = (ps['q1'] * training_data_mean_and_std['std_q1'] + training_data_mean_and_std['mean_q1'])
    ps['q2'] = (ps['q2'] * training_data_mean_and_std['std_q2'] + training_data_mean_and_std['mean_q2'])

    root_mean_squared_error = bf_metrics.root_mean_squared_error(
            estimates=ps,
            targets=test_data,
            variable_keys=param_names_global,
            variable_names=param_names_global,
        )
    
    calibration_errors = bf_metrics.calibration_error(
            estimates=ps,
            targets=test_data,
            variable_keys=param_names_global,
            variable_names=param_names_global,
        )
    
    average_rms = root_mean_squared_error['values'].mean()
    average_calibration = calibration_errors['values'].mean()
    
    return average_rms, average_calibration 



if __name__ == "__main__":

    

    param_names_global = ['m_nfw', 'r_s', 'q1', 'q2']
    pretty_param_names_global = [r'$M_{200}$', r'$r_s$', r'$q_1$', r'$q_2$']

    inference_conditions_name = ['j']

    param_names_local = ['prog_mass', 't_end',
                        'x_c', 'y_c', 'z_c',
                        'v_xc', 'v_yc', 'v_zc']
    pretty_param_names_local = [r'$M_{prog}$', r'$t_{end}$',
                                r'$x^c$', r'$y^c$', r'$z^c$',
                                r'$v^c_x$', r'$v^c_y$', r'$v^c_z$']

    training_data_raw = dict(np.load('./case_study5/projection_training_set_odisseo_triaxial.npz', allow_pickle=True))
    training_data_raw['m_nfw'] = (training_data_raw['m_nfw']  - training_data_raw['mean_m_nfw'])/ training_data_raw['std_m_nfw']
    training_data_raw['r_s'] = (training_data_raw['r_s']  - training_data_raw['mean_r_s'])/ training_data_raw['std_r_s']
    training_data_raw['q1'] = (training_data_raw['q1']  - training_data_raw['mean_q1'])/ training_data_raw['std_q1']
    training_data_raw['q2'] = (training_data_raw['q2']  - training_data_raw['mean_q2'])/ training_data_raw['std_q2']
    training_data_mean_and_std = {k: v for k, v in training_data_raw.items() if k.startswith('mean_') or k.startswith('std_')}
    # Filter out mean/std arrays from training data
    training_data = {k: v for k, v in training_data_raw.items() if not k.startswith('mean_') and not k.startswith('std_')}  
    # Find and remove samples with NaN values
    nan_mask = np.isnan(training_data['sim_data']).any(axis=(1, 2))
    valid_mask = ~nan_mask
    print(f"Removing {nan_mask.sum()} samples with NaN values")
    for key in training_data:
        if hasattr(training_data[key], 'shape') and len(training_data[key]) == len(nan_mask):
            training_data[key] = training_data[key][valid_mask]
    # Take only half of the training set
    n_total = len(training_data['sim_data'])
    n_half = n_total // 2
    print(f"Using {n_half} samples out of {n_total} (50% of training data)")
    
    # Option 1: First half
    for key in training_data:
        if hasattr(training_data[key], 'shape') and len(training_data[key]) == n_total:
            training_data[key] = training_data[key][:n_half]
            
    del training_data_raw
    gc.collect()
    torch.cuda.empty_cache()
    
    test_data_npz = dict(np.load('./case_study5/projection_test_set_multistream_odisseo_triaxial.npz', allow_pickle=True))
    test_data = {k: np.expand_dims(np.array(test_data_npz[k]), axis=-1) for k in test_data_npz.keys() if k not in ['sim_data']}
    test_data['sim_data'] = test_data_npz['sim_data']
    print('Shape after expand_dims')
    print({k: test_data[k].shape for k in test_data.keys()})
            
    adapter = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .concatenate(param_names_global, into="inference_variables")
        .rename("j", "inference_conditions")
        .rename("sim_data", "summary_variables")
    )


    study_name = 'study_CompositionalDiffusionModel'  # Unique identifier of the study.
    storage_name = 'sqlite:///study_CompositionalDiffusionModel.db'
    study = optuna.create_study(study_name=study_name, storage=storage_name,directions=['minimize', 'minimize'], load_if_exists=True)
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    study.optimize(objective, callbacks=[MaxTrialsCallback(50, states=(TrialState.COMPLETE, TrialState.FAIL))],)
