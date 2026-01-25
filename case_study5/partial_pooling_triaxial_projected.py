from autocvd import autocvd
autocvd(num_gpus = 1)

evaluate = True
global_evalute = True
local_evaluate = True
if evaluate:
    backend = "torch"
else:
    backend = "jax"

import os
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = backend
else:
    print(f"Using '{os.environ['KERAS_BACKEND']}' backend")

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import median_abs_deviation

import keras
print("keras version:", keras.__version__)
import bayesflow as bf

from bayesflow.diagnostics.metrics import root_mean_squared_error as nrmse
from bayesflow.diagnostics.metrics import calibration_error as ece

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

param_names_global = ['m_nfw', 'r_s', 'q1', 'q2']
pretty_param_names_global = [r'$M_{200}$', r'$r_s$', r'$q_1$', r'$q_2$']

inference_conditions_name = ['j']

param_names_local = ['prog_mass', 't_end',
                     'x_c', 'y_c', 'z_c',
                     'v_xc', 'v_yc', 'v_zc']
pretty_param_names_local = [r'$M_{prog}$', r'$t_{end}$',
                            r'$x^c$', r'$y^c$', r'$z^c$',
                            r'$v^c_x$', r'$v^c_y$', r'$v^c_z$']

training_data = dict(np.load('./case_study5/projection_training_set_odisseo_triaxial.npz', allow_pickle=True))
print(np.isnan(training_data['sim_data']).sum())
training_data['sim_data'] = training_data['sim_data'][~np.isnan(training_data['sim_data'])]
print(np.isnan(training_data['sim_data']).sum())


adapter = (
    bf.adapters.Adapter()
    .to_array()
    .convert_dtype("float64", "float32")
    .concatenate(param_names_global, into="inference_variables")
    .rename("j", "inference_conditions")
    .rename("sim_data", "summary_variables")
)

del training_data  # free up memory

workflow_global = bf.BasicWorkflow(
    adapter=adapter,
    summary_network=bf.networks.SetTransformer(summary_dim=32, dropout=0.1),
    inference_network=bf.networks.CompositionalDiffusionModel(),
    standardize=["inference_variables", "summary_variables"]
)


model_path = BASE / 'models' / 'partial_pooling_global_triaxial_projected.keras'
if not os.path.exists(model_path):
    os.makedirs(name= BASE / 'models', exist_ok=True)
    training_data_raw = dict(np.load('./case_study5/projection_training_set_odisseo_triaxial.npz', allow_pickle=True))
    training_data_raw['m_nfw'] = (training_data_raw['m_nfw']  - training_data_raw['mean_m_nfw'])/ training_data_raw['std_m_nfw']
    training_data_raw['r_s'] = (training_data_raw['r_s']  - training_data_raw['mean_r_s'])/ training_data_raw['std_r_s']
    training_data_raw['q1'] = (training_data_raw['q1']  - training_data_raw['mean_q1'])/ training_data_raw['std_q1']
    training_data_raw['q2'] = (training_data_raw['q2']  - training_data_raw['mean_q2'])/ training_data_raw['std_q2']
    # Filter out mean/std arrays from training data
    training_data = {k: v for k, v in training_data_raw.items() if not k.startswith('mean_') and not k.startswith('std_')}  
    # Find and remove samples with NaN values
    nan_mask = np.isnan(training_data['sim_data']).any(axis=(1, 2))
    valid_mask = ~nan_mask
    print(f"Removing {nan_mask.sum()} samples with NaN values")
    
    for key in training_data:
        if hasattr(training_data[key], 'shape') and len(training_data[key]) == len(nan_mask):
            training_data[key] = training_data[key][valid_mask]
            
    # Debug: print shapes of training data
    print("Training data shapes:")
    for k, v in training_data.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: type = {type(v)}")
    
    history = workflow_global.fit_offline(
        training_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    workflow_global.approximator.save(model_path)
    print("Model trained")
else:
    workflow_global.approximator = keras.models.load_model(model_path)

if evaluate:
    test_data_npz = dict(np.load('./case_study5/projection_test_set_multistream_odisseo_triaxial.npz', allow_pickle=True))


    test_data = {k: np.expand_dims(np.array(test_data_npz[k]), axis=-1) for k in test_data_npz.keys() if k not in ['sim_data']}
    test_data['sim_data'] = test_data_npz['sim_data']
    print('Shape after expand_dims')
    print({k: test_data[k].shape for k in test_data.keys()})

if evaluate and global_evalute:

    logging.info("Starting Partial-Pooling (global) inference...")

    # # Split test data into 4 quarters
    n_test_total = test_data['sim_data'].shape[0]
    quarter_size = n_test_total // 8

    global_posteriors = []
    import gc
    import torch

    for i in range(8):
        start_idx = i * quarter_size
        # For the last quarter, include any remaining samples
        end_idx = (i + 1) * quarter_size if i < 7 else n_test_total
        
        test_data_batch = {
            'sim_data': test_data['sim_data'][start_idx:end_idx],
            'j': test_data['j'][start_idx:end_idx]
        }
        
        logging.info(f"Processing quarter {i+1}/8 ({end_idx - start_idx} samples)...")
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

    logging.info("Concatenated posterior samples")

    # Denormalization
    training_data_raw = dict(np.load('./case_study5/projection_training_set_odisseo_triaxial.npz', allow_pickle=True))
    ps = global_posterior.copy()
    ps['m_nfw'] = (ps['m_nfw'] * training_data_raw['std_m_nfw'] + training_data_raw['mean_m_nfw'])
    ps['r_s'] = (ps['r_s'] * training_data_raw['std_r_s'] + training_data_raw['mean_r_s'])
    ps['q1'] = (ps['q1'] * training_data_raw['std_q1'] + training_data_raw['mean_q1'])
    ps['q2'] = (ps['q2'] * training_data_raw['std_q2'] + training_data_raw['mean_q2'])
    del training_data_raw



    ###############
    # PLOTS GLOBAL#
    ###############

    #true vs predicted recovery plots
    fig = bf.diagnostics.recovery(
        estimates=ps,
        targets=test_data,
        variable_names=pretty_param_names_global
    )
    os.makedirs(name= BASE / 'plots', exist_ok=True)
    fig.savefig(BASE / "plots" / "partial_pooling_global_recovery_triaxial_projected.png")
    plt.show()

    #corner plot for dataset_id = 0
    dataset_id = 0
    g = bf.diagnostics.plots.pairs_posterior(
        estimates=ps,
        targets=test_data,
        dataset_id=dataset_id,
        variable_names=pretty_param_names_global,
    )
    fig = g
    fig.savefig(BASE / "plots" / f"partial_pooling_global_corner_datasetid_{dataset_id}_triaxial_projected.png")
    plt.show()

    # calibration plot
    fig = bf.diagnostics.calibration_ecdf(
        estimates=ps,
        targets=test_data,
        difference=True,
        variable_names=pretty_param_names_global
    )
    fig.savefig(BASE / "plots" / "partial_pooling_global_calibration_triaxial_projected.png")
    plt.show()

    # histograms 
    f = bf.diagnostics.plots.calibration_histogram(
        estimates=ps, 
        targets=test_data,
        variable_names=pretty_param_names_global
    )
    f.savefig(BASE / "plots" / "partial_pooling_global_calibration_histogram_triaxial_projected.png")
    plt.show()


    # z_score contraction
    f = bf.diagnostics.plots.z_score_contraction(
        estimates=ps, 
        targets=test_data,
        variable_names=pretty_param_names_global
    )
    f.savefig(BASE / "plots" / "partial_pooling_global_zscore_contraction_triaxial_projected.png")
    plt.show()

    print('finish global model test')

###############
# local model # 
###############
inference_conditions_names = param_names_global + ["j"]  # Use + instead of .append()
training_data_raw = dict(np.load('./case_study5/projection_training_set_odisseo_triaxial.npz', allow_pickle=True))

adapter_subjects = (
    bf.adapters.Adapter()
    .to_array()
    .convert_dtype("float64", "float32")
    .concatenate(param_names_local, into="inference_variables")
    .concatenate(inference_conditions_names, into="inference_conditions")
    .rename("sim_data", "summary_variables")
)

workflow_local = bf.BasicWorkflow(
    adapter=adapter_subjects,
    summary_network=bf.networks.SetTransformer(summary_dim=32, dropout=0.1),
    inference_network=bf.networks.DiffusionModel(),
    standardize=["inference_variables", "summary_variables"]
)
model_path_local = BASE / 'models' / 'partial_pooling_local_triaxial_projected.keras'
if not os.path.exists(model_path_local):
    os.makedirs(name= BASE / 'models', exist_ok=True)
    training_data_raw = dict(np.load('./case_study5/partial_pooling_local_triaxial_projected.npz', allow_pickle=True))
    # Filter out mean/std arrays from training data
    training_data = {k: v for k, v in training_data_raw.items() if not k.startswith('mean_') and not k.startswith('std_')}  
    # Find and remove samples with NaN values
    nan_mask = np.isnan(training_data['sim_data']).any(axis=(1, 2))
    valid_mask = ~nan_mask
    print(f"Removing {nan_mask.sum()} samples with NaN values")
    for key in training_data:
        if hasattr(training_data[key], 'shape') and len(training_data[key]) == len(nan_mask):
            training_data[key] = training_data[key][valid_mask]

    # Debug: print shapes of training data
    print("Training data shapes:")
    for k, v in training_data.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: type = {type(v)}")
    
    history = workflow_local.fit_offline(
        training_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    workflow_local.approximator.save(model_path_local)
else:
    workflow_local.approximator = keras.models.load_model(model_path_local)

if evaluate and local_evaluate:
    # we do not need anymore the shape of (N_TEST, N_STREMS, N_PARTICLES, 6), we need to reshape it similar to training set
    logging.info("Starting Partial-Pooling (local) inference...")

    # test_data['sim_data'] shape: (N_TEST, N_SUBJECTS, N_PARTICLES, 6)
    # We need to flatten (N_TEST, N_SUBJECTS) -> (N_TEST * N_SUBJECTS,)
    # Then repeat for each posterior sample from global model

    N_PARTICLES = test_data['sim_data'].shape[2]
    N_FEATURES = test_data['sim_data'].shape[3]


    # Reshape sim_data: (N_TEST, N_SUBJECTS, N_PARTICLES, 6) -> (N_TEST * N_SUBJECTS, N_PARTICLES, 6)
    sim_data_flat = test_data['sim_data'].reshape(N_TEST * N_SUBJECTS, N_PARTICLES, N_FEATURES)

    # Repeat for each posterior sample
    # (N_TEST * N_SUBJECTS, N_PARTICLES, 6) -> (N_TEST * N_SUBJECTS * N_SAMPLES, N_PARTICLES, 6)
    test_data_local = np.repeat(
        sim_data_flat[:, None, :, :],  # (N_TEST * N_SUBJECTS, 1, N_PARTICLES, 6)
        N_SAMPLES,
        axis=1
    ).reshape(N_TEST * N_SUBJECTS * N_SAMPLES, N_PARTICLES, N_FEATURES)

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
        arr = np.repeat(arr[:, None, :, ...], N_SUBJECTS, axis=1)
        
        # Flatten: (N_TEST * N_SUBJECTS * N_SAMPLES, ...)
        return arr.reshape(N_TEST * N_SUBJECTS * N_SAMPLES, *arr.shape[3:])

    def expand_local_test_param(arr):
        """
        Expand local test parameters for local inference.
        
        arr: shape (N_TEST, N_SUBJECTS, 1) or (N_TEST, N_SUBJECTS)
        Returns: shape (N_TEST * N_SUBJECTS * N_SAMPLES, 1)
        
        Each (test, subject) pair gets repeated N_SAMPLES times.
        """
        arr = np.asarray(arr)
        
        # Ensure 3D: (N_TEST, N_SUBJECTS, 1)
        if arr.ndim == 2:
            arr = arr[..., None]
        
        # Repeat for samples: (N_TEST, N_SUBJECTS, N_SAMPLES, 1)
        arr = np.repeat(arr[:, :, None, :], N_SAMPLES, axis=2)
        
        # Flatten: (N_TEST * N_SUBJECTS * N_SAMPLES, 1)
        return arr.reshape(N_TEST * N_SUBJECTS * N_SAMPLES, -1)


    conditions = {
        "sim_data": test_data_local,
        "m_nfw": expand_global_posterior(global_posterior["m_nfw"] ),
        "r_s": expand_global_posterior(global_posterior["r_s"]),
        "q1": expand_global_posterior(global_posterior["q1"]),
        "q2": expand_global_posterior(global_posterior["q2"]),
        "j": expand_local_test_param(test_data["j"]),
    }

    # Debug: print shapes
    print("Conditions shapes for local inference:")
    for k, v in conditions.items():
        print(f"  {k}: {v.shape}")

    #sample the local posterior in batches
    samples_flat = sample_in_batches(
        workflow=workflow_local,
        data=conditions,
        num_samples=N_SAMPLES_LOCAL,
        batch_size=BATCH_SIZE,
        sampler_settings=dict(method="tsit5", steps=STEPS, max_steps=MAX_STEP)
    )

    # Reshape samples back to (N_TEST * N_SUBJECTS, N_SAMPLES, ...)
    samples = {}
    for k in samples_flat.keys():
        arr = samples_flat[k][:, 0]  # only one sample per condition
        # arr shape: (N_TEST * N_SUBJECTS * N_SAMPLES, ...)
        arr = arr.reshape(N_TEST * N_SUBJECTS, N_SAMPLES_LOCAL, *arr.shape[1:])
        samples[k] = arr

    # Prepare test targets for local parameters
    # test_data local params have shape (N_TEST, N_SUBJECTS, 1)
    # We need shape (N_TEST * N_SUBJECTS, 1) for recovery plots
    test_params_local = {}
    for p in param_names_local:
        # test_data[p] shape: (N_TEST, N_SUBJECTS, 1) or similar
        arr = test_data[p]
        if arr.ndim == 3:
            # (N_TEST, N_SUBJECTS, 1) -> (N_TEST * N_SUBJECTS, 1)
            test_params_local[p] = arr.reshape(N_TEST * N_SUBJECTS, -1)
        elif arr.ndim == 2:
            # (N_TEST, N_SUBJECTS) -> (N_TEST * N_SUBJECTS, 1)
            test_params_local[p] = arr.reshape(N_TEST * N_SUBJECTS, 1)
        else:
            test_params_local[p] = arr.reshape(-1, 1)

    # Debug: print shapes
    print("Test params local shapes:")
    for k, v in test_params_local.items():
        print(f"  {k}: {v.shape}")

    print("Samples shapes:")
    for k, v in samples.items():
        print(f"  {k}: {v.shape}")

    samples['prog_mass'] = samples['prog_mass'] * training_data_raw['std_prog_mass'] + training_data_raw['mean_prog_mass']
    samples['t_end'] = samples['t_end'] * training_data_raw['std_t_end'] + training_data_raw['mean_t_end']
    samples['x_c'] = samples['x_c'] * training_data_raw['std_x_c'] + training_data_raw['mean_x_c']
    samples['y_c'] = samples['y_c'] * training_data_raw['std_y_c'] + training_data_raw['mean_y_c']
    samples['z_c'] = samples['z_c'] * training_data_raw['std_z_c'] + training_data_raw['mean_z_c']
    samples['v_xc'] = samples['v_xc'] * training_data_raw['std_v_xc'] + training_data_raw['mean_v_xc']
    samples['v_yc'] = samples['v_yc'] * training_data_raw['std_v_yc'] + training_data_raw['mean_v_yc']
    samples['v_zc'] = samples['v_zc'] * training_data_raw['std_v_zc'] + training_data_raw['mean_v_zc']

    fig = bf.diagnostics.recovery(
        estimates=samples,
        targets=test_params_local,
        variable_names=pretty_param_names_local,
    )
    fig.savefig(BASE / "plots" / "partial_pooling_local_recovery_triaxial_projected.png")
    plt.show()

    fig = bf.diagnostics.calibration_ecdf(
        estimates=samples,
        targets=test_params_local,
        variable_names=pretty_param_names_local,
    )
    fig.savefig(BASE / "plots" / "partial_pooling_local_calibration_triaxial_projected.png")
    plt.show()

