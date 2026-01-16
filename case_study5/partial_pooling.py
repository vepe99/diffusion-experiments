from autocvd import autocvd
autocvd(num_gpus = 1)

import os
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"
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

from case_study5.settings import EPOCHS, BATCH_SIZE, N_TRAINING_BATCHES, N_TRIALS, N_SUBJECTS, N_TEST, N_SAMPLES, BASE, METHOD, STEPS, MAX_STEP

def prior_global_score(x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    m_nfw = x["m_nfw"]
    r_s = x["r_s"]

    score = {
        "m_nfw": np.zeros_like(m_nfw),
        "r_s": np.zeros_like(r_s),
    }

    return score

param_names_global = ['m_nfw', 'r_s',]
pretty_param_names_global = [r'$M_{200}$', r'$r_s$', ]

inference_conditions_name = ['j']

param_names_local = ['prog_mass', 't_end',
                     'x_c', 'y_c', 'z_c',
                     'v_xc', 'v_yc', 'v_zc']
pretty_params_names_local = [r'$M_{prog}$', r'$t_{end}$',
                            r'$x^c$', r'$y^c$', r'$z^c$',
                            r'$v^c_x$', r'$v^c_y$', r'$v^c_z$']

training_data = dict(np.load('./case_study5/training_set_odisseo.npz', allow_pickle=True))


adapter = (
    bf.adapters.Adapter()
    .to_array()
    .convert_dtype("float64", "float32")
    .standardize("sim_data",
                 mean=training_data['mean_sim_data'],
                 std=training_data['std_sim_data']) #this is not a problem
    # .standardize("m_nfw",
    #              mean=training_data['mean_m_nfw'],
    #              std=training_data['std_m_nfw'])
    # .standardize("r_s",
    #              mean=training_data['mean_r_s'],
    #              std=training_data['std_r_s'])
    .concatenate(param_names_global, into="inference_variables")
    .rename("j", "inference_conditions")
    .rename("sim_data", "summary_variables")
)

del training_data  # free up memory

workflow_global = bf.BasicWorkflow(
    adapter=adapter,
    summary_network=bf.networks.SetTransformer(summary_dim=16, dropout=0.1),
    inference_network=bf.networks.CompositionalDiffusionModel(),
)


model_path = BASE / 'models' / 'partial_pooling_global.keras'
if not os.path.exists(model_path):
    os.makedirs(name= BASE / 'models', exist_ok=True)
    training_data_raw = dict(np.load('./case_study5/training_set_odisseo.npz', allow_pickle=True))
    training_data_raw['m_nfw'] = (training_data_raw['m_nfw']  - training_data_raw['mean_m_nfw'])/ training_data_raw['std_m_nfw']
    training_data_raw['r_s'] = (training_data_raw['r_s']  - training_data_raw['mean_r_s'])/ training_data_raw['std_r_s']
    # Filter out mean/std arrays from training data
    training_data = {k: v for k, v in training_data_raw.items() if not k.startswith('mean_') and not k.startswith('std_')}  
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
else:
    workflow_global.approximator = keras.models.load_model(model_path)

training_data_raw = dict(np.load('./case_study5/training_set_odisseo.npz', allow_pickle=True))
test_data_npz = dict(np.load('./case_study5/test_set_multistream_odisseo.npz', allow_pickle=True))
test_data_npz['m_nfw'] = (test_data_npz['m_nfw']  - training_data_raw['mean_m_nfw'])/ training_data_raw['std_m_nfw']
test_data_npz['r_s'] = (test_data_npz['r_s']  - training_data_raw['mean_r_s'])/ training_data_raw['std_r_s']

test_data = {k: np.expand_dims(np.array(test_data_npz[k]), axis=-1) for k in test_data_npz.keys() if k not in ['sim_data']}
test_data['sim_data'] = test_data_npz['sim_data']
print('Shape after expand_dims')
print({k: test_data[k].shape for k in test_data.keys()})

logging.info("Starting Partial-Pooling (global) inference...")
global_posterior = workflow_global.compositional_sample(
    num_samples=N_SAMPLES,
    conditions={'sim_data': test_data['sim_data'], 
                "j": test_data["j"] },
    compute_prior_score=prior_global_score,
    compositional_bridge_d1=1/N_SUBJECTS,
    mini_batch_size=3,
    method=METHOD,
    steps=STEPS,
    max_steps=MAX_STEP
)

test_data['m_nfw'] = (test_data_npz['m_nfw'] *training_data_raw['std_m_nfw'] + training_data_raw['mean_m_nfw']) 
test_data['r_s'] = (test_data_npz['r_s'] *training_data_raw['std_r_s'] + training_data_raw['mean_r_s']) 

ps = global_posterior.copy()
ps['m_nfw'] = (ps['m_nfw'] *training_data_raw['std_m_nfw'] + training_data_raw['mean_m_nfw'])
ps['r_s'] = (ps['r_s'] *training_data_raw['std_r_s'] + training_data_raw['mean_r_s'])

del training_data_raw  # free up memory


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
fig.savefig(BASE / "plots" / "partial_pooling_global_recovery.png")
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
fig.savefig(BASE / "plots" / f"partial_pooling_global_corner_datasetid_{dataset_id}.png")
plt.show()

# calibration plot
fig = bf.diagnostics.calibration_ecdf(
    estimates=ps,
    targets=test_data,
    difference=True,
    variable_names=pretty_param_names_global
)
fig.savefig(BASE / "plots" / "partial_pooling_global_calibration.png")
plt.show()

# histograms 
f = bf.diagnostics.plots.calibration_histogram(
    estimates=ps, 
    targets=test_data,
    variable_names=pretty_param_names_global
)
f.savefig(BASE / "plots" / "partial_pooling_global_calibration_histogram.png")
plt.show()


# z_score contraction
f = bf.diagnostics.plots.z_score_contraction(
    estimates=ps, 
    targets=test_data,
    variable_names=pretty_param_names_global
)
f.savefig(BASE / "plots" / "partial_pooling_global_zscore_contraction.png")
plt.show()