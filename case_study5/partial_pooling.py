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

adapter = (
    bf.adapters.Adapter()
    .to_array()
    .convert_dtype("float64", "float32")
    .expand_dims("m_nfw", axis=-1)  # Ensure (batch,) -> (batch, 1)
    .expand_dims("r_s", axis=-1)    # Ensure (batch,) -> (batch, 1)
    .concatenate(param_names_global, into="inference_variables")
    .expand_dims("j", axis=-1)
    .rename("j", "inference_conditions")
    .rename("sim_data", "summary_variables")
)

workflow_global = bf.BasicWorkflow(
    adapter=adapter,
    summary_network=bf.networks.SetTransformer(summary_dim=16, dropout=0.1),
    inference_network=bf.networks.CompositionalDiffusionModel(),
)

model_path = BASE / 'models' / 'partial_pooling_global.keras'
if not os.path.exists(model_path):
    os.makedirs(name= BASE / 'models', exist_ok=True)

    training_data = dict(np.load('./case_study5/training_set_odisseo.npz', allow_pickle=True))
    training_data = {k: np.array(training_data[k]) for k in training_data.keys()}
    history = workflow_global.fit_offline(
        training_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    workflow_global.approximator.save(model_path)
else:
    workflow_global.approximator = keras.models.load_model(model_path)

test_data = dict(np.load('./case_study5/test_set_multistream_odisseo.npz', allow_pickle=True))
global_posterior = workflow_global.compositional_sample(
    num_samples=N_SAMPLES,
    conditions={'sim_data': test_data['sim_data'], 'j': test_data['j']},
    compute_prior_score=prior_global_score,
    compositional_bridge_d1=1/N_SUBJECTS,
    mini_batch_size=3,
    method=METHOD,
    steps=STEPS,
    max_steps=MAX_STEP
)

ps = global_posterior.copy()
fig = bf.diagnostics.recovery(
    estimates=ps,
    targets=test_data,
    variable_names=pretty_param_names_global
)
os.makedirs(name= BASE / 'plots', exist_ok=True)
fig.savefig(BASE / "plots" / "partial_pooling_global_recovery.png")
plt.show()
