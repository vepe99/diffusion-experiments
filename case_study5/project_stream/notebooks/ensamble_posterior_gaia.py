import os
os.environ['CUDA_VISIBILE_DEVICES'] = ""
import numpy as np
import matplotlib.pyplot as plt

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

import bayesflow as bf

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────


param_names_global_pretty = [
    "$M_{NFW}$",
    "$r_{NFW}$",
    "$q_{NFW}$",
    "$\\rho_t$",
    "$h_t$",
    "$z_t$",
    "$\\rho_k$",
    "$h_k$",
    "$z_k$",
]

param_name_global = [
    "m_Triaxial_halo",
    "r_Triaxial_halo",
    "q2_Triaxial_halo",
    "rho_thin_disk",
    "hr_thin_disk",
    "hz_thin_disk",
    "rho_thick_disk",
    "hr_thick_disk",
    "hz_thick_disk",
]

# Posterior names to load (one per stream + global)
posterior_names = [
    'global_posterior.npz',
    'Pal5_posterior.npz',
    'NGC3201_posterior.npz',
    'M68_posterior.npz',
]

base_dir = '/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/plots/gala6D/'

posterior_sample_dirs = [
    # 'model31_60k_500epochs/',
    # 'model35_60k_500epochs/',
    # 'model40_60k_1000epochs/',
    f'new_hyper/modelDEFAULT_60k_1000epochs/',
    f'new_hyper/model54_60k_1000epochs/',
]


posterior_sample_dirs = [os.path.join(base_dir, d) for d in posterior_sample_dirs]

# ──────────────────────────────────────────────
# Load & ensemble posteriors
# ──────────────────────────────────────────────

ensemble_posteriors = {}

for post_name in posterior_names:
    print(f"\n--- Processing {post_name} ---")

    posterior_paths = [os.path.join(d, post_name) for d in posterior_sample_dirs]

    loaded_posteriors = [dict(np.load(p)) for p in posterior_paths]

    combined_posterior = {}
    for k in loaded_posteriors[0].keys():
        combined_posterior[k] = np.concatenate(
            [p[k] for p in loaded_posteriors], axis=1
        )

    ensemble_posteriors[post_name] = combined_posterior

    for k, v in combined_posterior.items():
        print(f'  {k} shape: {v.shape}')

# Flatten each posterior array to 1-D (draws only)
for post_name in ensemble_posteriors:
    for k in ensemble_posteriors[post_name]:
        ensemble_posteriors[post_name][k] = ensemble_posteriors[post_name][k].reshape(-1)

# ──────────────────────────────────────────────
# Load true / observed data
# ──────────────────────────────────────────────

true_data_path = (
    '/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/'
    'data/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz'
)
print(f'\nLoading true data from {true_data_path}')
true_data = dict(np.load(true_data_path, allow_pickle=True))
true_data = {
    k: true_data[k]
    for k in ["sim_data_projected", "j", "attention_mask", "magnitudes"]
}

# ──────────────────────────────────────────────
# Build DataFrames
# ──────────────────────────────────────────────

import pandas as pd

df_global   = pd.DataFrame(ensemble_posteriors['global_posterior.npz'])
df_Pal5     = pd.DataFrame(ensemble_posteriors['Pal5_posterior.npz'])
df_NGC3201  = pd.DataFrame(ensemble_posteriors['NGC3201_posterior.npz'])
df_M68      = pd.DataFrame(ensemble_posteriors['M68_posterior.npz'])

# ──────────────────────────────────────────────
# Corner plot with ChainConsumer
# ──────────────────────────────────────────────

from chainconsumer import Chain, ChainConsumer, ChainConfig

c = ChainConsumer()
c.add_chain(Chain(samples=df_global,  name="Global"))
c.add_chain(Chain(samples=df_Pal5,    name="Pal5"))
c.add_chain(Chain(samples=df_NGC3201, name="NGC3201"))
c.add_chain(Chain(samples=df_M68,     name="M68"))

c.set_override(ChainConfig(shade=False))
fig = c.plotter.plot()
fig.savefig('./global_cornerplot_gaia_ensambled.pdf')
print('Corner plot saved to global_cornerplot_gaia_ensambled.pdf')