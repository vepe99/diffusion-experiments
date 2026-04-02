import os
os.environ['CUDA_VISIBILE_DEVICES'] = ""
import numpy as np
import matplotlib.pyplot as plt

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"

import bayesflow as bf

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


param_names_gloabl_pretty = [
    "$M_{NFW}$",
    "$r_{NFW}$",
    "$q_{NFW}$",
    "$\\rho_t$",
    "$h_t$",
    "$z_t$",
    "$\\rho_k$",
    "$h_k$",
    "$z_k$"
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
    "hz_thick_disk"
]

N = 333
# add_nocomposition = 'nocomposition_'
add_nocomposition = ''
base_dir = '/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/plots/gala6D/'
posterior_sample_dir = [
    # f'model28_60k_500epochs/{add_nocomposition}_{N}test',
    # f'model40_60k_500epochs/{add_nocomposition}_{N}test',
    f'model31_60k_500epochs/{add_nocomposition}{N}test',
    f'model35_60k_500epochs/{add_nocomposition}{N}test',
]
if add_nocomposition != '':
     posterior_path = [os.path.join(base_dir, p, 'posterior.npz') for p in posterior_sample_dir]    
else:
    posterior_path = [os.path.join(base_dir, p, 'global_posterior.npz') for p in posterior_sample_dir]

posteriors = [dict(np.load(p)) for p in posterior_path]
ensable_posterior = {}
for k in posteriors[0].keys():
    ensable_posterior[k] = np.concatenate(
        tuple([posteriors[i][k] for i in range(len(posteriors))]), axis=1
    )
for k in ensable_posterior:
    print(f'{k} shape: {ensable_posterior[k].shape}')

test_data = dict(np.load(
    f'/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/streams/data_multistream_gala/simulation_multistream_{N}.npz'
))

if add_nocomposition != '':
    test_data_new = {}
    for k in param_name_global:
        test_data_new[k] = np.repeat(test_data[k], 3, axis=0).reshape(-1, 1)
else:
    test_data_new = test_data
print('Test data keys')
for k in test_data_new:
    print(f'{k} shape: {test_data_new[k].shape}')

from utils.utils_plot import calibration_ecdf
fig = calibration_ecdf(
    estimates=ensable_posterior,
    targets=test_data_new,
    difference=True,
    variable_names=param_names_gloabl_pretty,
    stacked = True,
    rank_ecdf_color=plt.cm.magma(np.linspace(0, 1, len(param_names_gloabl_pretty))),
)
for ax in fig.get_axes():
        ax.grid(False)

fig.savefig('./calibration_ensamble.pdf')
print('Calibration plot saved to calibration_ensamble.pdf')


print('hello')