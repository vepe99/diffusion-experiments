from autocvd import autocvd
autocvd(num_gpus=1, interval=1)

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import special

if "KERAS_BACKEND" not in os.environ: 
    os.environ["KERAS_BACKEND"] = "jax"

import bayesflow as bf

paramater_global_pretty = ["$M_{NFW}$",
                           "$r_{NFW}$",
                           "$\\hat{x}_{NFW}$",
                           "$\\hat{y}_{NFW}$",
                           "$\\hat{z}_{NFW}$",
                           "$M_{MN}$",
                           "$r_{MN}$",
                           "$M_{bulge}$",
                           "$q_{NFW}$"]

params_global = ["m_Triaxial_rotated_halo",
                 "r_Triaxial_rotated_halo",
                 "dirx_Triaxial_rotated_halo",
                 "diry_Triaxial_rotated_halo",
                 "dirz_Triaxial_rotated_halo",
                 "m_disk_MW2014",
                 "r_disk_MW2014",
                 "m_bulge",
                 "$q_{NFW}$"]

base_dir = '/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/plots'
# posterior_sample_dir = ['plots_streamax_concatenation_500_hyper_nocomposition',
#                         'plots_streamax_concatenate_500_hyper2nd_nocomposition',
#                         'plots_streamax_concatenation_500_highestcalibration_nocomposition', ]
posterior_sample_dir = ['plots_streamax_concatenation_500_hyper',
                        'plots_streamax_concatenate_500_hyper2nd',
                        'plots_streamax_concatenation_500_highestcalibration', ]
posterior_path = [os.path.join(base_dir, p, 'global_posterior.npz') for p in posterior_sample_dir]

posteriors = [dict(np.load(p)) for p in posterior_path]
ensable_posterior = {}
for k in params_global:
    ensable_posterior[k] = np.concatenate(tuple([posteriors[i][k] for i in range(len(posteriors))]),axis=1)
for k in ensable_posterior:
    print(f'{k} shape: {ensable_posterior[k].shape}')

test_data = dict(np.load('/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/streams/data_multistream_streamax/simulation_multistream_1000.npz'))
q_min = 0.5
q_max = 1.5
r_test = np.sqrt(test_data['dirx_Triaxial_rotated_halo']**2 + test_data['diry_Triaxial_rotated_halo']**2 + test_data['dirz_Triaxial_rotated_halo']**2)
u_uniform_test = special.erf(r_test/np.sqrt(2)) - np.sqrt(2/np.pi)*r_test*np.exp(-(r_test**2)/2)
test_data['$q_{NFW}$'] = q_min + (q_max-q_min)*u_uniform_test
mask_posterior = test_data['dirz_Triaxial_rotated_halo'] < 0
test_data['dirz_Triaxial_rotated_halo'][mask_posterior] *= -1
test_data['dirx_Triaxial_rotated_halo'][mask_posterior] *= -1
test_data['diry_Triaxial_rotated_halo'][mask_posterior] *= -1

test_data_posterior = {}
for k in params_global:
    test_data_posterior[k] = test_data[k]

# for k in test_data_posterior:
#     test_data_posterior[k] = np.repeat(test_data_posterior[k], 3, axis=0).reshape(-1, 1)
#     print(f'{k} shape: {test_data_posterior[k].shape}')

os.makedirs('/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/plots/ensamble_plots', exist_ok=True)

fig = bf.diagnostics.calibration_ecdf(
            estimates=ensable_posterior,
            targets=test_data_posterior,
            difference=True,
            variable_names=paramater_global_pretty
        )
fig.savefig('/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/plots/ensamble_plots/ecdf_calibration_ensamble.pdf', dpi=300)
plt.show()
fig = bf.diagnostics.calibration_ecdf(
            estimates=ensable_posterior,
            targets=test_data_posterior,
            difference=False,
            variable_names=paramater_global_pretty
        )
fig.savefig('/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/plots/ensamble_plots/ecdf_calibration_ensamble_nodiff.pdf', dpi=300)
plt.show()
fig = bf.diagnostics.recovery(
            estimates=ensable_posterior,
            targets=test_data_posterior,
            variable_names=paramater_global_pretty
        )
fig.savefig('/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/plots/ensamble_plots/recovery_ensamble.pdf', dpi=300)