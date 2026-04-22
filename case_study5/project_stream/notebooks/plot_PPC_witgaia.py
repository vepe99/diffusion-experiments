from astropy.io import ascii
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import galstreams


data = np.load("../data/gaia_observed_streams.npz")
sim_data = data['sim_data_projected']
j = data['j']
name_to_plot = ['Pal5', 'NGC3201', 'M68']
posterior_sample = np.load('/export/data/vgiusepp/diffusion_experiments_test_new/diffusion-experiments/case_study5/project_stream/data/streams/data_multistream_gala_posterior_predictive_check/model_54_60k_1000epochs_local2_mode/ppc_10samples.npz')
posteriorpredictive_sample = posterior_sample['sim_data_projected'] 

print(f"Loaded sim_data shape: {sim_data.shape}")
print(f"Loaded j shape: {j.shape}")
attention_mask = data['attention_mask'].astype(bool)
fig = plt.figure(figsize=(7, 5))
for i in range(sim_data.shape[1]):
    plt.scatter(sim_data[0, i, attention_mask[i, 0, :], 0], sim_data[0, i, attention_mask[i, 0, :], 1], s=1, label=f"{name_to_plot[i]}")
    plt.scatter(posteriorpredictive_sample[0, i,:,  0], posteriorpredictive_sample[0, i, :, 1], s=1, label=f"{name_to_plot[i]} Posterior Predictive", alpha=0.5)
plt.legend()
plt.xlabel('$\\alpha$ ', fontsize=20)
plt.ylabel('$\\delta$ ', fontsize=20)
plt.savefig('./PPC_1sample.pdf')