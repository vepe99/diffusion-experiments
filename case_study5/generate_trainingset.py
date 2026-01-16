from autocvd import autocvd
autocvd(num_gpus = 1)

import os
import numpy as np
import jax
from jax import jit, random
import jax.numpy as jnp
from functools import partial
import jax.lax as lax

from astropy import units as u

from stream_simulator_odisseo import sample_hierarchical_stream_priors, simulate_stream

n_training = 20_000
batch_training = 400
n_stars = 1000 


prior_samples = {}
for i in range(0, n_training):
    if i == 0:
        sample = sample_hierarchical_stream_priors(n_streams=1)
        for key in sample.keys():
            prior_samples[key] = [sample[key]]
    else:
        sample = sample_hierarchical_stream_priors(n_streams=1)
        for key in sample.keys():
            prior_samples[key].append(sample[key])

# Convert to arrays and expand dims for all parameters except sim_data
prior_samples_arrays = {}
for key in prior_samples.keys():
    arr = np.array(prior_samples[key])
    # Add trailing dimension: (n_training,) -> (n_training, 1)
    if arr.ndim == 1:
        prior_samples_arrays[key] = np.expand_dims(arr, axis=-1)
    else:
        prior_samples_arrays[key] = arr

# Debug: print shapes
print("Prior samples shapes (after expand_dims):")
for key, arr in prior_samples_arrays.items():
    print(f"  {key}: {arr.shape}")

def simulate_one(p):
    return simulate_stream(
        prog_mass=p["prog_mass"],
        t_end=p["t_end"],
        x_c=p["x_c"],
        y_c=p["y_c"],
        z_c=p["z_c"],
        v_xc=p["v_xc"],
        v_yc=p["v_yc"],
        v_zc=p["v_zc"],
        m_nfw=p["m_nfw"],
        r_s=p["r_s"],
        gamma=p["gamma"],
        j=p["j"],
        n_streams=1,
        n_stars=n_stars,
        key=p["key"],
    )["sim_data"]

# For simulation, use the original (non-expanded) arrays
params_for_sim = {k: jnp.asarray(prior_samples[k]) for k in prior_samples.keys()}
N = len(prior_samples["prog_mass"])
keys = random.split(random.PRNGKey(0), N)
params_with_keys = dict(**params_for_sim, key=keys)

batched_map = partial(
    lax.map,
    simulate_one,
    batch_size=batch_training,
)

sim_data = batched_map(params_with_keys)
print("sim_data shape:", sim_data.shape)  # (n_training, n_stars, 6)

# Save with expanded parameter shapes
np.savez('./case_study5/training_set_odisseo.npz', 
         sim_data=sim_data,
         **prior_samples_arrays)

# Verify saved shapes
print("\nFinal saved shapes:")
print(f"  sim_data: {sim_data.shape}")
for key, arr in prior_samples_arrays.items():
    print(f"  {key}: {arr.shape}")

# add mean and std computation
data = np.load('./case_study5/training_set_odisseo.npz', allow_pickle=True)
sim_data = data['sim_data']
mean_sim_data = np.mean(sim_data, axis=(0, 1))  
std_sim_data = np.std(sim_data, axis=(0, 1))   

m_nfw = data['m_nfw']
mean_m_nfw = np.mean(m_nfw, axis=0)
std_m_nfw = np.std(m_nfw, axis=0)

r_s = data['r_s']
mean_r_s = np.mean(r_s, axis=0)
std_r_s = np.std(r_s, axis=0)

prog_mass = data['prog_mass']
mean_prog_mass = np.mean(prog_mass, axis=0)
std_prog_mass = np.std(prog_mass, axis=0)

t_end = data['t_end']
mean_t_end = np.mean(t_end, axis=0)
std_t_end = np.std(t_end, axis=0)

x_c = data['x_c']
mean_x_c = np.mean(x_c, axis=0)
std_x_c = np.std(x_c, axis=0)
y_c = data['y_c']
mean_y_c = np.mean(y_c, axis=0)
std_y_c = np.std(y_c, axis=0)
z_c = data['z_c']
mean_z_c = np.mean(z_c, axis=0)
std_z_c = np.std(z_c, axis=0)
v_xc = data['v_xc']
mean_v_xc = np.mean(v_xc, axis=0)
std_v_xc = np.std(v_xc, axis=0)
v_yc = data['v_yc']
mean_v_yc = np.mean(v_yc, axis=0)
std_v_yc = np.std(v_yc, axis=0)
v_zc = data['v_zc']
mean_v_zc = np.mean(v_zc, axis=0)
std_v_zc = np.std(v_zc, axis=0)

np.savez('./case_study5/training_set_odisseo.npz',
        j=data['j'],
        gamma=data['gamma'],
        sim_data = sim_data,
        mean_sim_data = mean_sim_data,
        std_sim_data = std_sim_data,
        m_nfw = m_nfw,
        mean_m_nfw = mean_m_nfw,
        std_m_nfw = std_m_nfw,
        r_s = r_s,
        mean_r_s = mean_r_s,
        std_r_s = std_r_s,
        prog_mass = prog_mass,
        mean_prog_mass = mean_prog_mass,
        std_prog_mass = std_prog_mass,
        t_end = t_end,
        mean_t_end = mean_t_end,
        std_t_end = std_t_end,
        x_c = x_c,
        mean_x_c = mean_x_c,
        std_x_c = std_x_c,
        y_c = y_c,
        mean_y_c = mean_y_c,
        std_y_c = std_y_c,
        z_c = z_c,
        mean_z_c = mean_z_c,
        std_z_c = std_z_c,
        v_xc = v_xc,
        mean_v_xc = mean_v_xc,
        std_v_xc = std_v_xc,
        v_yc = v_yc,
        mean_v_yc = mean_v_yc,
        std_v_yc = std_v_yc,
        v_zc = v_zc,
        mean_v_zc = mean_v_zc,
        std_v_zc = std_v_zc 
)