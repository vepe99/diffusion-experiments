from autocvd import autocvd
autocvd(num_gpus = 1)


import os
# if "KERAS_BACKEND" not in os.environ:
#     os.environ["KERAS_BACKEND"] = "jax"
# else:
#     print(f"Using '{os.environ['KERAS_BACKEND']}' backend")
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# import bayesflow as bf


import numpy as np
import jax
from jax import jit, random
import jax.numpy as jnp
from functools import partial
import jax.lax as lax

from astropy import units as u

# from case_study5.stream_simulator_gala import sample_hierarchical_stream_priors, simulate_stream
# from case_study5.stream_simulator_gala import simulator_hierarchical

# from stream_simulator_galax import sample_hierarchical_stream_priors, simulate_stream
# from stream_simulator_galax import simulator_hierarchical

from stream_simulator_odisseo import sample_hierarchical_stream_priors, simulate_stream
# from stream_simulator_odisseo import simulator_hierarchical

n_training = 50_000
batch_training = 100
n_stars = 1000 


prior_samples = {}
for i in range(0, n_training, ):
    if i == 0:
        sample = sample_hierarchical_stream_priors(n_streams=1, )
        for key in sample.keys():
            prior_samples[key] = [sample[key]]
    else:
        sample = sample_hierarchical_stream_priors(n_streams=1, )
        for key in sample.keys():
            prior_samples[key].append(sample[key])

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
        j=p["j"],                    # or fixed global j
        n_streams=1,
        n_stars=n_stars,
        key=p["key"],
    )["sim_data"]

params = {k: jnp.asarray(v) for k, v in prior_samples.items()}
N = params["prog_mass"].shape[0]
keys = random.split(random.PRNGKey(0), N)
params_with_keys = dict(**params, key=keys)

batched_map = partial(
    lax.map,
    simulate_one,
    batch_size=batch_training,   # tune for GPU/TPU memory
)

sim_data = batched_map(params_with_keys)
print(sim_data.shape)   # (n_training, n_stars, 6)

sim_data = {"sim_data": sim_data,}

np.savez('./training_set_odisseo.npz', 
         **sim_data,
         **prior_samples)


    