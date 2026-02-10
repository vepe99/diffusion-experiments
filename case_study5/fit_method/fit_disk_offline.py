from autocvd import autocvd
autocvd(num_gpus = 1)

backend = "jax"
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = backend

import bayesflow as bf
import numpy as np


param_names_global = ['m_Triaxial_halo', 'r_Triaxial_halo', 'q2_Triaxial_halo', 'rho_thin_disk', 'hr_thin_disk', 'hz_thin_disk', 'rho_thick_disk', 'hr_thick_disk', 'hz_thick_disk']
        
    
sim_data = "sim_data_projected"
adapter = (
    bf.adapters.Adapter()
    .to_array()
    .convert_dtype("float64", "float32")
    .concatenate(param_names_global, into="inference_variables")
    .rename("sim_data_projected", "summary_variables")
    .rename("j", "inference_conditions")
)
workflow_global = bf.BasicWorkflow(
    adapter=adapter,
    summary_network=bf.networks.SetTransformer(summary_dim=64, 
                                            #    num_heads=(4, 4),
                                               dropout=0.1),
    inference_network=bf.networks.CompositionalDiffusionModel(),
    standardize=["inference_variables", "summary_variables"]
)

# Using fit_disk
# def load_npz_as_dict(file):
#     with np.load(file) as data_npz:
#         data = {k: v for k, v in data_npz.items() if k in param_names_global or k in ['sim_data_projected', 'j']}
#         data['sim_data_projected'] = data['sim_data_projected'][:100]
#         return data
    
# history = workflow_global.fit_disk(
#         root = '/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/streams/data/',
#         pattern = "*.npz",
#         load_fn = load_npz_as_dict,
#         epochs = 2,
#         batch_size = 20_000,
#         verbose=2,
#     )


#using fit_offline
import numpy as np
import os

def load_simulations_to_dict(directory, N_simulations, param_names_global, sim_key='sim_data_projected', j_key='j'):
    """
    Loads simulation files into a dictionary of numpy arrays.

    Parameters
    ----------
    directory : str
        Path to the directory containing simulation_*.npz files.
    N_simulations : int
        Number of simulations to load.
    param_names_global : list of str
        List of parameter names to extract.
    sim_key : str
        Key for the projected simulation data in the npz files.
    j_key : str
        Key for the 'j' variable in the npz files.

    Returns
    -------
    data_dict : dict
        Dictionary with keys param_names_global + [sim_key, j_key], each value is a numpy array of length N_simulations.
    """
    # First, load the first file to get shapes
    first_file = os.path.join(directory, f"simulation_0.npz")
    with np.load(first_file) as data:
        sim_shape = data[sim_key].shape
        j_shape = data[j_key].shape
        param_shapes = {k: data[k].shape for k in param_names_global}

    # Preallocate arrays
    data_dict = {}
    for k in param_names_global:
        data_dict[k] = np.zeros((N_simulations,) + param_shapes[k], dtype=np.float32)
    data_dict[sim_key] = np.zeros((N_simulations,) + sim_shape, dtype=np.float32)
    data_dict[j_key] = np.zeros((N_simulations,) + j_shape, dtype=np.float32)

    # Fill arrays
    for i in range(N_simulations):
        fname = os.path.join(directory, f"simulation_{i}.npz")
        if not os.path.exists(fname):
            print(f"File not found: {fname}")
            continue
        with np.load(fname) as data:
            for k in param_names_global:
                data_dict[k][i] = data[k]
            data_dict[sim_key][i] = data[sim_key]
            data_dict[j_key][i] = data[j_key]
        if i % 10_000 == 0:
            print(f"Loaded {i} simulations...")

    return data_dict

# Example usage:
directory = "/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/streams/data/"
N_simulations = 300_000
data_dict = load_simulations_to_dict(directory, N_simulations, param_names_global)
np.savez("../project_stream/data/streams/data/training_data_300000.npz", **data_dict)

data_dict = np.load("../project_stream/data/streams/data/training_data_300000.npz")
data_dict = dict(data_dict)
print({k: v.shape for k, v in data_dict.items()})

# training_data = data_dict

# history = workflow_global.fit_offline(
#         training_data,
#         epochs=2,
#         batch_size=128,
#         verbose=2,
#     )

# print(history)