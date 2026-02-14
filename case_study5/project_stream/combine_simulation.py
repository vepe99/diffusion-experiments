import numpy as np
import os
import yaml



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

if __name__ == "__main__":
    # Example usage:
    directory = "/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/streams/data_galax/"
    N_simulations = 300_000
    with open(os.path.join(directory, '.hydra', 'config.yaml'), "r") as f:
        test_sim_config = yaml.safe_load(f)
    param_names_global = list(test_sim_config['priors_global'].keys())
    print("Parameter names global: ", param_names_global)
    data_dict = load_simulations_to_dict(directory, N_simulations, param_names_global)
    np.savez(f"/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/streams/data_galax/training_data_{N_simulations}.npz", **data_dict)