import numpy as np
import os
import yaml



# def load_simulations_to_dict(base_directory, subfolders, N_simulations_per_folder, param_names_global, 
#                               sim_key='sim_data_projected', j_key='j', n_particles_subsample=1000, seed=42):
#     """
#     Loads simulation files from multiple subfolders into a dictionary of numpy arrays.

#     Parameters
#     ----------
#     base_directory : str
#         Path to the base directory containing subfolders.
#     subfolders : list of str
#         List of subfolder names (e.g., ['data_streamax_1', 'data_streamax_2', 'data_streamax_3']).
#     N_simulations_per_folder : list of tuple
#         List of (start_idx, end_idx) for each subfolder.
#     param_names_global : list of str
#         List of parameter names to extract.
#     sim_key : str
#         Key for the projected simulation data in the npz files.
#     j_key : str
#         Key for the 'j' variable in the npz files.
#     n_particles_subsample : int
#         Number of particles to randomly subsample from each simulation.
#     seed : int
#         Random seed for reproducible subsampling.

#     Returns
#     -------
#     data_dict : dict
#         Dictionary with keys param_names_global + [sim_key, j_key], each value is a numpy array.
#     """
#     rng = np.random.default_rng(seed)
    
#     # Compute total number of simulations
#     N_total = sum(end - start for start, end in N_simulations_per_folder)

#     # First, load the first file to get shapes
#     first_subfolder = subfolders[0]
#     first_start = N_simulations_per_folder[0][0]
#     first_file = os.path.join(base_directory, first_subfolder, f"simulation_{first_start}.npz")
#     with np.load(first_file) as data:
#         j_shape = data[j_key].shape
#         param_shapes = {k: data[k].shape for k in param_names_global}

#     # Preallocate arrays
#     data_dict = {}
#     for k in param_names_global:
#         data_dict[k] = np.zeros((N_total,) + param_shapes[k], dtype=np.float32)
#     data_dict[sim_key] = np.zeros((N_total, n_particles_subsample, 6), dtype=np.float32)
#     data_dict[j_key] = np.zeros((N_total,) + j_shape, dtype=np.float32)

#     # Fill arrays
#     global_idx = 0
#     for subfolder, (start_idx, end_idx) in zip(subfolders, N_simulations_per_folder):
#         folder_path = os.path.join(base_directory, subfolder)
#         print(f"Loading from {folder_path} (simulations {start_idx} to {end_idx - 1})...")
        
#         for i in range(start_idx, end_idx):
#             fname = os.path.join(folder_path, f"simulation_{i}.npz")
#             if not os.path.exists(fname):
#                 print(f"File not found: {fname}")
#                 global_idx += 1
#                 continue
#             with np.load(fname) as data:
#                 for k in param_names_global:
#                     data_dict[k][global_idx] = data[k]
                
#                 # Subsample particles: (N_particles, 6) -> (n_particles_subsample, 6)
#                 sim_data = data[sim_key]
#                 n_particles_total = sim_data.shape[0]
#                 if n_particles_total <= n_particles_subsample:
#                     # Pad with zeros if fewer particles than requested
#                     data_dict[sim_key][global_idx, :n_particles_total] = sim_data
#                 else:
#                     indices = rng.choice(n_particles_total, size=n_particles_subsample, replace=False)
#                     indices.sort()  # keep particle ordering
#                     data_dict[sim_key][global_idx] = sim_data[indices]
                
#                 data_dict[j_key][global_idx] = data[j_key]
            
#             if global_idx % 10_000 == 0:
#                 print(f"Loaded {global_idx}/{N_total} simulations...")
#             global_idx += 1

#     print(f"Finished loading {global_idx} simulations.")
#     return data_dict

def load_simulations_to_dict(base_directory, subfolders, N_simulations_per_folder, param_names_global, 
                              sim_key='sim_data_projected', j_key='j', n_particles_subsample=1000, seed=42):
    rng = np.random.default_rng(seed)
    N_total = sum(end - start for start, end in N_simulations_per_folder)

    # First, load the first file to get shapes
    first_subfolder = subfolders[0]
    first_start = N_simulations_per_folder[0][0]
    first_file = os.path.join(base_directory, first_subfolder, f"simulation_{first_start}.npz")
    with np.load(first_file) as data:
        j_shape = data[j_key].shape
        param_shapes = {k: data[k].shape for k in param_names_global}

    # Preallocate lists to collect valid simulations
    valid_params = {k: [] for k in param_names_global}
    valid_sim_data = []
    valid_j = []
    valid_indices = []

    global_idx = 0
    for subfolder, (start_idx, end_idx) in zip(subfolders, N_simulations_per_folder):
        folder_path = os.path.join(base_directory, subfolder)
        print(f"Loading from {folder_path} (simulations {start_idx} to {end_idx - 1})...")
        for i in range(start_idx, end_idx):
            fname = os.path.join(folder_path, f"simulation_{i}.npz")
            if not os.path.exists(fname):
                print(f"File not found: {fname}")
                global_idx += 1
                continue
            with np.load(fname) as data:
                # Check for NaNs in global parameters
                has_nan_global = any(np.isnan(data[k]).any() for k in param_names_global)
                if has_nan_global:
                    print(f"Skipping simulation {i} due to NaN in global parameters.")
                    global_idx += 1
                    continue

                # Remove stars with NaNs in sim_data_projected
                sim_data = data[sim_key]
                mask = ~np.isnan(sim_data).any(axis=1)
                sim_data_clean = sim_data[mask]
                n_particles_clean = sim_data_clean.shape[0]

                # Skip simulation if not enough clean stars
                if n_particles_clean < n_particles_subsample:
                    print(f"Skipping simulation {i} due to too few clean stars ({n_particles_clean}).")
                    global_idx += 1
                    continue

                # Subsample clean stars
                indices = rng.choice(n_particles_clean, size=n_particles_subsample, replace=False)
                indices.sort()
                sim_data_subsampled = sim_data_clean[indices]

                # Collect valid data
                for k in param_names_global:
                    valid_params[k].append(data[k])
                valid_sim_data.append(sim_data_subsampled)
                valid_j.append(data[j_key])
                valid_indices.append(i)

            if global_idx % 10_000 == 0:
                print(f"Processed {global_idx}/{N_total} simulations...")
            global_idx += 1

    # Convert lists to arrays
    N_valid = len(valid_sim_data)
    print(f"Total valid simulations: {N_valid}")
    data_dict = {}
    for k in param_names_global:
        data_dict[k] = np.array(valid_params[k], dtype=np.float32)
    data_dict[sim_key] = np.array(valid_sim_data, dtype=np.float32)
    data_dict[j_key] = np.array(valid_j, dtype=np.float32)
    data_dict['simulation_indices'] = np.array(valid_indices, dtype=np.int64)

    print(f"Finished loading {N_valid} valid simulations.")
    return data_dict

if __name__ == "__main__":
    base_directory = "/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/streams/data_streamax/"
    
    subfolders = ["data_streamax_1", "data_streamax_2", "data_streamax_3"]
    # subfolders = ["data_streamax_validation"]
    # Infer N_simulations_per_folder from the number of .npz files in each subfolder
    N_simulations_per_folder = []
    cumulative = 0
    for subfolder in subfolders:
        folder_path = os.path.join(base_directory, subfolder)
        n_files = len([f for f in os.listdir(folder_path) if f.startswith("simulation_") and f.endswith(".npz")])
        N_simulations_per_folder.append((cumulative, cumulative + n_files))
        cumulative += n_files
        print(f"{subfolder}: {n_files} files (indices {N_simulations_per_folder[-1][0]} to {N_simulations_per_folder[-1][1] - 1})")
    N_total = cumulative
    print(f"Total simulations: {N_total}")
    n_particles_subsample = 1000

    # Load config from first subfolder
    config_path = os.path.join(base_directory, subfolders[0], '.hydra', 'config.yaml')
    with open(config_path, "r") as f:
        test_sim_config = yaml.safe_load(f)
    param_names_global = list(test_sim_config['priors_global'].keys())
    print("Parameter names global: ", param_names_global)
    
    data_dict = load_simulations_to_dict(
        base_directory, 
        subfolders, 
        N_simulations_per_folder, 
        param_names_global,
        n_particles_subsample=n_particles_subsample,
    )
    
    save_path = os.path.join(base_directory, f"training_data_{N_total}.npz")
    # save_path = os.path.join(base_directory, f"validation_data_{N_total}.npz")
    np.savez(save_path, **data_dict)
    print(f"Saved to {save_path}")