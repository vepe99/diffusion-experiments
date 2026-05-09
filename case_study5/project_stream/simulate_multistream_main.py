
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set this to the GPU you want to use
import omegaconf
from tqdm import tqdm


from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.core.config_store import ConfigStore

from astropy import units as u
import numpy as np

from utils.utils_simulate import (sample_parameters_parallel, 
                            sky_projection_astropy)
from config.SimulatorConfig import SimulatorConfig

cs = ConfigStore.instance()
cs.store(name="simulator_config", node=SimulatorConfig)

@hydra.main(version_base=None, config_path="config", config_name="simulate_multistream_config",)
def main(cfg: SimulatorConfig):
    os.makedirs(os.path.join(cfg.base_dir, cfg.data_dir), exist_ok=True)
    print('Using simulator:', cfg.simulator)
    if cfg.simulator == "odisseo":
        print(cfg.odisseo_config)
        from autocvd import autocvd
        autocvd(num_gpus = 1)
    elif cfg.simulator == "galax":
        print(cfg.galax_config)
        from autocvd import autocvd
        autocvd(num_gpus = 1)
    elif cfg.simulator == "StreaMax":
        print(cfg.streamax_config)
        from autocvd import autocvd
        autocvd(num_gpus = 1)
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set this to the GPU you want to use
    elif cfg.simulator == "gala":
        print(cfg.gala_config)

    if isinstance(cfg.n_simulations, omegaconf.listconfig.ListConfig):
        index_sim_start = cfg.n_simulations[0]
        index_sim_end = cfg.n_simulations[1]
        total_n_simulations = index_sim_end - index_sim_start   # total number to sample
    else:
        index_sim_start = 0
        index_sim_end = cfg.n_simulations
        total_n_simulations = cfg.n_simulations
        
    rng_seed = index_sim_end
    np.random.seed(rng_seed)
    prior_samples = sample_parameters_parallel(prior_global_dict=cfg.priors_global, prior_local_dict=cfg.priors_local, n_samples=total_n_simulations, target_streams=cfg.target_streams, key_seed=rng_seed)
    print(' The shapes of the samples are :', {k: v.shape for k, v in prior_samples.items()})
    for k in cfg.priors_global.keys():
        prior_samples[k] = np.repeat(prior_samples[k][:, np.newaxis, :], len(cfg.target_streams), axis=1) # reshape global par ameters to have shape (n_samples, n_streans, 1)
    print(' The shapes of the samples after broadcasting are :', {k: v.shape for k, v in prior_samples.items()})
    for k in prior_samples.keys():
        prior_samples[k] = prior_samples[k].reshape(-1, 1)
    print(' The shapes of the samples flattened are :', {k: v.shape for k, v in prior_samples.items()})

    if cfg.simulator == "odisseo":
        import jax
        import jax.numpy as jnp
        from odisseo.option_classes import SimulationConfig
        from odisseo.units import CodeUnits
        from utils.utils_odisseo_simulator import (convert_to_integer_externalacc, 
                                                convert_to_integer_config,
                                                simulate_stream_odisseo)
        
        simulate_stream = simulate_stream_odisseo
        code_length = 1 * u.kpc
        code_mass = 1e3 * u.Msun
        code_time = 1 * u.Gyr
        code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  
        print(convert_to_integer_externalacc(cfg.odisseo_config.external_accelerations), )
        config = SimulationConfig(N_particles = cfg.odisseo_config.N_particles, 
                                return_snapshots = cfg.odisseo_config.return_snapshots, 
                                num_snapshots = cfg.odisseo_config.num_snapshots, 
                                num_timesteps = cfg.odisseo_config.num_timesteps, 
                                external_accelerations = convert_to_integer_externalacc(cfg.odisseo_config.external_accelerations), 
                                acceleration_scheme = convert_to_integer_config(cfg.odisseo_config.acceleration_scheme),
                                softening = (cfg.odisseo_config.softening * u.pc).to(code_units.code_length).value,
                                integrator= convert_to_integer_config(cfg.odisseo_config.integrator),
                                fixed_timestep= cfg.odisseo_config.fixed_timestep,
                                diffrax_solver= convert_to_integer_config(cfg.odisseo_config.diffrax_solver),
                                glorder=cfg.odisseo_config.glorder) #default values
        
    
    elif cfg.simulator == "galax":
        import jax
        import jax.numpy as jnp
        from utils.utils_galax_simulator import simulate_stream_galax

        simulate_stream = simulate_stream_galax
        config = cfg.galax_config
        code_units = None #galax does not use code units, but we need to pass something to the function

    elif cfg.simulator == "StreaMax":
        import jax
        import jax.numpy as jnp
        from utils.utils_StreaMax_simulator import simulate_stream_StreaMAX

        simulate_stream = simulate_stream_StreaMAX
        config = cfg.streamax_config
        code_units = None #StreaMax does not use code units, but we need to pass something to the function
    
    elif cfg.simulator == "gala":
        from joblib import Parallel, delayed
        from utils.utils_gala_simulator import simulate_stream_gala, _run_gala_single
        config = cfg.gala_config
        code_units = None #gala does not use code units, but we need to pass something to the function
        
    
    save_dict = {'sim_data_carthesian': np.ones((total_n_simulations* len(cfg.target_streams), cfg.odisseo_config.N_particles, 6)),
                 'sim_data_projected': np.ones((total_n_simulations* len(cfg.target_streams),cfg.odisseo_config.N_particles, 6))}
    for batch_start in tqdm(range(0, total_n_simulations * len(cfg.target_streams), cfg.batch_size)):
        batch_end = min(batch_start + cfg.batch_size, total_n_simulations * len(cfg.target_streams))
        batch_indices = np.arange(batch_start, batch_end)
        # Prepare batch of parameters
        if (cfg.simulator == "odisseo")|(cfg.simulator == "galax")|(cfg.simulator == "StreaMax"):
            batch_params = {k: jnp.array(v[batch_indices]) for k, v in prior_samples.items()}
            sim_data_batch = jax.vmap(simulate_stream, in_axes=(0, None, None, 0))(batch_params, config, code_units, jnp.array(batch_indices))  # shape (batch_size, ...)
            if cfg.simulator == "StreaMax":
                n_particles_subsample = 1000
                sim_data_clean = []
                for sim in sim_data_batch:
                    mask = ~np.isnan(sim).any(axis=1)
                    sim_no_nan = sim[mask]
                    n_timesteps_clean = sim_no_nan.shape[0]
                    # Subsample timesteps
                    if n_timesteps_clean >= n_particles_subsample:
                        indices = np.random.choice(n_timesteps_clean, size=n_particles_subsample, replace=False)
                        indices.sort()
                        sim_subsampled = sim_no_nan[indices]
                    else:
                        # If not enough timesteps, pad with NaNs
                        pad = np.full((n_particles_subsample - n_timesteps_clean, sim_no_nan.shape[1]), np.nan)
                        sim_subsampled = np.vstack([sim_no_nan, pad])
                    sim_data_clean.append(sim_subsampled)
                sim_data_batch = np.stack(sim_data_clean, axis=0)
            else:
                pass
        # elif cfg.simulator == "gala":
        #     batch_params = {k: np.array(v[batch_indices]) for k, v in prior_samples.items()}
    
            # # Convert to individual parameter dicts
            # individual_params = [
            #     {k: v[i] for k, v in batch_params.items()}
            #     for i in range(len(batch_indices))
            # ]
            
            # # Run in parallel with joblib
            # results = Parallel(n_jobs=config.n_workers)(
            #     delayed(simulate_stream_gala)(
            #         param_dict, config, code_units, int(batch_indices[i])
            #     )
            #     for i, param_dict in enumerate(individual_params)
            # )
            
            # # Stack results
            # sim_data_batch = np.stack(results, axis=0)
        # elif cfg.simulator == "gala":
        #     batch_params = {k: np.array(v[batch_indices]) for k, v in prior_samples.items()}

        #     individual_params = [
        #         {k: v[i] for k, v in batch_params.items()}
        #         for i in range(len(batch_indices))
        #     ]

        #     # ── safe wrapper ──────────────────────────────────────────────
        #     n_particles = cfg.odisseo_config.N_particles

        #     def _safe_simulate(param_dict, config, code_units, idx):
        #         try:
        #             return simulate_stream_gala(param_dict, config, code_units, idx)
        #         except RuntimeError as e:
        #             print(
        #                 f"[WARNING] Integration failed for index {idx} "
        #                 f"(code -4 or similar): {e}. Returning NaNs."
        #             )
        #             return np.full((n_particles+2, 6), np.nan)
        #     # ─────────────────────────────────────────────────────────────

        #     results = Parallel(n_jobs=config.n_workers)(
        #         delayed(_safe_simulate)(
        #             param_dict, config, code_units, int(batch_indices[i])
        #         )
        #         for i, param_dict in enumerate(individual_params)
        #     )
        #     for r in results:
        #         print('Result shape:', r.shape)
        #     sim_data_batch = np.stack(results, axis=0)

        elif cfg.simulator == "gala":
            batch_params = {k: np.array(v[batch_indices]) for k, v in prior_samples.items()}

            individual_params = [
                {k: v[i] for k, v in batch_params.items()}
                for i in range(len(batch_indices))
            ]

            n_particles_gala = cfg.odisseo_config.N_particles + 2

            def _safe_simulate(param_dict, config, code_units, idx):
                try:
                    return simulate_stream_gala(param_dict, config, code_units, idx)
                except RuntimeError as e:
                    print(f"[WARNING] Integration failed for index {idx} (code -4 or similar): {e}. Returning NaNs.")
                    return np.full((n_particles_gala, 6), np.nan)

            def _normalize_shape(arr, n):
                if arr.shape[0] == n:
                    return arr
                elif arr.shape[0] > n:
                    idx = np.linspace(0, arr.shape[0] - 1, n, dtype=int)
                    return arr[idx]
                else:
                    pad = np.full((n - arr.shape[0], 6), np.nan)
                    return np.vstack([arr, pad])

            results = Parallel(n_jobs=config.n_workers)(
                delayed(_safe_simulate)(param_dict, config, code_units, int(batch_indices[i]))
                for i, param_dict in enumerate(individual_params)
            )
            results = [_normalize_shape(r, n_particles_gala) for r in results]

            # ── retry loop ────────────────────────────────────────────────
            n_streams = len(cfg.target_streams)

            max_retries = 10
            retry_count = 0
            while retry_count < max_retries:
                failed_local = [i for i, r in enumerate(results) if np.isnan(r).any()]
                if not failed_local:
                    break

                # Map failed stream slots → unique failed simulation indices
                failed_sim_indices = list(dict.fromkeys(
                    int(batch_indices[local_i]) // n_streams for local_i in failed_local
                ))
                print(f"[RETRY {retry_count + 1}/{max_retries}] Resampling {len(failed_sim_indices)} failed simulation(s) (all streams)...")

                # Resample one full simulation (all streams) per failed sim
                new_samples = sample_parameters_parallel(
                    prior_global_dict=cfg.priors_global,
                    prior_local_dict=cfg.priors_local,
                    n_samples=len(failed_sim_indices),
                    target_streams=cfg.target_streams,
                    key_seed=rng_seed + retry_count + 1,
                )
                for k in cfg.priors_global.keys():
                    new_samples[k] = np.repeat(new_samples[k][:, np.newaxis, :], n_streams, axis=1)
                for k in new_samples.keys():
                    new_samples[k] = new_samples[k].reshape(-1, 1)  # shape: (n_failed_sims * n_streams, 1)

                # Build individual param dicts for every (failed_sim, stream) slot
                retry_local_indices = []   # positions in `results` to patch
                retry_param_dicts  = []

                for j, sim_idx in enumerate(failed_sim_indices):
                    for s in range(n_streams):
                        flat_row   = j * n_streams + s          # row in new_samples
                        # find the local position in the batch for this (sim, stream)
                        global_i   = sim_idx * n_streams + s
                        local_i    = int(np.where(batch_indices == global_i)[0][0])

                        retry_local_indices.append(local_i)
                        retry_param_dicts.append({k: new_samples[k][flat_row] for k in new_samples.keys()})

                retry_results = Parallel(n_jobs=config.n_workers)(
                    delayed(_safe_simulate)(param_dict, config, code_units, int(batch_indices[local_i]))
                    for param_dict, local_i in zip(retry_param_dicts, retry_local_indices)
                )
                retry_results = [_normalize_shape(r, n_particles_gala) for r in retry_results]

                # Patch results and prior_samples for the entire triplet
                for idx, (local_i, param_dict) in enumerate(zip(retry_local_indices, retry_param_dicts)):
                    results[local_i] = retry_results[idx]
                    global_i = int(batch_indices[local_i])
                    for k in new_samples.keys():
                        prior_samples[k][global_i] = param_dict[k]

                retry_count += 1

            still_failed = [i for i, r in enumerate(results) if np.isnan(r).any()]
            if still_failed:
                print(f"[WARNING] {len(still_failed)} simulation(s) still NaN after {max_retries} retries: local indices {still_failed}")

            sim_data_batch = np.stack(results, axis=0)
        # Save each simulation in the batch
        sim_data_projected_batch = sky_projection_astropy(sim_data_batch)
        save_dict['sim_data_carthesian'][batch_indices] = sim_data_batch
        save_dict['sim_data_projected'][batch_indices] = sim_data_projected_batch

    print('sim_data_batch shape:', sim_data_batch.shape)
    
    save_dict['sim_data_carthesian'] = save_dict['sim_data_carthesian'].reshape(total_n_simulations, len(cfg.target_streams), cfg.odisseo_config.N_particles, 6)
    save_dict['sim_data_projected'] = save_dict['sim_data_projected'].reshape(total_n_simulations, len(cfg.target_streams), cfg.odisseo_config.N_particles, 6)
    for k in prior_samples.keys():
        if k in cfg.priors_global.keys():
            prior_samples[k] = prior_samples[k].reshape(-1, len(cfg.target_streams),  1)[:, 0, :] # reshape back to (n_samples, 1) for global parameters
        else:
            prior_samples[k] = prior_samples[k].reshape(-1, len(cfg.target_streams),  1) 
    save_dict.update(prior_samples)
    print(' The shapes of the simulations and parameters before saving are :', {k: v.shape for k, v in save_dict.items()})
    np.savez(os.path.join(cfg.base_dir, cfg.data_dir, f'simulation_multistream_{total_n_simulations}.npz'), **save_dict)
    
if __name__ == "__main__":
    main()