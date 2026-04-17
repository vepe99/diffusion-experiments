import os
from tqdm import tqdm

import omegaconf
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.core.config_store import ConfigStore

from astropy import units as u
import numpy as np


from utils.utils_simulate import (sample_parameters, 
                             sky_projection_astropy)
from config.SimulatorConfig import SimulatorConfig


cs = ConfigStore.instance()
cs.store(name="simulator_config", node=SimulatorConfig)

@hydra.main(version_base=None, config_path="config", config_name="simulate_config",)
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
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set this to the GPU you want to use
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
    prior_samples = sample_parameters(prior_global_dict=cfg.priors_global, prior_local_dict=cfg.priors_local, n_samples=total_n_simulations, target_streams=cfg.target_streams, key_seed=rng_seed)
    print(' The shapes of the samples are :', {k: v.shape for k, v in prior_samples.items()})

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
        if cfg.use_flattened_halo:
            from utils.utils_galax_simulator_generalized import simulate_stream_galax
        else:
            from utils.utils_galax_simulator import simulate_stream_galax

        simulate_stream = simulate_stream_galax
        config = cfg.galax_config
        code_units = None #galax does not use code units, but we need to pass something to the function
    
    elif cfg.simulator == "StreaMax":
        import jax
        import jax.numpy as jnp
        from utils_StreaMax_simulator import simulate_stream_StreaMAX

        simulate_stream = simulate_stream_StreaMAX
        config = cfg.streamax_config
        code_units = None #StreaMax does not use code units, but we need to pass something to the function

    elif cfg.simulator == "gala":
        from joblib import Parallel, delayed
        from utils.utils_gala_simulator import simulate_stream_gala, simulate_stream_gala_Rotated, _run_gala_single, simulate_stream_gala_SCF
        if cfg.use_rotated_halo:
            simulate_stream_gala_fn = simulate_stream_gala_Rotated
        elif cfg.use_SCF:
            simulate_stream_gala_fn = simulate_stream_gala_SCF
        else:
            simulate_stream_gala_fn = simulate_stream_gala
        config = cfg.gala_config
        code_units = None #gala does not use code units, but we need to pass something to the function
        # Override n_workers with the number of truly free cores
        # n_free = get_free_cores(threshold_percent=10.0)
        # print(f"Detected {n_free} idle cores (out of {os.cpu_count()}). Using them as workers.")
        


    for batch_start in tqdm(range(index_sim_start, index_sim_end, cfg.batch_size)):
        batch_end = min(batch_start + cfg.batch_size, index_sim_end)
        batch_indices = np.arange(batch_start, batch_end)
        local_indices = batch_indices - index_sim_start 
        # Prepare batch of parameters
        # Run vectorized simulation
        if (cfg.simulator == "odisseo")|(cfg.simulator == "galax")|(cfg.simulator == "StreaMax"):
            batch_params = {k: jnp.array(v[local_indices]) for k, v in prior_samples.items()}
            sim_data_batch = jax.vmap(simulate_stream, in_axes=(0, None, None, 0))(batch_params, config, code_units, jnp.array(batch_indices))  # shape (batch_size, ...)
        elif cfg.simulator == "gala":
            batch_params = {k: np.array(v[local_indices]) for k, v in prior_samples.items()}
    
            # Convert to individual parameter dicts
            individual_params = [
                {k: v[i] for k, v in batch_params.items()}
                for i in range(len(local_indices))
            ]
            
            # Run in parallel with joblib
            results = Parallel(n_jobs=config.n_workers)(
                delayed(simulate_stream_gala_fn)(
                    param_dict, config, code_units, int(batch_indices[i])
                )
                for i, param_dict in enumerate(individual_params)
            )
            
            # Stack results
            sim_data_batch = np.stack(results, axis=0)

        # Save each simulation in the batch
        sim_data_projected_batch = sky_projection_astropy(sim_data_batch)
        for i, idx in enumerate(batch_indices):
            local_idx = idx - index_sim_start
            params = {k: v[local_idx] for k, v in prior_samples.items()}
            sim_data = sim_data_batch[i]
            sim_data_projected = sim_data_projected_batch[i]
            # Save both Cartesian and projected data
            save_dict = {'sim_data_carthesian': sim_data, 'sim_data_projected': sim_data_projected}
            save_dict.update(params)
            np.savez(os.path.join(cfg.base_dir, cfg.data_dir, f'simulation_{idx}.npz'), **save_dict)

    print('sim_data_batch shape:', sim_data_batch.shape)
    
    
if __name__ == "__main__":
    main()