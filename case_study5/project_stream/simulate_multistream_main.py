from autocvd import autocvd
autocvd(num_gpus = 1)
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set this to the GPU you want to use
from tqdm import tqdm


from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.core.config_store import ConfigStore

from astropy import units as u
import numpy as np
import jax
import jax.numpy as jnp

from utils_simulate import (sample_parameters_parallel, 
                            sky_projection_astropy)
from simulate_config import SimulatorConfig

cs = ConfigStore.instance()
cs.store(name="simulator_config", node=SimulatorConfig)

@hydra.main(version_base=None, config_path="config", config_name="simulate_multistream_config",)
def main(cfg: SimulatorConfig):
    os.makedirs(os.path.join(cfg.base_dir, cfg.data_dir), exist_ok=True)
    print('Using simulator:', cfg.simulator)
    if cfg.simulator == "odisseo":
        print(cfg.odisseo_config)
    elif cfg.simulator == "gala":
        print(cfg.gala_config)
    elif cfg.simulator == "galax":
        print(cfg.galax_config)

    prior_samples = sample_parameters_parallel(prior_global_dict=cfg.priors_global, 
                                               prior_local_dict=cfg.priors_local, 
                                               n_samples=cfg.n_simulations, 
                                               target_streams=cfg.target_streams)
    print(' The shapes of the samples are :', {k: v.shape for k, v in prior_samples.items()})
    for k in cfg.priors_global.keys():
        prior_samples[k] = np.repeat(prior_samples[k][:, np.newaxis, :], len(cfg.target_streams), axis=1) # reshape global par ameters to have shape (n_samples, n_streans, 1)
    print(' The shapes of the samples after broadcasting are :', {k: v.shape for k, v in prior_samples.items()})
    for k in prior_samples.keys():
        prior_samples[k] = prior_samples[k].reshape(-1, 1)
    print(' The shapes of the samples flattened are :', {k: v.shape for k, v in prior_samples.items()})

    if cfg.simulator == "odisseo":
        from odisseo.option_classes import SimulationConfig
        from odisseo.units import CodeUnits
        from utils_odisseo_simulator import (convert_to_integer_externalacc, 
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
        
    elif cfg.simulator == "gala":
        config = cfg.gala_config
        code_units = None #gala does not use code units, but we need to pass something to the function
        pass

    elif cfg.simulator == "galax":
        from utils_galax_simulator import simulate_stream_galax

        simulate_stream = simulate_stream_galax
        config = cfg.galax_config
        code_units = None #galax does not use code units, but we need to pass something to the function
        
        
    
    save_dict = {'sim_data_carthesian': np.zeros((cfg.n_simulations * len(cfg.target_streams), cfg.odisseo_config.N_particles, 6)),
                 'sim_data_projected': np.zeros((cfg.n_simulations * len(cfg.target_streams),cfg.odisseo_config.N_particles, 6))}
    for batch_start in tqdm(range(0, cfg.n_simulations, cfg.batch_size)):
        batch_end = min(batch_start + cfg.batch_size, cfg.n_simulations)
        batch_indices = np.arange(batch_start, batch_end)
        # Prepare batch of parameters
        batch_params = {k: jnp.array(v[batch_indices]) for k, v in prior_samples.items()}
        # Run vectorized simulation
        sim_data_batch = jax.vmap(simulate_stream, in_axes=(0, None, None, 0))(batch_params, config, code_units, jnp.array(batch_indices))  # shape (batch_size, ...)
        
        # Save each simulation in the batch
        sim_data_projected_batch = sky_projection_astropy(sim_data_batch)
        save_dict['sim_data_carthesian'][batch_indices] = sim_data_batch
        save_dict['sim_data_projected'][batch_indices] = sim_data_projected_batch
    print('sim_data_batch shape:', sim_data_batch.shape)
    
    save_dict['sim_data_carthesian'] = save_dict['sim_data_carthesian'].reshape(cfg.n_simulations, len(cfg.target_streams), cfg.odisseo_config.N_particles, 6)
    save_dict['sim_data_projected'] = save_dict['sim_data_projected'].reshape(cfg.n_simulations, len(cfg.target_streams), cfg.odisseo_config.N_particles, 6)
    for k in prior_samples.keys():
        if k in cfg.priors_global.keys():
            prior_samples[k] = prior_samples[k].reshape(-1, len(cfg.target_streams),  1)[:, 0, :] # reshape back to (n_samples, 1) for global parameters
        else:
            prior_samples[k] = prior_samples[k].reshape(-1, len(cfg.target_streams),  1) 
    save_dict.update(prior_samples)
    print(' The shapes of the simulations and parameters before saving are :', {k: v.shape for k, v in save_dict.items()})
    np.savez(os.path.join(cfg.base_dir, cfg.data_dir, f'simulation_multistream_{cfg.n_simulations}.npz'), **save_dict)
    
if __name__ == "__main__":
    main()