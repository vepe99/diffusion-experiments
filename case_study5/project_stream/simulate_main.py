from autocvd import autocvd
autocvd(num_gpus = 1)
import os
from tqdm import tqdm


from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.core.config_store import ConfigStore

from astropy import units as u
import numpy as np
import jax
import jax.numpy as jnp

from odisseo.option_classes import SimulationConfig
from odisseo.units import CodeUnits


from utils_simulate import (sample_parameters, 
                             convert_to_integer_externalacc, 
                             convert_to_integer_config,
                             simulate_stream,
                             sky_projection_astropy)
from simulate_config import SimulatorConfig

cs = ConfigStore.instance()
cs.store(name="simulator_config", node=SimulatorConfig)

@hydra.main(version_base=None, config_path="config", config_name="simulate_config",)
def main(cfg: SimulatorConfig):
    os.makedirs(os.path.join(cfg.base_dir, cfg.data_dir), exist_ok=True)
    print(cfg.odisseo_config)

    prior_samples = sample_parameters(prior_global_dict=cfg.priors_global, prior_local_dict=cfg.priors_local, n_samples=cfg.n_simulations, target_streams=cfg.target_streams)
    print(' The shapes of the samples are :', {k: v.shape for k, v in prior_samples.items()})

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
    
    
    for batch_start in tqdm(range(0, cfg.n_simulations, cfg.batch_size)):
        batch_end = min(batch_start + cfg.batch_size, cfg.n_simulations)
        batch_indices = np.arange(batch_start, batch_end)
        # Prepare batch of parameters
        batch_params = {k: jnp.array(v[batch_indices]) for k, v in prior_samples.items()}
        # Run vectorized simulation
        sim_data_batch = jax.vmap(simulate_stream, in_axes=(0, None, None, 0))(batch_params, config, code_units, jnp.array(batch_indices))  # shape (batch_size, ...)
        print('sim_data_batch shape:', sim_data_batch.shape)
        # Save each simulation in the batch
        sim_data_projected_batch = sky_projection_astropy(sim_data_batch)
        for i, idx in enumerate(batch_indices):
            params = {k: v[idx] for k, v in prior_samples.items()}
            sim_data = sim_data_batch[i]
            sim_data_projected = sim_data_projected_batch[i]
            # Save both Cartesian and projected data
            save_dict = {'sim_data_carthesian': sim_data, 'sim_data_projected': sim_data_projected}
            save_dict.update(params)
            np.savez(os.path.join(cfg.base_dir, cfg.data_dir, f'simulation_{idx}.npz'), **save_dict)
    
    
if __name__ == "__main__":
    main()