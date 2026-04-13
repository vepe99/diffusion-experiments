
# import os
# # os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set this to the GPU you want to use
# import omegaconf
# from tqdm import tqdm


# from omegaconf import DictConfig, OmegaConf, open_dict
# import hydra
# from hydra.core.config_store import ConfigStore

# from astropy import units as u
# import numpy as np

# from utils.utils_simulate import (sample_parameters_parallel, 
#                             sky_projection_astropy)
# from config.SimulatorConfig import SimulatorConfig

# cs = ConfigStore.instance()
# cs.store(name="simulator_config", node=SimulatorConfig)

# @hydra.main(version_base=None, config_path="config", config_name="posterior_predictive_check_config",)
# def main(cfg: SimulatorConfig):
#     os.makedirs(os.path.join(cfg.base_dir, cfg.data_dir), exist_ok=True)
#     print('Using simulator:', cfg.simulator)
#     if cfg.simulator == "odisseo":
#         print(cfg.odisseo_config)
#         from autocvd import autocvd
#         autocvd(num_gpus = 1)
#     elif cfg.simulator == "galax":
#         print(cfg.galax_config)
#         from autocvd import autocvd
#         autocvd(num_gpus = 1)
#     elif cfg.simulator == "StreaMax":
#         print(cfg.streamax_config)
#         from autocvd import autocvd
#         autocvd(num_gpus = 1)
#         # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set this to the GPU you want to use
#     elif cfg.simulator == "gala":
#         print(cfg.gala_config)

#     if isinstance(cfg.n_simulations, omegaconf.listconfig.ListConfig):
#         index_sim_start = cfg.n_simulations[0]
#         index_sim_end = cfg.n_simulations[1]
#         total_n_simulations = index_sim_end - index_sim_start   # total number to sample
#     else:
#         index_sim_start = 0
#         index_sim_end = cfg.n_simulations
#         total_n_simulations = cfg.n_simulations
        
#     rng_seed = index_sim_end
#     np.random.seed(rng_seed)

#     # ── Overwrite priors with posterior means ──────────────────────────────
#     global_posterior_path = os.path.join(
#         cfg.base_dir,
#         '../plots/gala6D_aug/new_hyper/model54_60k_1000epochs/global_posterior.npz'
#     )
#     local_posterior_path = os.path.join(
#         cfg.base_dir,
#         '../plots/plots_local/gala6D_aug/new_hyper/model54_60k_1000epochs/gaia_local_posterior.npz'
#     )

#     print('Loading global posterior from:', global_posterior_path)
#     global_posterior = dict(np.load(global_posterior_path, allow_pickle=True))

#     print('Loading local posterior from:', local_posterior_path)
#     local_posterior  = dict(np.load(local_posterior_path,  allow_pickle=True))

#     stream_names = list(cfg.target_streams.keys())   # ['Pal5', 'NGC3201', 'M68']
#     n_streams    = len(stream_names)

#     with open_dict(cfg):
#         # --- Global parameters ---
#         for param in list(cfg.priors_global.keys()):
#             if param in global_posterior:
#                 mean_val = float(np.asarray(global_posterior[param]).mean())
#                 cfg.priors_global[param] = {
#                     'type': 'identity',
#                     'prior_parameters': [mean_val],
#                 }
#                 print(f'  [global] {param}: identity({mean_val:.6g})')
#             else:
#                 print(f'  [global] {param}: not in posterior, kept as-is')

#         # --- Local parameters ---
#         # local_posterior[param] shape: (1, N_STREAMS, N_SAMPLES, 1) or (N_STREAMS, N_SAMPLES, 1)
#         for param in ['vr', 'r', 'mu_ra_cosdec', 'mu_dec']:
#             if param not in local_posterior:
#                 print(f'  [local]  {param}: not in posterior, kept as-is')
#                 continue
#             arr = np.asarray(local_posterior[param])          # flatten all but stream axis
#             arr = arr.reshape(n_streams, -1)                  # (N_STREAMS, N_SAMPLES)
#             for s_idx, stream_name in enumerate(stream_names):
#                 mean_val = float(arr[s_idx].mean())
#                 cfg.priors_local[stream_name][param] = {
#                     'type': 'identity',
#                     'prior_parameters': [mean_val],
#                 }
#                 print(f'  [local]  {stream_name}.{param}: identity({mean_val:.6g})')

#     print('\nPriors after posterior override:')
#     for k, v in cfg.priors_global.items():
#         print(f'  [global] {k}: {v}')
#     for stream_name in stream_names:
#         for k, v in cfg.priors_local[stream_name].items():
#             print(f'  [local]  {stream_name}.{k}: {v}')


#     prior_samples = sample_parameters_parallel(prior_global_dict=cfg.priors_global, 
#                                                prior_local_dict=cfg.priors_local, 
#                                                n_samples=total_n_simulations, 
#                                                target_streams=cfg.target_streams, 
#                                                key_seed=rng_seed)
    
#     print(' The shapes of the samples are :',)
#     for k, v in prior_samples.items():
#         print(f' {k}: {v.shape}')

#     for k in cfg.priors_global.keys():
#         prior_samples[k] = np.repeat(prior_samples[k][:, np.newaxis, :], len(cfg.target_streams), axis=1) # reshape global par ameters to have shape (n_samples, n_streans, 1)
#     print(' The shapes of the samples after broadcasting are :', {k: v.shape for k, v in prior_samples.items()})
#     for k in prior_samples.keys():
#         prior_samples[k] = prior_samples[k].reshape(-1, 1)
#     print(' The shapes of the samples flattened are :', {k: v.shape for k, v in prior_samples.items()})

#     if cfg.simulator == "odisseo":
#         import jax
#         import jax.numpy as jnp
#         from odisseo.option_classes import SimulationConfig
#         from odisseo.units import CodeUnits
#         from utils.utils_odisseo_simulator import (convert_to_integer_externalacc, 
#                                                 convert_to_integer_config,
#                                                 simulate_stream_odisseo)
        
#         simulate_stream = simulate_stream_odisseo
#         code_length = 1 * u.kpc
#         code_mass = 1e3 * u.Msun
#         code_time = 1 * u.Gyr
#         code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  
#         print(convert_to_integer_externalacc(cfg.odisseo_config.external_accelerations), )
#         config = SimulationConfig(N_particles = cfg.odisseo_config.N_particles, 
#                                 return_snapshots = cfg.odisseo_config.return_snapshots, 
#                                 num_snapshots = cfg.odisseo_config.num_snapshots, 
#                                 num_timesteps = cfg.odisseo_config.num_timesteps, 
#                                 external_accelerations = convert_to_integer_externalacc(cfg.odisseo_config.external_accelerations), 
#                                 acceleration_scheme = convert_to_integer_config(cfg.odisseo_config.acceleration_scheme),
#                                 softening = (cfg.odisseo_config.softening * u.pc).to(code_units.code_length).value,
#                                 integrator= convert_to_integer_config(cfg.odisseo_config.integrator),
#                                 fixed_timestep= cfg.odisseo_config.fixed_timestep,
#                                 diffrax_solver= convert_to_integer_config(cfg.odisseo_config.diffrax_solver),
#                                 glorder=cfg.odisseo_config.glorder) #default values
        
    
#     elif cfg.simulator == "galax":
#         import jax
#         import jax.numpy as jnp
#         from utils.utils_galax_simulator import simulate_stream_galax

#         simulate_stream = simulate_stream_galax
#         config = cfg.galax_config
#         code_units = None #galax does not use code units, but we need to pass something to the function

#     elif cfg.simulator == "StreaMax":
#         import jax
#         import jax.numpy as jnp
#         from utils.utils_StreaMax_simulator import simulate_stream_StreaMAX

#         simulate_stream = simulate_stream_StreaMAX
#         config = cfg.streamax_config
#         code_units = None #StreaMax does not use code units, but we need to pass something to the function
    
#     elif cfg.simulator == "gala":
#         from joblib import Parallel, delayed
#         from utils.utils_gala_simulator import simulate_stream_gala, _run_gala_single
#         config = cfg.gala_config
#         code_units = None #gala does not use code units, but we need to pass something to the function
        
    
#     save_dict = {'sim_data_carthesian': np.ones((total_n_simulations* len(cfg.target_streams), cfg.odisseo_config.N_particles+2, 6)),
#                  'sim_data_projected': np.ones((total_n_simulations* len(cfg.target_streams),cfg.odisseo_config.N_particles+2, 6))}
#     for batch_start in tqdm(range(0, total_n_simulations * len(cfg.target_streams), cfg.batch_size)):
#         batch_end = min(batch_start + cfg.batch_size, total_n_simulations * len(cfg.target_streams))
#         batch_indices = np.arange(batch_start, batch_end)
#         # Prepare batch of parameters
#         if (cfg.simulator == "odisseo")|(cfg.simulator == "galax")|(cfg.simulator == "StreaMax"):
#             batch_params = {k: jnp.array(v[batch_indices]) for k, v in prior_samples.items()}
#             sim_data_batch = jax.vmap(simulate_stream, in_axes=(0, None, None, 0))(batch_params, config, code_units, jnp.array(batch_indices))  # shape (batch_size, ...)
#             if cfg.simulator == "StreaMax":
#                 n_particles_subsample = 1000
#                 sim_data_clean = []
#                 for sim in sim_data_batch:
#                     mask = ~np.isnan(sim).any(axis=1)
#                     sim_no_nan = sim[mask]
#                     n_timesteps_clean = sim_no_nan.shape[0]
#                     # Subsample timesteps
#                     if n_timesteps_clean >= n_particles_subsample:
#                         indices = np.random.choice(n_timesteps_clean, size=n_particles_subsample, replace=False)
#                         indices.sort()
#                         sim_subsampled = sim_no_nan[indices]
#                     else:
#                         # If not enough timesteps, pad with NaNs
#                         pad = np.full((n_particles_subsample - n_timesteps_clean, sim_no_nan.shape[1]), np.nan)
#                         sim_subsampled = np.vstack([sim_no_nan, pad])
#                     sim_data_clean.append(sim_subsampled)
#                 sim_data_batch = np.stack(sim_data_clean, axis=0)
#             else:
#                 pass
#         elif cfg.simulator == "gala":
#             batch_params = {k: np.array(v[batch_indices]) for k, v in prior_samples.items()}
    
#             # Convert to individual parameter dicts
#             individual_params = [
#                 {k: v[i] for k, v in batch_params.items()}
#                 for i in range(len(batch_indices))
#             ]
            
#             # Run in parallel with joblib
#             results = Parallel(n_jobs=config.n_workers)(
#                 delayed(simulate_stream_gala)(
#                     param_dict, config, code_units, int(batch_indices[i])
#                 )
#                 for i, param_dict in enumerate(individual_params)
#             )
            
#             # Stack results
#             sim_data_batch = np.stack(results, axis=0)

#         # Save each simulation in the batch
#         sim_data_projected_batch = sky_projection_astropy(sim_data_batch)
#         print('sim_data_batch shape before saving:', sim_data_batch.shape)
#         save_dict['sim_data_carthesian'][batch_indices] = sim_data_batch
#         save_dict['sim_data_projected'][batch_indices] = sim_data_projected_batch

#     print('sim_data_batch shape:', sim_data_batch.shape)
    
#     save_dict['sim_data_carthesian'] = save_dict['sim_data_carthesian'].reshape(total_n_simulations, len(cfg.target_streams), cfg.odisseo_config.N_particles+2, 6)
#     save_dict['sim_data_projected'] = save_dict['sim_data_projected'].reshape(total_n_simulations, len(cfg.target_streams), cfg.odisseo_config.N_particles+2, 6)
#     for k in prior_samples.keys():
#         if k in cfg.priors_global.keys():
#             prior_samples[k] = prior_samples[k].reshape(-1, len(cfg.target_streams),  1)[:, 0, :] # reshape back to (n_samples, 1) for global parameters
#         else:
#             prior_samples[k] = prior_samples[k].reshape(-1, len(cfg.target_streams),  1) 
#     save_dict.update(prior_samples)
#     print(' The shapes of the simulations and parameters before saving are :', {k: v.shape for k, v in save_dict.items()})
#     np.savez(os.path.join(cfg.base_dir, cfg.data_dir, f'simulation_multistream_{total_n_simulations}.npz'), **save_dict)
    
# if __name__ == "__main__":
#     main()



import os
import omegaconf
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.core.config_store import ConfigStore
from astropy import units as u
import numpy as np
from utils.utils_simulate import sample_parameters_parallel, sky_projection_astropy
from config.SimulatorConfig import SimulatorConfig

cs = ConfigStore.instance()
cs.store(name="simulator_config", node=SimulatorConfig)

# ── PPC mode ───────────────────────────────────────────────────────────────────
# Set N_POSTERIOR_SAMPLES = 1 to use the posterior mean.
# Set N_POSTERIOR_SAMPLES > 1 to draw that many samples from the posterior,
# producing N_POSTERIOR_SAMPLES simulations in total.
N_POSTERIOR_SAMPLES = 10   # <── change this
# ──────────────────────────────────────────────────────────────────────────────


def _draw_posterior_values(posterior_dict, param_names, n_samples, use_mean):
    """
    Returns a dict {param: array of shape (n_samples,)}.
    If use_mean=True, every entry is the posterior mean repeated n_samples times.
    If use_mean=False, draws n_samples indices (with replacement) from the posterior.
    """
    # Flatten each param to 1-D pool of samples
    flat = {p: np.asarray(posterior_dict[p]).reshape(-1) for p in param_names if p in posterior_dict}

    if use_mean:
        return {p: np.full(n_samples, flat[p].mean()) for p in flat}
    else:
        # Draw a shared set of indices so global params stay correlated across draws
        pool_size = min(v.shape[0] for v in flat.values())
        idx = np.random.choice(pool_size, size=n_samples, replace=(n_samples > pool_size))
        return {p: flat[p][idx] for p in flat}


def _draw_local_posterior_values(local_posterior, param_names, stream_names, n_samples, use_mean):
    """
    Returns a dict {stream_name: {param: array of shape (n_samples,)}}.
    local_posterior[param] is reshaped to (n_streams, N_flat) before sampling.
    """
    n_streams = len(stream_names)
    result = {s: {} for s in stream_names}

    # Collect per-stream flat arrays
    per_stream = {s: {} for s in stream_names}
    for param in param_names:
        if param not in local_posterior:
            continue
        arr = np.asarray(local_posterior[param]).reshape(n_streams, -1)  # (N_STREAMS, N_SAMPLES)
        for s_idx, stream in enumerate(stream_names):
            per_stream[stream][param] = arr[s_idx]

    for stream in stream_names:
        if not per_stream[stream]:
            continue
        if use_mean:
            for param, vals in per_stream[stream].items():
                result[stream][param] = np.full(n_samples, vals.mean())
        else:
            pool_size = min(v.shape[0] for v in per_stream[stream].values())
            idx = np.random.choice(pool_size, size=n_samples, replace=(n_samples > pool_size))
            for param, vals in per_stream[stream].items():
                result[stream][param] = vals[idx]

    return result


@hydra.main(version_base=None, config_path="config", config_name="posterior_predictive_check_config")
def main(cfg: SimulatorConfig):
    os.makedirs(os.path.join(cfg.base_dir, cfg.data_dir), exist_ok=True)
    print('Using simulator:', cfg.simulator)

    if cfg.simulator in ("odisseo", "galax", "StreaMax"):
        from autocvd import autocvd
        autocvd(num_gpus=1)
    elif cfg.simulator == "gala":
        print(cfg.gala_config)

    if isinstance(cfg.n_simulations, omegaconf.listconfig.ListConfig):
        index_sim_start = cfg.n_simulations[0]
        index_sim_end   = cfg.n_simulations[1]
    else:
        index_sim_start = 0
        index_sim_end   = cfg.n_simulations

    rng_seed = index_sim_end
    np.random.seed(rng_seed)

    # ── Load posteriors ────────────────────────────────────────────────────────
    global_posterior_path = os.path.join(
        cfg.base_dir,
        '../plots/gala6D_aug/new_hyper/model54_60k_1000epochs/global_posterior.npz'
    )
    local_posterior_path = os.path.join(
        cfg.base_dir,
        '../plots/plots_local/gala6D_aug/new_hyper/model54_60k_1000epochs/gaia_local_posterior.npz'
    )
    print('Loading global posterior from:', global_posterior_path)
    global_posterior = dict(np.load(global_posterior_path, allow_pickle=True))
    print('Loading local posterior from:', local_posterior_path)
    local_posterior  = dict(np.load(local_posterior_path,  allow_pickle=True))

    stream_names = list(cfg.target_streams.keys())
    n_streams    = len(stream_names)
    use_mean     = (N_POSTERIOR_SAMPLES == 1)
    n_ppc        = N_POSTERIOR_SAMPLES

    print(f'\nPPC mode: {"posterior mean (1 simulation)" if use_mean else f"{n_ppc} posterior samples"}')

    # Draw global and local parameter values for all PPC runs at once
    global_draws = _draw_posterior_values(
        global_posterior, list(cfg.priors_global.keys()), n_ppc, use_mean
    )
    local_draws = _draw_local_posterior_values(
        local_posterior, ['vr', 'r', 'mu_ra_cosdec', 'mu_dec'], stream_names, n_ppc, use_mean
    )

    # ── Simulator setup (done once) ───────────────────────────────────────────
    if cfg.simulator == "odisseo":
        import jax, jax.numpy as jnp
        from odisseo.option_classes import SimulationConfig
        from odisseo.units import CodeUnits
        from utils.utils_odisseo_simulator import (convert_to_integer_externalacc,
                                                    convert_to_integer_config,
                                                    simulate_stream_odisseo)
        simulate_stream = simulate_stream_odisseo
        code_units = CodeUnits(1*u.kpc, 1e3*u.Msun, G=1, unit_time=1*u.Gyr)
        config = SimulationConfig(
            N_particles=cfg.odisseo_config.N_particles,
            return_snapshots=cfg.odisseo_config.return_snapshots,
            num_snapshots=cfg.odisseo_config.num_snapshots,
            num_timesteps=cfg.odisseo_config.num_timesteps,
            external_accelerations=convert_to_integer_externalacc(cfg.odisseo_config.external_accelerations),
            acceleration_scheme=convert_to_integer_config(cfg.odisseo_config.acceleration_scheme),
            softening=(cfg.odisseo_config.softening*u.pc).to(code_units.code_length).value,
            integrator=convert_to_integer_config(cfg.odisseo_config.integrator),
            fixed_timestep=cfg.odisseo_config.fixed_timestep,
            diffrax_solver=convert_to_integer_config(cfg.odisseo_config.diffrax_solver),
            glorder=cfg.odisseo_config.glorder,
        )
    elif cfg.simulator == "galax":
        import jax, jax.numpy as jnp
        from utils.utils_galax_simulator import simulate_stream_galax
        simulate_stream = simulate_stream_galax
        config = cfg.galax_config
        code_units = None
    elif cfg.simulator == "StreaMax":
        import jax, jax.numpy as jnp
        from utils.utils_StreaMax_simulator import simulate_stream_StreaMAX
        simulate_stream = simulate_stream_StreaMAX
        config = cfg.streamax_config
        code_units = None
    elif cfg.simulator == "gala":
        from joblib import Parallel, delayed
        from utils.utils_gala_simulator import simulate_stream_gala
        config = cfg.gala_config
        code_units = None

    # ── Loop over PPC runs ────────────────────────────────────────────────────
    all_prior_samples   = []
    all_sim_carthesian  = []
    all_sim_projected   = []

    for ppc_idx in tqdm(range(n_ppc), desc='PPC runs'):

        # Override priors for this draw
        with open_dict(cfg):
            for param in list(cfg.priors_global.keys()):
                if param in global_draws:
                    cfg.priors_global[param] = {
                        'type': 'identity',
                        'prior_parameters': [float(global_draws[param][ppc_idx])],
                    }
            for stream in stream_names:
                for param, vals in local_draws[stream].items():
                    cfg.priors_local[stream][param] = {
                        'type': 'identity',
                        'prior_parameters': [float(vals[ppc_idx])],
                    }

        # Sample parameters (n_simulations=1 per PPC draw)
        prior_samples = sample_parameters_parallel(
            prior_global_dict=cfg.priors_global,
            prior_local_dict=cfg.priors_local,
            n_samples=1,
            target_streams=cfg.target_streams,
            key_seed=rng_seed + ppc_idx,
        )

        # Broadcast global params and flatten
        for k in cfg.priors_global.keys():
            prior_samples[k] = np.repeat(prior_samples[k][:, np.newaxis, :], n_streams, axis=1)
        for k in prior_samples.keys():
            prior_samples[k] = prior_samples[k].reshape(-1, 1)

        # Simulate (1 simulation * n_streams batches)
        if cfg.simulator in ("odisseo", "galax", "StreaMax"):
            batch_params   = {k: jnp.array(v) for k, v in prior_samples.items()}
            sim_data_batch = jax.vmap(simulate_stream, in_axes=(0, None, None, 0))(
                batch_params, config, code_units, jnp.arange(n_streams)
            )
            sim_data_batch = np.asarray(sim_data_batch)
        elif cfg.simulator == "gala":
            individual_params = [{k: v[i] for k, v in prior_samples.items()} for i in range(n_streams)]
            results = Parallel(n_jobs=config.n_workers)(
                delayed(simulate_stream_gala)(p, config, code_units, i)
                for i, p in enumerate(individual_params)
            )
            sim_data_batch = np.stack(results, axis=0)

        sim_data_projected_batch = sky_projection_astropy(sim_data_batch)

        # Reshape back: (n_streams, N_particles, 6) -> (1, n_streams, N_particles, 6)
        n_particles = sim_data_batch.shape[1]
        all_sim_carthesian.append(sim_data_batch.reshape(1, n_streams, n_particles, 6))
        all_sim_projected.append(sim_data_projected_batch.reshape(1, n_streams, n_particles, 6))

        # Reshape prior samples back: flatten -> (1, n_streams, 1) or (1, 1)
        ps_reshaped = {}
        for k, v in prior_samples.items():
            if k in cfg.priors_global.keys():
                ps_reshaped[k] = v.reshape(1, n_streams, 1)[:, 0, :]   # (1, 1)
            else:
                ps_reshaped[k] = v.reshape(1, n_streams, 1)             # (1, n_streams, 1)
        all_prior_samples.append(ps_reshaped)

    # ── Concatenate all PPC runs ──────────────────────────────────────────────
    save_dict = {
        'sim_data_carthesian': np.concatenate(all_sim_carthesian, axis=0),   # (n_ppc, n_streams, N_p, 6)
        'sim_data_projected':  np.concatenate(all_sim_projected,  axis=0),   # (n_ppc, n_streams, N_p, 6)
    }
    for k in all_prior_samples[0].keys():
        save_dict[k] = np.concatenate([ps[k] for ps in all_prior_samples], axis=0)

    print('\nShapes before saving:')
    for k, v in save_dict.items():
        print(f'  {k}: {v.shape}')

    out_path = os.path.join(cfg.base_dir, cfg.data_dir, f'ppc_{n_ppc}samples.npz')
    np.savez(out_path, **save_dict)
    print(f'\nSaved to {out_path}')


if __name__ == "__main__":
    main()