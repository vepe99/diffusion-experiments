
# import os
# import omegaconf
# from tqdm import tqdm
# from omegaconf import DictConfig, OmegaConf, open_dict
# import hydra
# from hydra.core.config_store import ConfigStore
# from astropy import units as u
# import numpy as np
# from utils.utils_simulate import sample_parameters_parallel, sky_projection_astropy
# from config.SimulatorConfig import SimulatorConfig

# cs = ConfigStore.instance()
# cs.store(name="simulator_config", node=SimulatorConfig)

# # ── PPC mode ───────────────────────────────────────────────────────────────────
# # Set N_POSTERIOR_SAMPLES = 1 to use the posterior mean.
# # Set N_POSTERIOR_SAMPLES > 1 to draw that many samples from the posterior,
# # producing N_POSTERIOR_SAMPLES simulations in total.
# N_POSTERIOR_SAMPLES = 1 # <── change this
# # ──────────────────────────────────────────────────────────────────────────────


# def _draw_posterior_values(posterior_dict, param_names, n_samples, use_mean, idx=None):
#     flat = {p: np.asarray(posterior_dict[p]).reshape(-1) for p in param_names if p in posterior_dict}
#     if use_mean:
#         return {p: np.full(n_samples, flat[p].mean()) for p in flat}
#     return {p: flat[p][idx] for p in flat}


# def _draw_local_posterior_values(local_posterior, param_names, stream_names, n_samples, use_mean, idx=None):
#     n_streams = len(stream_names)
#     result = {s: {} for s in stream_names}
#     per_stream = {s: {} for s in stream_names}
#     for param in param_names:
#         if param not in local_posterior:
#             continue
#         arr = np.asarray(local_posterior[param]).reshape(n_streams, -1)  # (N_STREAMS, N_flat)
#         for s_idx, stream in enumerate(stream_names):
#             per_stream[stream][param] = arr[s_idx]
#     for stream in stream_names:
#         if not per_stream[stream]:
#             continue
#         if use_mean:
#             for param, vals in per_stream[stream].items():
#                 result[stream][param] = np.full(n_samples, vals.mean())
#         else:
#             for param, vals in per_stream[stream].items():
#                 result[stream][param] = vals[idx]
#     return result


# @hydra.main(version_base=None, config_path="config", config_name="posterior_predictive_check_config")
# def main(cfg: SimulatorConfig):
#     os.makedirs(os.path.join(cfg.base_dir, cfg.data_dir), exist_ok=True)
#     print('Using simulator:', cfg.simulator)

#     if cfg.simulator in ("odisseo", "galax", "StreaMax"):
#         from autocvd import autocvd
#         autocvd(num_gpus=1)
#     elif cfg.simulator == "gala":
#         print(cfg.gala_config)

#     if isinstance(cfg.n_simulations, omegaconf.listconfig.ListConfig):
#         index_sim_start = cfg.n_simulations[0]
#         index_sim_end   = cfg.n_simulations[1]
#     else:
#         index_sim_start = 0
#         index_sim_end   = cfg.n_simulations

#     rng_seed = index_sim_end
#     np.random.seed(rng_seed)

#     # ── Load posteriors ────────────────────────────────────────────────────────
#     global_posterior_path = os.path.join(
#         cfg.base_dir,
#         '../plots/gala6D_aug/new_hyper/model54_60k_1000epochs/global_posterior.npz'
#     )
#     local_posterior_path = os.path.join(
#         cfg.base_dir,
#         '../hyperparameter_tuning/gala/local/jonas/streamnomr_standardize/model_2/gaia/gaia_local_posterior.npz'
#     )
#     print('Loading global posterior from:', global_posterior_path)
#     global_posterior = dict(np.load(global_posterior_path, allow_pickle=True))
#     print('Loading local posterior from:', local_posterior_path)
#     local_posterior  = dict(np.load(local_posterior_path,  allow_pickle=True))

#     stream_names = list(cfg.target_streams.keys())
#     n_streams    = len(stream_names)
#     use_mean = (N_POSTERIOR_SAMPLES == 1)
#     n_ppc    = N_POSTERIOR_SAMPLES

#     # Draw ONE shared index vector so global and local samples stay correlated
#     if not use_mean:
#         # Use the minimum pool size across all posterior params to be safe
#         all_arrays = (
#             [np.asarray(global_posterior[p]).reshape(-1) for p in cfg.priors_global.keys() if p in global_posterior] +
#             [np.asarray(local_posterior[p]).reshape(-1) for p in ['vr', 'r', 'mu_ra_cosdec', 'mu_dec'] if p in local_posterior]
#         )
#         pool_size = min(a.shape[0] for a in all_arrays)
#         shared_idx = np.random.choice(pool_size, size=n_ppc, replace=(n_ppc > pool_size))
#         print(f'Drew {n_ppc} shared indices from pool of size {pool_size}')
#     else:
#         shared_idx = None

#     global_draws = _draw_posterior_values(
#         global_posterior, list(cfg.priors_global.keys()), n_ppc, use_mean, idx=shared_idx
#     )
#     local_draws = _draw_local_posterior_values(
#         local_posterior, ['vr', 'r', 'mu_ra_cosdec', 'mu_dec'], stream_names, n_ppc, use_mean, idx=shared_idx
#     )

#     # ── Simulator setup (done once) ───────────────────────────────────────────
#     if cfg.simulator == "odisseo":
#         import jax, jax.numpy as jnp
#         from odisseo.option_classes import SimulationConfig
#         from odisseo.units import CodeUnits
#         from utils.utils_odisseo_simulator import (convert_to_integer_externalacc,
#                                                     convert_to_integer_config,
#                                                     simulate_stream_odisseo)
#         simulate_stream = simulate_stream_odisseo
#         code_units = CodeUnits(1*u.kpc, 1e3*u.Msun, G=1, unit_time=1*u.Gyr)
#         config = SimulationConfig(
#             N_particles=cfg.odisseo_config.N_particles,
#             return_snapshots=cfg.odisseo_config.return_snapshots,
#             num_snapshots=cfg.odisseo_config.num_snapshots,
#             num_timesteps=cfg.odisseo_config.num_timesteps,
#             external_accelerations=convert_to_integer_externalacc(cfg.odisseo_config.external_accelerations),
#             acceleration_scheme=convert_to_integer_config(cfg.odisseo_config.acceleration_scheme),
#             softening=(cfg.odisseo_config.softening*u.pc).to(code_units.code_length).value,
#             integrator=convert_to_integer_config(cfg.odisseo_config.integrator),
#             fixed_timestep=cfg.odisseo_config.fixed_timestep,
#             diffrax_solver=convert_to_integer_config(cfg.odisseo_config.diffrax_solver),
#             glorder=cfg.odisseo_config.glorder,
#         )
#     elif cfg.simulator == "galax":
#         import jax, jax.numpy as jnp
#         from utils.utils_galax_simulator import simulate_stream_galax
#         simulate_stream = simulate_stream_galax
#         config = cfg.galax_config
#         code_units = None
#     elif cfg.simulator == "StreaMax":
#         import jax, jax.numpy as jnp
#         from utils.utils_StreaMax_simulator import simulate_stream_StreaMAX
#         simulate_stream = simulate_stream_StreaMAX
#         config = cfg.streamax_config
#         code_units = None
#     elif cfg.simulator == "gala":
#         from joblib import Parallel, delayed
#         from utils.utils_gala_simulator import simulate_stream_gala
#         config = cfg.gala_config
#         code_units = None

#     # ── Loop over PPC runs ────────────────────────────────────────────────────
#     all_prior_samples   = []
#     all_sim_carthesian  = []
#     all_sim_projected   = []

#     for ppc_idx in tqdm(range(n_ppc), desc='PPC runs'):

#         # Override priors for this draw
#         with open_dict(cfg):
#             for param in list(cfg.priors_global.keys()):
#                 if param in global_draws:
#                     cfg.priors_global[param] = {
#                         'type': 'identity',
#                         'prior_parameters': [float(global_draws[param][ppc_idx])],
#                     }
#             for stream in stream_names:
#                 for param, vals in local_draws[stream].items():
#                     cfg.priors_local[stream][param] = {
#                         'type': 'identity',
#                         'prior_parameters': [float(vals[ppc_idx])],
#                     }

#         # Sample parameters (n_simulations=1 per PPC draw)
#         prior_samples = sample_parameters_parallel(
#             prior_global_dict=cfg.priors_global,
#             prior_local_dict=cfg.priors_local,
#             n_samples=1,
#             target_streams=cfg.target_streams,
#             key_seed=rng_seed + ppc_idx,
#         )

#         # Broadcast global params and flatten
#         for k in cfg.priors_global.keys():
#             prior_samples[k] = np.repeat(prior_samples[k][:, np.newaxis, :], n_streams, axis=1)
#         for k in prior_samples.keys():
#             prior_samples[k] = prior_samples[k].reshape(-1, 1)

#         # Simulate (1 simulation * n_streams batches)
#         if cfg.simulator in ("odisseo", "galax", "StreaMax"):
#             batch_params   = {k: jnp.array(v) for k, v in prior_samples.items()}
#             sim_data_batch = jax.vmap(simulate_stream, in_axes=(0, None, None, 0))(
#                 batch_params, config, code_units, jnp.arange(n_streams)
#             )
#             sim_data_batch = np.asarray(sim_data_batch)
#         elif cfg.simulator == "gala":
#             individual_params = [{k: v[i] for k, v in prior_samples.items()} for i in range(n_streams)]
#             results = Parallel(n_jobs=config.n_workers)(
#                 delayed(simulate_stream_gala)(p, config, code_units, i)
#                 for i, p in enumerate(individual_params)
#             )
#             sim_data_batch = np.stack(results, axis=0)

#         sim_data_projected_batch = sky_projection_astropy(sim_data_batch)

#         # Reshape back: (n_streams, N_particles, 6) -> (1, n_streams, N_particles, 6)
#         n_particles = sim_data_batch.shape[1]
#         all_sim_carthesian.append(sim_data_batch.reshape(1, n_streams, n_particles, 6))
#         all_sim_projected.append(sim_data_projected_batch.reshape(1, n_streams, n_particles, 6))

#         # Reshape prior samples back: flatten -> (1, n_streams, 1) or (1, 1)
#         ps_reshaped = {}
#         for k, v in prior_samples.items():
#             if k in cfg.priors_global.keys():
#                 ps_reshaped[k] = v.reshape(1, n_streams, 1)[:, 0, :]   # (1, 1)
#             else:
#                 ps_reshaped[k] = v.reshape(1, n_streams, 1)             # (1, n_streams, 1)
#         all_prior_samples.append(ps_reshaped)

#     # ── Concatenate all PPC runs ──────────────────────────────────────────────
#     save_dict = {
#         'sim_data_carthesian': np.concatenate(all_sim_carthesian, axis=0),   # (n_ppc, n_streams, N_p, 6)
#         'sim_data_projected':  np.concatenate(all_sim_projected,  axis=0),   # (n_ppc, n_streams, N_p, 6)
#     }
#     for k in all_prior_samples[0].keys():
#         save_dict[k] = np.concatenate([ps[k] for ps in all_prior_samples], axis=0)

#     print('\nShapes before saving:')
#     for k, v in save_dict.items():
#         print(f'  {k}: {v.shape}')

#     out_path = os.path.join(cfg.base_dir, cfg.data_dir, f'ppc_{n_ppc}samples.npz')
#     np.savez(out_path, **save_dict)
#     print(f'\nSaved to {out_path}')


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
from scipy.stats import gaussian_kde
from utils.utils_simulate import sample_parameters_parallel, sky_projection_astropy
from config.SimulatorConfig import SimulatorConfig

cs = ConfigStore.instance()
cs.store(name="simulator_config", node=SimulatorConfig)

# ── PPC mode ───────────────────────────────────────────────────────────────────
# Set N_POSTERIOR_SAMPLES = 1 to use the posterior MODE (via KDE).
# Set N_POSTERIOR_SAMPLES > 1 to draw that many samples from the posterior,
# producing N_POSTERIOR_SAMPLES simulations in total.
N_POSTERIOR_SAMPLES = 1000  # <── change this
# ──────────────────────────────────────────────────────────────────────────────


# ── Helpers ────────────────────────────────────────────────────────────────────

def _kde_mode(samples: np.ndarray, n_grid: int = 2000) -> float:
    """Estimate the mode of a 1-D continuous distribution via KDE."""
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError("Cannot estimate mode of an empty / all-NaN array.")
    if samples.size == 1:
        return float(samples[0])
    kde = gaussian_kde(samples)
    x = np.linspace(samples.min(), samples.max(), n_grid)
    return float(x[np.argmax(kde(x))])


def _draw_posterior_values(posterior_dict, param_names, n_samples, use_mode, idx=None):
    flat = {p: np.asarray(posterior_dict[p]).reshape(-1) for p in param_names if p in posterior_dict}
    if use_mode:
        return {p: np.full(n_samples, _kde_mode(flat[p])) for p in flat}
    return {p: flat[p][idx] for p in flat}


def _draw_local_posterior_values(local_posterior, param_names, stream_names, n_samples, use_mode, idx=None):
    n_streams = len(stream_names)
    result = {s: {} for s in stream_names}
    per_stream = {s: {} for s in stream_names}
    for param in param_names:
        if param not in local_posterior:
            continue
        arr = np.asarray(local_posterior[param]).reshape(n_streams, -1)  # (N_STREAMS, N_flat)
        for s_idx, stream in enumerate(stream_names):
            per_stream[stream][param] = arr[s_idx]
    for stream in stream_names:
        if not per_stream[stream]:
            continue
        if use_mode:
            for param, vals in per_stream[stream].items():
                result[stream][param] = np.full(n_samples, _kde_mode(vals))
        else:
            for param, vals in per_stream[stream].items():
                result[stream][param] = vals[idx]
    return result


# ── ChainConsumer cornerplot ───────────────────────────────────────────────────

def _plot_cornerplot(global_posterior, local_posterior, cfg, stream_names, out_dir):
    """
    Plot a ChainConsumer cornerplot of all global and per-stream local parameters.

    Global parameters are shared across streams; local parameters are anchored
    to individual streams and are labelled as ``<param>_<stream_name>``.

    The combined chain columns are therefore:
        [*global_params, vr_S0, r_S0, mu_ra_cosdec_S0, mu_dec_S0,
                         vr_S1, r_S1, ...]
    """
    try:
        from chainconsumer import ChainConsumer, Chain
        import pandas as pd
        _new_api = True
    except ImportError:
        from chainconsumer import ChainConsumer
        _new_api = False

    LOCAL_PARAMS = ['vr', 'r', 'mu_ra_cosdec', 'mu_dec']
    n_streams = len(stream_names)

    # ── Collect global samples ────────────────────────────────────────────────
    global_param_names = list(cfg.priors_global.keys())
    chain_dict = {}
    n_global = None

    for p in global_param_names:
        if p not in global_posterior:
            continue
        arr = np.asarray(global_posterior[p]).reshape(-1)
        chain_dict[p] = arr
        if n_global is None:
            n_global = len(arr)

    if n_global is None:
        raise RuntimeError("No global posterior parameters found – cannot build cornerplot.")

    # ── Collect local samples (per stream) ───────────────────────────────────
    for p in LOCAL_PARAMS:
        if p not in local_posterior:
            continue
        arr = np.asarray(local_posterior[p]).reshape(n_streams, -1)  # (n_streams, N_flat)
        for s_idx, stream in enumerate(stream_names):
            label = f'{p}_{stream}'
            chain_dict[label] = arr[s_idx]  # length may differ from global

    # ── Align lengths to the shortest chain ──────────────────────────────────
    min_len = min(len(v) for v in chain_dict.values())
    chain_dict = {k: v[:min_len] for k, v in chain_dict.items()}

    # ── Pretty LaTeX labels ───────────────────────────────────────────────────
    LATEX = {
        'vr':           r'$v_r$',
        'r':            r'$r$',
        'mu_ra_cosdec': r'$\mu_{\alpha*}$',
        'mu_dec':       r'$\mu_\delta$',
    }

    def _label(name):
        # local param with stream suffix: "vr_GD1" -> "$v_r$ GD1"
        for p in LOCAL_PARAMS:
            if name.startswith(f'{p}_'):
                stream = name[len(p) + 1:]
                return f'{LATEX.get(p, p)} {stream}'
        return name

    rename_map = {k: _label(k) for k in chain_dict}
    import pandas as pd
    df = pd.DataFrame(chain_dict).rename(columns=rename_map)

    # ── Build and plot ────────────────────────────────────────────────────────
    if _new_api:
        c = ChainConsumer()
        c.add_chain(Chain(samples=df, name="Posterior"))
        fig = c.plotter.plot()
    else:
        # Legacy ChainConsumer < 1.0
        samples_array = df.values
        c = ChainConsumer()
        c.add_chain(samples_array, parameters=list(df.columns), name="Posterior")
        fig = c.plotter.plot()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'posterior_cornerplot.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved cornerplot to {out_path}')
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

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
        '../plots/gala6D/new_hyper/model54_60k_1000epochs/global_posterior.npz'
    )
    local_posterior_path = os.path.join(
        cfg.base_dir,
        '../hyperparameter_tuning/gala/local/jonas/streamnomr_standardize/model_2/gaia/gaia_local_posterior.npz'
    )
    print('Loading global posterior from:', global_posterior_path)
    global_posterior = dict(np.load(global_posterior_path, allow_pickle=True))
    print('Loading local posterior from:', local_posterior_path)
    local_posterior  = dict(np.load(local_posterior_path,  allow_pickle=True))

    stream_names = list(cfg.target_streams.keys())
    n_streams    = len(stream_names)
    use_mode = (N_POSTERIOR_SAMPLES == 1)   # ← was use_mean; now uses KDE mode
    n_ppc    = N_POSTERIOR_SAMPLES

    # ── Cornerplot (done once, before simulation loop) ────────────────────────
    plot_dir = os.path.join(cfg.base_dir, cfg.data_dir)
    print('Generating posterior cornerplot...')
    _plot_cornerplot(global_posterior, local_posterior, cfg, stream_names, plot_dir)

    # Draw ONE shared index vector so global and local samples stay correlated
    if not use_mode:
        all_arrays = (
            [np.asarray(global_posterior[p]).reshape(-1)
             for p in cfg.priors_global.keys() if p in global_posterior] +
            [np.asarray(local_posterior[p]).reshape(-1)
             for p in ['vr', 'r', 'mu_ra_cosdec', 'mu_dec'] if p in local_posterior]
        )
        pool_size = min(a.shape[0] for a in all_arrays)
        shared_idx = np.random.choice(pool_size, size=n_ppc, replace=(n_ppc > pool_size))
        print(f'Drew {n_ppc} shared indices from pool of size {pool_size}')
    else:
        shared_idx = None

    global_draws = _draw_posterior_values(
        global_posterior, list(cfg.priors_global.keys()), n_ppc, use_mode, idx=shared_idx
    )
    local_draws = _draw_local_posterior_values(
        local_posterior, ['vr', 'r', 'mu_ra_cosdec', 'mu_dec'], stream_names, n_ppc, use_mode, idx=shared_idx
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