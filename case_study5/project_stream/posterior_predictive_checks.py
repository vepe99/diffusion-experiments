
import os
import omegaconf
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.core.config_store import ConfigStore
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from utils.utils_simulate import sample_parameters_parallel, sky_projection_astropy
from config.SimulatorConfig import SimulatorConfig
import gala
from gala.units import galactic
import gala.potential as gp



cs = ConfigStore.instance()
cs.store(name="simulator_config", node=SimulatorConfig)

# ── Sampling mode ──────────────────────────────────────────────────────────────
# Three mutually exclusive modes, selected by N_POSTERIOR_SAMPLES and
# QUANTILE_SAMPLING:
#
#   N_POSTERIOR_SAMPLES == 1, QUANTILE_SAMPLING = False
#       → posterior MODE via KDE (original behaviour)
#
#   N_POSTERIOR_SAMPLES > 1,  QUANTILE_SAMPLING = False
#       → uniform random draw from the full joint posterior (original behaviour)
#
#   N_POSTERIOR_SAMPLES > 1,  QUANTILE_SAMPLING = True
#       → draw only from the central [QUANTILE_LOW, QUANTILE_HIGH] band of the
#         joint posterior, measured by Mahalanobis distance.  The band can be
#         asymmetric (e.g. 0.0–0.84 gives the inner 84 %).
#
# QUANTILE_LOW / QUANTILE_HIGH are the CDF-level bounds on the Mahalanobis
# distance percentile, so (0.16, 0.84) keeps samples between the 16th and 84th
# percentile of the joint distance distribution — equivalent to the ±1σ band
# for a Gaussian posterior.
# ──────────────────────────────────────────────────────────────────────────────
N_POSTERIOR_SAMPLES = 500  # <── total PPC runs
QUANTILE_SAMPLING   = True   # <── set False to fall back to random / mode
QUANTILE_LOW        = 0.16   # <── lower CDF bound (0.0 = include everything below median)
QUANTILE_HIGH       = 0.84  # <── upper CDF bound
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


def _build_joint_matrix(posterior_dict, param_names):
    """
    Stack all requested parameters into a single (N_samples, D) matrix,
    handling the case where arrays may have leading stream dimensions by
    flattening per-stream slices separately and then concatenating columns.

    Returns
    -------
    mat : np.ndarray, shape (N, D)
    names : list[str]   column names (for diagnostics)
    """
    cols, col_names = [], []
    for p in param_names:
        if p not in posterior_dict:
            continue
        arr = np.asarray(posterior_dict[p]).reshape(-1)   # 1-D
        arr = arr[np.isfinite(arr)]
        cols.append(arr)
        col_names.append(p)

    # Align to the shortest column
    min_len = min(len(c) for c in cols)
    mat = np.column_stack([c[:min_len] for c in cols])   # (N, D)
    return mat, col_names


def _mahalanobis_distances(mat: np.ndarray) -> np.ndarray:
    """
    Compute the Mahalanobis distance of every row from the sample mean.

    If the covariance matrix is singular (e.g. D > N or degenerate parameters),
    falls back to standardised Euclidean distance.

    Parameters
    ----------
    mat : (N, D) array of posterior samples

    Returns
    -------
    dists : (N,) array of non-negative distances
    """
    mean = mat.mean(axis=0)
    centred = mat - mean
    cov = np.cov(mat, rowvar=False)

    if mat.shape[1] == 1:
        # Scalar case: just use standardised distance
        std = np.sqrt(cov) if cov.ndim == 0 else np.sqrt(cov[0, 0])
        return np.abs(centred[:, 0]) / (std + 1e-30)

    try:
        cov_inv = np.linalg.inv(cov)
        dists = np.sqrt(np.einsum('ni,ij,nj->n', centred, cov_inv, centred))
    except np.linalg.LinAlgError:
        # Fallback: standardised Euclidean
        stds = mat.std(axis=0) + 1e-30
        dists = np.sqrt(((centred / stds) ** 2).sum(axis=1))

    return dists



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



def _marginal_quantile_mask(
    param_arrays: dict,          # {name: 1-D np.ndarray}, all same length
    q_low: float,
    q_high: float,
) -> np.ndarray:
    """
    Boolean mask: True for samples where EVERY parameter falls within its own
    [q_low, q_high] marginal quantile range simultaneously.
    """
    mask = np.ones(len(next(iter(param_arrays.values()))), dtype=bool)
    for arr in param_arrays.values():
        lo = np.quantile(arr, q_low)
        hi = np.quantile(arr, q_high)
        mask &= (arr >= lo) & (arr <= hi)
    return mask


def _select_quantile_indices(
    global_posterior: dict,
    local_posterior: dict,
    global_param_names: list,
    local_param_names: list,
    n_samples: int,
    q_low: float = 0.16,
    q_high: float = 0.84,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    # Build flat 1-D arrays for every parameter, aligned to the shortest
    param_arrays = {}
    for p in global_param_names:
        if p in global_posterior:
            param_arrays[p] = np.asarray(global_posterior[p]).reshape(-1)
    for p in local_param_names:
        if p in local_posterior:
            param_arrays[p] = np.asarray(local_posterior[p]).reshape(-1)

    min_len = min(len(v) for v in param_arrays.values())
    param_arrays = {k: v[:min_len] for k, v in param_arrays.items()}

    mask  = _marginal_quantile_mask(param_arrays, q_low, q_high)
    valid = np.where(mask)[0]

    if len(valid) == 0:
        raise RuntimeError(
            f"No samples satisfy the per-parameter [{q_low:.2f}, {q_high:.2f}] "
            f"marginal quantile constraint simultaneously. Try widening the range."
        )

    print(
        f"Quantile selection: {len(valid)} / {min_len} samples within "
        f"per-parameter [{q_low:.2f}, {q_high:.2f}] band  →  drawing {n_samples} PPC runs."
    )

    replace = n_samples > len(valid)
    if replace:
        print(f"  Warning: requested {n_samples} > {len(valid)} valid; sampling with replacement.")

    return rng.choice(valid, size=n_samples, replace=replace)

# ── ChainConsumer cornerplot ───────────────────────────────────────────────────

def _plot_cornerplot(global_posterior, local_posterior, cfg, stream_names, out_dir):
    """
    Plot a ChainConsumer cornerplot of all global and per-stream local parameters,
    with dashed vertical/horizontal lines marking the [QUANTILE_LOW, QUANTILE_HIGH]
    Mahalanobis-band interval on every panel.
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
        arr = np.asarray(local_posterior[p]).reshape(n_streams, -1)
        for s_idx, stream in enumerate(stream_names):
            chain_dict[f'{p}_{stream}'] = arr[s_idx]

    # ── Align lengths ─────────────────────────────────────────────────────────
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
        for p in LOCAL_PARAMS:
            if name.startswith(f'{p}_'):
                stream = name[len(p) + 1:]
                return f'{LATEX.get(p, p)} {stream}'
        return name

    rename_map = {k: _label(k) for k in chain_dict}
    import pandas as pd
    df_full = pd.DataFrame(chain_dict).rename(columns=rename_map)

    # ── Build and plot ────────────────────────────────────────────────────────
    if _new_api:
        c = ChainConsumer()
        c.add_chain(Chain(samples=df_full, name="Posterior"))
        fig = c.plotter.plot()
    else:
        c = ChainConsumer()
        c.add_chain(df_full.values, parameters=list(df_full.columns), name="Posterior")
        fig = c.plotter.plot()

    # ── Compute per-parameter interval bounds and draw lines ──────────────────
    if QUANTILE_SAMPLING and N_POSTERIOR_SAMPLES > 1:
        # Use the Mahalanobis mask to find which samples are in the joint band,
        # then take the marginal min/max of those samples as the interval per param.
        mask = _marginal_quantile_mask(chain_dict, QUANTILE_LOW, QUANTILE_HIGH)

        # Bounds are just the per-parameter quantiles — no need to look at
        # the selected subset, they're exactly np.quantile at q_low / q_high
        param_bounds = {
            rename_map[k]: (
                np.quantile(chain_dict[k], QUANTILE_LOW),
                np.quantile(chain_dict[k], QUANTILE_HIGH),
            )
            for k in chain_dict
    }

        param_labels = list(df_full.columns)
        D = len(param_labels)

        # ChainConsumer produces a D×D axes grid (lower-triangular used)
        all_axes = [a for a in fig.get_axes() if a.get_visible()]
        try:
            ax_grid = np.array(all_axes).reshape(D, D)
        except ValueError:
            ax_grid = np.array(all_axes[:D * D]).reshape(D, D)

        line_kw = dict(color="tomato", linestyle="--", linewidth=1.0, alpha=0.85)

        for i, p_y in enumerate(param_labels):
            for j, p_x in enumerate(param_labels):
                if j > i:
                    continue
                ax = ax_grid[i, j]

                # vertical lines mark the x-axis parameter interval
                if p_x in param_bounds:
                    lo, hi = param_bounds[p_x]
                    ax.axvline(lo, **line_kw)
                    ax.axvline(hi, **line_kw)

                # horizontal lines mark the y-axis parameter interval (2-D panels only)
                if i != j and p_y in param_bounds:
                    lo, hi = param_bounds[p_y]
                    ax.axhline(lo, **line_kw)
                    ax.axhline(hi, **line_kw)

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
    rng = np.random.default_rng(rng_seed)

    # ── Load posteriors ────────────────────────────────────────────────────────
    global_posterior_path = os.path.join(
        cfg.base_dir,
        # '../plots/gala6D_aug/new_hyper/model54_60k_1000epochs/global_posterior.npz'
        '../plots/gala6D/new_hyper/model54_60k_1000epochs/global_posterior.npz'
        # '../hyperparameter_tuning/gala/new_aug_bigheads/model_121/gaiastreams/global_posterior.npz'
        # '../hyperparameter_tuning/gala/300k/model_21/gaiastreams/global_posterior.npz'
    )
    local_posterior_path = os.path.join(
        cfg.base_dir,
        # '../hyperparameter_tuning/gala/local/new_aug_jonas_300k_200epoch/streamnorm_standardize/model_79/gaia_121/gaia_local_posterior.npz'
        # '../hyperparameter_tuning/gala/local/jonas/streamnomr_standardize/model_2/gaia_121/gaia_local_posterior.npz'
        '../hyperparameter_tuning/gala/local/jonas/streamnomr_standardize/model_2/gaia_54/gaia_local_posterior.npz'
    )
    print('Loading global posterior from:', global_posterior_path)
    global_posterior = dict(np.load(global_posterior_path, allow_pickle=True))
    print('Loading local posterior from:', local_posterior_path)
    local_posterior  = dict(np.load(local_posterior_path,  allow_pickle=True))



    #we are going to plot the circular velocity for the global_posterior_mean,
    parameters_dict = {p: np.mean(global_posterior[p]) for p in global_posterior.keys()}
    for k in parameters_dict.keys():
        print(f'{k}: {parameters_dict[k]}')
    parameters_dict['m_bulge'] = 4501365375.06545
    parameters_dict['alpha_bulge'] = 1.8
    parameters_dict['r_bulge'] = 1.9
    radial_distance = np.linspace(0.1, 30, 100)  # kpc
    pot = gp.CCompositePotential()

    pot['halo'] = gp.NFWPotential(m     = parameters_dict['m_Triaxial_halo'],
                                r_s   = parameters_dict['r_Triaxial_halo'],
                                a     = 1,
                                b     = 1,
                                  c     = parameters_dict['q2_Triaxial_halo'],
                                units = galactic)

    pot['thin_disk'] = gp.MN3ExponentialDiskPotential(m = 4 * np.pi * parameters_dict['rho_thin_disk']*parameters_dict['hr_thin_disk']**2 * parameters_dict['hz_thin_disk'],
                                                    h_R=parameters_dict['hr_thin_disk'],
                                                    h_z=parameters_dict['hz_thin_disk'],
                                                    units=galactic,
                                                    positive_density=True)
    pot['thick_disk'] = gp.MN3ExponentialDiskPotential(m = 4 * np.pi * parameters_dict['rho_thick_disk'] *parameters_dict['hr_thick_disk']**2 * parameters_dict['hz_thick_disk'],
                                                    h_R=parameters_dict['hr_thick_disk'],
                                                    h_z=parameters_dict['hz_thick_disk'],
                                                    units=galactic,
                                                    positive_density=True)
    pot['bulge'] = gp.PowerLawCutoffPotential(m=parameters_dict['m_bulge'],
                                            r_c=parameters_dict['r_bulge'],
                                            alpha=parameters_dict['alpha_bulge'],
                                            units=galactic)
    grid = np.linspace(-30., 30., 100)
    fig = pot.plot_contours(grid=(grid, 0, grid))
    fig.savefig(os.path.join(cfg.base_dir, cfg.data_dir, 'potential_contours_mean.pdf'), dpi=150, bbox_inches='tight')
    plt.show()

    fig, ax = pot.plot_rotation_curve(R_grid = radial_distance)
    fig.savefig(os.path.join(cfg.base_dir, cfg.data_dir, 'rotation_curve_mean.pdf'), dpi=150, bbox_inches='tight')
    plt.show()
    print('Plotted potential contours and rotation curve for global posterior mean parameters.')
    # print('We stop here')

    # exit()

    stream_names = list(cfg.target_streams.keys())
    n_streams    = len(stream_names)

    use_mode = (N_POSTERIOR_SAMPLES == 1) and not QUANTILE_SAMPLING
    n_ppc    = N_POSTERIOR_SAMPLES

    LOCAL_PARAM_NAMES = ['vr', 'r', 'mu_ra_cosdec', 'mu_dec']

    # ── Cornerplot (done once, before simulation loop) ────────────────────────
    plot_dir = os.path.join(cfg.base_dir, cfg.data_dir)
    print('Generating posterior cornerplot...')
    _plot_cornerplot(global_posterior, local_posterior, cfg, stream_names, plot_dir)

    # ── Draw shared index vector ──────────────────────────────────────────────
    if use_mode:
        shared_idx = None

    elif QUANTILE_SAMPLING:
        # ── Quantile-band sampling (new) ──────────────────────────────────────
        shared_idx = _select_quantile_indices(
            global_posterior=global_posterior,
            local_posterior=local_posterior,
            global_param_names=list(cfg.priors_global.keys()),
            local_param_names=LOCAL_PARAM_NAMES,
            n_samples=n_ppc,
            q_low=QUANTILE_LOW,
            q_high=QUANTILE_HIGH,
            rng=rng,
        )

    else:
        # ── Uniform random sampling (original) ───────────────────────────────
        all_arrays = (
            [np.asarray(global_posterior[p]).reshape(-1)
             for p in cfg.priors_global.keys() if p in global_posterior] +
            [np.asarray(local_posterior[p]).reshape(-1)
             for p in LOCAL_PARAM_NAMES if p in local_posterior]
        )
        pool_size  = min(a.shape[0] for a in all_arrays)
        shared_idx = rng.choice(pool_size, size=n_ppc, replace=(n_ppc > pool_size))
        print(f'Drew {n_ppc} random indices from pool of size {pool_size}')

    global_draws = _draw_posterior_values(
        global_posterior, list(cfg.priors_global.keys()), n_ppc, use_mode, idx=shared_idx
    )
    print('global_draws:', global_draws)
    local_draws = _draw_local_posterior_values(
        local_posterior, LOCAL_PARAM_NAMES, stream_names, n_ppc, use_mode, idx=shared_idx
    )


    #we are going to get the rotation curve from the global_draws and get the error bars
    def rotation_curve(draw):
        pot = gp.CCompositePotential()

        pot['halo'] = gp.NFWPotential(m     = draw['m_Triaxial_halo'],
                                    r_s   = draw['r_Triaxial_halo'],
                                    a     = 1,
                                    b     = 1,
                                    c     = draw['q2_Triaxial_halo'],
                                    units = galactic)

        pot['thin_disk'] = gp.MN3ExponentialDiskPotential(m = 4 * np.pi * draw['rho_thin_disk']*draw['hr_thin_disk']**2 * draw['hz_thin_disk'],
                                                        h_R=draw['hr_thin_disk'],
                                                        h_z=draw['hz_thin_disk'],
                                                        units=galactic,
                                                        positive_density=True)
        pot['thick_disk'] = gp.MN3ExponentialDiskPotential(m = 4 * np.pi * draw['rho_thick_disk'] *draw['hr_thick_disk']**2 * draw['hz_thick_disk'],
                                                        h_R=draw['hr_thick_disk'],
                                                        h_z=draw['hz_thick_disk'],
                                                        units=galactic,
                                                        positive_density=True)
        pot['bulge'] = gp.PowerLawCutoffPotential(m=parameters_dict['m_bulge'],
                                                r_c=parameters_dict['r_bulge'],
                                                alpha=parameters_dict['alpha_bulge'],
                                                units=galactic)
        rotation_curve_on_grid = pot.circular_velocity(R=radial_distance, z=np.zeros_like(radial_distance))
        return rotation_curve_on_grid

    rotation_curve_for_plot = []
    for i in range(len(global_draws['m_Triaxial_halo'])):
        draw = {k: global_draws[k][i] for k in global_draws.keys()}  # fix: k not j
        rotation_curve_for_plot.append(rotation_curve(draw))

    rotation_curve_for_plot = np.array(rotation_curve_for_plot)        # shape: (n_draws, n_radii)
    mean_rotation_curve = np.mean(rotation_curve_for_plot, axis=0)    # fix: axis=0
    std_rotation_curve  = np.std(rotation_curve_for_plot,  axis=0)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(radial_distance, mean_rotation_curve, color='steelblue', label='Mean')
    ax.fill_between(
        radial_distance,
        mean_rotation_curve - 3*std_rotation_curve,
        mean_rotation_curve + 3*std_rotation_curve,
        alpha=0.3,
        color='steelblue',
        label=r'$\pm 3\sigma$',
    )
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Circular velocity [km/s]')
    ax.legend()
    fig.savefig(os.path.join(cfg.base_dir, cfg.data_dir, 'rotation_curve_sample.pdf'), dpi=150, bbox_inches='tight')
    print('Saved rotation curve samples')

    exit()

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
                    # if (stream != 'M68') and (param != 'mu_dec'):
                    # if (stream != 'M68'):
                    if True:
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
            # individual_params = [{k: v[i] for k, v in prior_samples.items()} for i in range(n_streams)]
            # results = Parallel(n_jobs=config.n_workers)(
            #     delayed(simulate_stream_gala)(p, config, code_units, i)
            #     for i, p in enumerate(individual_params)
            # )
            # sim_data_batch = np.stack(results, axis=0)
            individual_params = [{k: v[i] for k, v in prior_samples.items()} for i in range(n_streams)]

            def _safe_simulate(p, config, code_units, i):
                try:
                    return simulate_stream_gala(p, config, code_units, i)
                except Exception as e:
                    print(f"  [ppc_idx={ppc_idx}, stream={i}] simulation failed: {e}")
                    return None   # sentinel

            results = Parallel(n_jobs=config.n_workers)(
                delayed(_safe_simulate)(p, config, code_units, i)
                for i, p in enumerate(individual_params)
            )

            # If any stream failed, skip the entire PPC sample
            if any(r is None for r in results):
                failed = [i for i, r in enumerate(results) if r is None]
                print(f"  Skipping ppc_idx={ppc_idx}: streams {failed} failed.")
                continue

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

    # Encode sampling mode in filename so outputs don't overwrite each other
    if QUANTILE_SAMPLING and n_ppc > 1:
        tag = f'q{int(QUANTILE_LOW*100)}-{int(QUANTILE_HIGH*100)}'
    elif use_mode:
        tag = 'mode'
    else:
        tag = 'random'

    out_path = os.path.join(cfg.base_dir, cfg.data_dir, f'ppc_{n_ppc}samples_{tag}.npz')
    np.savez(out_path, **save_dict)
    print(f'\nSaved to {out_path}')


    


if __name__ == "__main__":
    main()