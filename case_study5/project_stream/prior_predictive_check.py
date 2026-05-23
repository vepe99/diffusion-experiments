from __future__ import annotations

from autocvd import autocvd
# autocvd(num_gpus = 1)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""    
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as mcm
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils.utils_train_jax_new_rotationcurve_fixedvlosmask import AugmentationsClass #we will need to use the augmentations on the test_set
from config.EvalConfig import EvalConfig
import hydra
from hydra.core.config_store import ConfigStore
import corner

cs = ConfigStore.instance()
cs.store(name="eval_config", node=EvalConfig)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

STREAM_NAMES: dict[int, str] = {0: "Pal5", 1: "NGC3201", 2: "M68"}
N_DIMS = 6
ALL_LABELS: list[str] = [
    r"$\alpha$ [deg]",
    r"$\delta$ [deg]",
    r"$\pi$ [mas]",
    r"$\mu_{\alpha*}$ [mas/yr]",
    r"$\mu_{\delta}$ [mas/yr]",
    r"$v_{\rm los}$ [km/s]",
]

_DEFAULT_TRAIN_CMAPS:  dict[int, str] = {0: "Blues",   1: "Greens",  2: "Purples"}
_DEFAULT_OBS_COLORS:   dict[int, str] = {0: "#d62728", 1: "#ff7f0e", 2: "#17becf"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _squeeze_to_2d(arr: np.ndarray) -> np.ndarray:
    """(N,1,S) or (1,N,S)  →  (N,S).  (N,S) is left untouched."""
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr[:, 0, :] if arr.shape[1] == 1 else arr[0]
    return arr

def _select_stream(data: dict, stream_idx: int, j_key: str = "j") -> dict:
    """Slice every array in *data* to rows whose j value equals *stream_idx*."""
    j   = np.asarray(data[j_key]).flatten().astype(int)   # cast: avoids float 0. != int 0
    sel = j == int(stream_idx)
    n   = len(j)
    if sel.sum() == 0:
        raise ValueError(
            f"_select_stream: no rows found for stream_idx={stream_idx}. "
            f"Unique j values present: {np.unique(j).tolist()}"
        )
    return {
        k: (v[sel] if isinstance(v, np.ndarray) and v.shape[0] == n else v)
        for k, v in data.items()
    }


def _get_mask(data: dict, key: str, n_sims: int, n_stars: int) -> np.ndarray:
    """Return boolean (n_sims, n_stars) mask; all-True when key is absent."""
    if key in data and data[key] is not None:
        return _squeeze_to_2d(data[key]).astype(bool)
    return np.ones((n_sims, n_stars), dtype=bool)


def _flatten_for_corner(
    sim_data: np.ndarray,              # (N_sims, N_stars, D≥6)
    att_mask: np.ndarray,              # (N_sims, N_stars)
    vlos_mask: Optional[np.ndarray],   # (N_sims, N_stars); True = star HAS v_los
    dims_to_show: list[int],
) -> np.ndarray:
    """
    Single-pass collection of valid stars across all simulations.

    When dim 5 (v_los) is in *dims_to_show*, only stars where *vlos_mask*
    is True are kept, so every row in the returned array is consistent.

    Returns
    -------
    samples : ndarray  (N_total_stars, len(dims_to_show))
        Ready to pass straight to ``corner.corner``.
    """
    need_vlos = 5 in dims_to_show
    parts: list[np.ndarray] = []

    for i in range(sim_data.shape[0]):
        valid = att_mask[i].copy()
        if need_vlos and vlos_mask is not None:
            valid &= vlos_mask[i]
        if valid.sum() == 0:
            continue
        stars = sim_data[i, valid, :N_DIMS]   # (k, 6)
        parts.append(stars[:, dims_to_show])   # (k, n_d)  — fancy index

    return np.concatenate(parts, axis=0) if parts else np.empty((0, len(dims_to_show)))


def _safe_color(cmap_name: str, level: float = 0.70):
    """Sample a solid colour from a colormap (works with matplotlib ≥3.2)."""
    try:
        return matplotlib.colormaps[cmap_name](level)
    except AttributeError:                      # matplotlib < 3.5
        import matplotlib.cm as mcm
        return mcm.get_cmap(cmap_name)(level)

from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde


def _corner_matplotlib(
    samples: np.ndarray,          # (N, D)
    labels: list[str],
    ranges: list[tuple],
    color,
    bins: int = 60,
    smooth1d: float = 0.5,
    scatter_alpha: float = 0.05,
    scatter_size: float = 0.5,
    hist_lw: float = 1.4,
    figsize_per_cell: float = 2.5,
    fig: plt.Figure | None = None,
) -> plt.Figure:
    from scipy.ndimage import gaussian_filter1d

    n_d = samples.shape[1]
    if fig is None:
        fig, axes = plt.subplots(
            n_d, n_d,
            figsize=(figsize_per_cell * n_d, figsize_per_cell * n_d),
            squeeze=False,
        )
    else:
        axes = np.array(fig.subplots(n_d, n_d, squeeze=False))

    for row in range(n_d):
        for col in range(n_d):
            ax = axes[row, col]

            # ── hide upper triangle ────────────────────────────────────────
            if col > row:
                ax.set_visible(False)
                continue

            # ── shared range / tick formatting ─────────────────────────────
            ax.set_xlim(ranges[col])
            if row != col:
                ax.set_ylim(ranges[row])

            # labels only on outer edges
            if row == n_d - 1:
                ax.set_xlabel(labels[col], fontsize=13)
            else:
                ax.set_xticklabels([])
            if col == 0 and row != 0:
                ax.set_ylabel(labels[row], fontsize=13)
            else:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=9)

            x = samples[:, col]
            y = samples[:, row]

            # ── diagonal: density-normalised histogram ─────────────────────
            if col == row:
                counts, edges = np.histogram(
                    x, bins=bins, range=ranges[col], density=True, 
                )
                if smooth1d > 0:
                    counts = gaussian_filter1d(counts.astype(float), sigma=smooth1d)
                centres = 0.5 * (edges[:-1] + edges[1:])
                ax.step(centres, counts, where="mid", color=color, linewidth=hist_lw, label='Prior')
                ax.set_ylim(bottom=0)

            # ── off-diagonal: rasterised scatter ──────────────────────────
            else:
                ax.scatter(
                    x, y,
                    s=scatter_size,
                    alpha=scatter_alpha,
                    color=color,
                    linewidths=0,
                    rasterized=True,   # key: PDF/SVG won't embed N million vectors
                )

    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def prior_predictive_check(
    training_set: dict,
    obs_data: dict,
    path_to_save: str,
    # ── dimension / stream selection ───────────────────────────────────────
    dims_to_show: Optional[list[int]] = None,
    stream_indices: Optional[list[int]] = None,
    stream_names: Optional[dict[int, str]] = None,
    # ── data keys ─────────────────────────────────────────────────────────
    sim_data_key: str = "sim_data_projected",
    attention_mask_key: str = "attention_mask",
    vlos_mask_key: str = "vlos_mask",
    j_key: str = "j",
    # ── aesthetics ────────────────────────────────────────────────────────
    bins: int = 60,
    figsize_per_cell: float = 2.5,
    train_cmaps: Optional[dict[int, str]] = None,
    obs_colors: Optional[dict[int, str]] = None,
    scatter_alpha: float = 0.80,
    scatter_size: float = 12.0,
    percentile_range: tuple[float, float] = (1.0, 99.0),
    smooth: float = 0.5,
    dpi: int = 150,
) -> None:
    """
    Save one corner-plot figure per stream under *path_to_save*.

    The training prior is rendered via ``corner.corner`` (filled density
    contours + 1-D histograms).  Gaia observations are overlaid as scatter
    points (lower triangle) and a dashed histogram (diagonal).

    Parameters
    ----------
    dims_to_show : list[int], optional
        Indices of dimensions to show.
          0=α  1=δ  2=π  3=μ_α*  4=μ_δ  5=v_los
        **Default: [0,1,2,3,4]  (v_los excluded).**

    stream_indices : list[int], optional
        Streams to plot. Default: all j values found in *training_set*.

    smooth : float
        Gaussian smoothing passed to ``corner.corner`` for contours and
        the 1-D histogram.  Set to 0 to disable.

    vlos_mask convention
        True (1) → star *has* a real v_los measurement and is included.
        False (0) → v_los is masked/missing; star is excluded from any
        panel involving dimension 5.
    """
    os.makedirs(path_to_save, exist_ok=True)

    # ── defaults ──────────────────────────────────────────────────────────
    if dims_to_show is None:
        dims_to_show = [0, 1, 2, 3, 4]
    if stream_names is None:
        stream_names = STREAM_NAMES
    if train_cmaps is None:
        train_cmaps = _DEFAULT_TRAIN_CMAPS
    if obs_colors is None:
        obs_colors = _DEFAULT_OBS_COLORS

    n_d    = len(dims_to_show)
    labels = [ALL_LABELS[d] for d in dims_to_show]

    # ── discover streams ───────────────────────────────────────────────────
    j_all = training_set[j_key].flatten()
    if stream_indices is None:
        stream_indices = sorted(int(v) for v in np.unique(j_all))

    # ══════════════════════════════════════════════════════════════════════
    # One figure per stream
    # ══════════════════════════════════════════════════════════════════════
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, 4))
    for s_idx in stream_indices:
        sname   = stream_names.get(s_idx, f"stream_{s_idx}")
        cmap    = train_cmaps.get(s_idx, "Blues")
        # c_train = _safe_color(cmap, level=0.68)
        c_train = colors[s_idx+1]
        # c_obs   = obs_colors.get(s_idx, "k")
        c_obs = "k"

        print(f"\n── {sname} ──")

        # ── slice to this stream ───────────────────────────────────────────
        # Compute the boolean mask directly from j — never rely on shape heuristics
        j_tr = training_set[j_key].flatten().astype(int)
        j_ob = obs_data[j_key].flatten().astype(int)
        sel_tr = j_tr == s_idx
        sel_ob = j_ob == s_idx

        print(f"  training sims  : {sel_tr.sum()} / {len(j_tr)} selected for {sname}")
        print(f"  observed sims  : {sel_ob.sum()} / {len(j_ob)} selected for {sname}")

        if sel_tr.sum() == 0:
            print(f"  [skip] no training simulations found for stream_idx={s_idx}. "
                f"Unique j values: {np.unique(j_tr).tolist()}")
            continue

        sim_tr = training_set[sim_data_key][sel_tr]   # (K_sims, N_stars, D)
        sim_ob = obs_data[sim_data_key][sel_ob]        # (1,      N_stars, D)

        # helper to slice a mask array (shape may be 2D or 3D) by the stream selector
        def _get_mask_for_stream(data, key, sel, n_sims, n_stars):
            if key not in data or data[key] is None:
                return np.ones((n_sims, n_stars), dtype=bool)
            raw = np.asarray(data[key])
            # squeeze any middle size-1 dimension: (N,1,S) → (N,S)
            if raw.ndim == 3 and raw.shape[1] == 1:
                raw = raw[:, 0, :]
            elif raw.ndim == 3:
                raw = raw[0]
            return raw[sel].astype(bool)

        att_tr  = _get_mask_for_stream(training_set, attention_mask_key, sel_tr, *sim_tr.shape[:2])
        att_ob  = _get_mask_for_stream(obs_data,     attention_mask_key, sel_ob, *sim_ob.shape[:2])

        def _get_vlos_for_stream(data, key, sel):
            if key not in data or data[key] is None:
                return None
            raw = _squeeze_to_2d(np.asarray(data[key]))
            return raw[sel].astype(bool)

        vlos_tr = _get_vlos_for_stream(training_set, vlos_mask_key, sel_tr)
        vlos_ob = _get_vlos_for_stream(obs_data,     vlos_mask_key, sel_ob)

        # ── collect all valid stars in one vectorised pass ─────────────────
        train_samples = _flatten_for_corner(sim_tr, att_tr, vlos_tr, dims_to_show)
        obs_samples   = _flatten_for_corner(sim_ob, att_ob, vlos_ob, dims_to_show)
        print(f"  training stars : {train_samples.shape[0]:,}")
        print(f"  observed stars : {obs_samples.shape[0]}")

        if train_samples.shape[0] < 2:
            print(f"  [skip] not enough training stars for {sname}")
            continue

        # ── axis ranges from training percentiles ──────────────────────────
        ranges = []
        for i in range(n_d):
            t_min, t_max = np.nanpercentile(train_samples[:, i], list(percentile_range))
            if obs_samples.shape[0] > 0:
                o_min = np.nanmin(obs_samples[:, i])
                o_max = np.nanmax(obs_samples[:, i])
                ranges.append((min(t_min, o_min), max(t_max, o_max)))
            else:
                ranges.append((t_min, t_max))

        # ── corner plot of the training prior ─────────────────────────────
        # fig = corner.corner(
        #     train_samples,
        #     labels=labels,
        #     range=ranges,
        #     bins=bins,
        #     smooth=smooth,
        #     smooth1d=smooth,
        #     color=c_train,
        #     plot_datapoints=False,
        #     plot_density=True,
        #     fill_contours=False,
        #     levels=(0.90, 1.0),
        #     contourf_kwargs={"alpha": 0.35},
        #     contour_kwargs={"linewidths": 1.2},
        #     hist_kwargs={"linewidth": 1.4,"density":True },
        #     label_kwargs={"fontsize": 15},
        #     tick_kwargs={"labelsize": 15},
        #     fig=plt.figure(figsize=(figsize_per_cell * n_d, figsize_per_cell * n_d)),
        # )
        fig = _corner_matplotlib(
                train_samples,
                labels=labels,
                ranges=ranges,
                color=c_train,
                bins=bins,
                smooth1d=smooth,
                scatter_alpha=0.5,   # tune to taste — lower for denser clouds
                scatter_size=0.3,
                hist_lw=1.4,
                figsize_per_cell=figsize_per_cell,
            )
        fig.suptitle(sname, fontsize=25, fontweight="bold", y=1.005)

        # ── retrieve the axes grid that corner built ───────────────────────
        ax_grid = np.array(fig.axes).reshape((n_d, n_d))

        # ── overlay observations ───────────────────────────────────────────
        if obs_samples.shape[0]:
            for row_i in range(n_d):

                # diagonal: dashed histogram
                ax_diag = ax_grid[row_i, row_i]
                n_bins_obs = max(5, min(bins, obs_samples.shape[0] // 2))
                ax_diag.hist(
                    obs_samples[:, row_i],
                    bins=n_bins_obs,
                    range=ranges[row_i],
                    density=True,
                    histtype="step",
                    color=c_obs,
                    linewidth=2.4,
                    linestyle="-",
                    zorder=5,
                    label="Gaia"
                )
                if row_i == 0:
                    ax_diag.legend(fontsize=15, loc="upper right")
                # lower triangle: scatter
                for col_i in range(row_i):
                    ax_grid[row_i, col_i].scatter(
                        obs_samples[:, col_i],
                        obs_samples[:, row_i],
                        s=scatter_size,
                        c=c_obs,
                        alpha=scatter_alpha,
                        zorder=5,
                        linewidths=0,
                        edgecolors="none",
                    )

        # ── legend ─────────────────────────────────────────────────────────
        # fig.legend(
        #     handles=[
        #         Line2D([0], [0], color=c_train, linewidth=3,
        #                label="Training prior"),
        #         Line2D([0], [0], color=c_obs, linewidth=2.5,
        #                linestyle="--", label="Observations"),
        #     ],
        #     loc="upper right",
        #     bbox_to_anchor=(1.0, 1.0),
        #     fontsize=9,
        #     framealpha=0.9,
        # )

        plt.tight_layout()

        dim_tag  = "_".join(map(str, dims_to_show))
        out_path = os.path.join(path_to_save, f"prior_predictive_{sname}_dims{dim_tag}.pdf")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  → saved {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Imports needed for the new function (add at the top of your file)
# ─────────────────────────────────────────────────────────────────────────────
import astropy.units as u
import gala.potential as gp
from gala.units import galactic


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build potential and return v_circ at r_vc
# ─────────────────────────────────────────────────────────────────────────────

def _build_potential_and_vcirc(params: dict, r_vc_kpc: float = 8.0) -> float:
    """
    Instantiate a CCompositePotential from a single-row parameter dict
    and return the circular velocity (km/s) at r_vc_kpc.

    Missing keys fall back to the defaults listed below, so you can safely
    comment out any parameters you don't want to vary.

    Defaults
    --------
    q1_Triaxial_halo  → 1.0   (spherical b-axis)
    q2_Triaxial_halo  → 1.0   (spherical c-axis)
    m_bulge / r_bulge / alpha_bulge → bulge component omitted entirely
    """
    def _s(key, default=None):
        """Scalar value from a possibly length-1 array; uses default if key absent."""
        if key not in params:
            if default is not None:
                return float(default)
            raise KeyError(
                f"_build_potential_and_vcirc: required key '{key}' missing from params "
                f"and no default is defined."
            )
        return float(np.asarray(params[key]).flat[0])

    pot = gp.CCompositePotential()

    pot["halo"] = gp.NFWPotential(
        m   = _s("m_Triaxial_halo"),
        r_s = _s("r_Triaxial_halo"),
        a   = 1.0,
        b   = _s("q1_Triaxial_halo", default=1.0),   # ← default: spherical
        c   = _s("q2_Triaxial_halo", default=1.0),   # ← default: spherical
        units=galactic,
    )

    rho_thin = _s("rho_thin_disk")
    hr_thin  = _s("hr_thin_disk")
    hz_thin  = _s("hz_thin_disk")
    pot["thin_disk"] = gp.MN3ExponentialDiskPotential(
        m    = 4 * np.pi * rho_thin * hr_thin**2 * hz_thin,
        h_R  = hr_thin,
        h_z  = hz_thin,
        units=galactic,
        positive_density=True,
    )

    rho_thick = _s("rho_thick_disk")
    hr_thick  = _s("hr_thick_disk")
    hz_thick  = _s("hz_thick_disk")
    pot["thick_disk"] = gp.MN3ExponentialDiskPotential(
        m    = 4 * np.pi * rho_thick * hr_thick**2 * hz_thick,
        h_R  = hr_thick,
        h_z  = hz_thick,
        units=galactic,
        positive_density=True,
    )

    # Bulge is optional — only added when all three keys are present
    bulge_keys = ("m_bulge", "r_bulge", "alpha_bulge")
    if all(k in params for k in bulge_keys):
        pot["bulge"] = gp.PowerLawCutoffPotential(
            m     = _s("m_bulge"),
            r_c   = _s("r_bulge"),
            alpha = _s("alpha_bulge"),
            units =galactic,
        )

    vc = pot.circular_velocity(q=[r_vc_kpc, 0.0, 0.0] * u.kpc)
    return float(vc.to(u.km / u.s).value.flat[0])

# ─────────────────────────────────────────────────────────────────────────────
# Main plotting function
# ─────────────────────────────────────────────────────────────────────────────

# def prior_parameters_corner(
#     parameters: dict[str, np.ndarray],
#     path_to_save: str,
#     # ── v_circ constraint ──────────────────────────────────────────────────
#     vc_target_kms: float = 220.0,
#     vc_tolerance:  float = 0.10,        # fractional; 0.10 → ±10 %
#     r_vc_kpc:      float = 8.0,
#     # ── which parameters to display ───────────────────────────────────────
#     param_keys: Optional[list[str]] = None,
#     param_labels: Optional[dict[str, str]] = None,
#     # ── aesthetics ────────────────────────────────────────────────────────
#     bins:             int   = 40,
#     smooth:           float = 0.5,
#     figsize_per_cell: float = 2.2,
#     scatter_size:     float = 8.0,
#     scatter_alpha:    float = 0.60,
#     prior_color:      str   = "#aec7e8",   # light blue  – all prior samples
#     vc_color:         str   = "#d62728",   # red         – v_circ-consistent
#     percentile_range: tuple[float, float] = (0.5, 99.5),
#     dpi:              int   = 150,
#     filename:         str   = "prior_parameters_corner.png",
#     verbose:          bool  = True,
# ) -> None:
#     """
#     Corner plot of the prior parameter distribution, with a highlighted
#     subset whose circular velocity at ``r_vc_kpc`` lies within
#     ``vc_tolerance`` of ``vc_target_kms``.

#     Parameters
#     ----------
#     parameters : dict[str, np.ndarray]
#         Flat dict mapping parameter names → 1-D arrays of length N_samples.
#         Expected keys (must all be present for the v_circ calculation):
#             m_Triaxial_halo, r_Triaxial_halo, q1_Triaxial_halo, q2_Triaxial_halo,
#             rho_thin_disk, hr_thin_disk, hz_thin_disk,
#             rho_thick_disk, hr_thick_disk, hz_thick_disk,
#             m_bulge, r_bulge, alpha_bulge
#     param_keys : list[str], optional
#         Subset of parameter names to include in the corner axes.
#         Defaults to all keys in *parameters*.
#     param_labels : dict[str, str], optional
#         Mapping from key → axis label.  Missing keys fall back to the key name.
#     vc_target_kms : float
#         Target circular velocity in km/s  (default 220).
#     vc_tolerance : float
#         Fractional half-width of the accepted band  (default 0.10 → ±10 %).
#     r_vc_kpc : float
#         Galactocentric radius for the v_circ evaluation  (default 8 kpc).
#     verbose : bool
#         Print per-sample progress and summary statistics.
#     """
#     os.makedirs(path_to_save, exist_ok=True)

#     # ── choose which parameters to display ────────────────────────────────
#     if param_keys is None:
#         param_keys = list(parameters.keys())
#     if param_labels is None:
#         param_labels = {}

#     labels    = [param_labels.get(k, k) for k in param_keys]
#     n_samples = len(next(iter(parameters.values())))
#     n_d       = len(param_keys)

#     # stack into (N, n_d) matrix for corner
#     data_matrix = np.column_stack([
#         np.asarray(parameters[k]).flatten()[:n_samples]
#         for k in param_keys
#     ])

#     # ── compute v_circ for every sample ───────────────────────────────────
#     vc_values  = np.full(n_samples, np.nan)
#     vc_lo      = vc_target_kms * (1.0 - vc_tolerance)
#     vc_hi      = vc_target_kms * (1.0 + vc_tolerance)

#     if verbose:
#         print(f"\nEvaluating v_circ at {r_vc_kpc} kpc for {n_samples:,} samples …")
#         print(f"  Accepted band : [{vc_lo:.1f}, {vc_hi:.1f}] km/s")

#     for i in range(n_samples):
#         if verbose and i % max(1, n_samples // 20) == 0:
#             print(f"  {i:>{len(str(n_samples))}}/{n_samples}", end="\r")
#         try:
#             single = {k: parameters[k][i] for k in parameters}
#             vc_values[i] = _build_potential_and_vcirc(single, r_vc_kpc=r_vc_kpc)
#         except Exception as exc:
#             if verbose:
#                 print(f"\n  [warn] sample {i} raised {exc!r} – skipped")

#     vc_mask      = (vc_values >= vc_lo) & (vc_values <= vc_hi)
#     n_vc_ok      = vc_mask.sum()
#     frac_vc_ok   = n_vc_ok / n_samples if n_samples > 0 else 0.0

#     if verbose:
#         print(f"\n  v_circ stats  : min={np.nanmin(vc_values):.1f}  "
#               f"median={np.nanmedian(vc_values):.1f}  "
#               f"max={np.nanmax(vc_values):.1f} km/s")
#         print(f"  Accepted      : {n_vc_ok}/{n_samples}  ({100*frac_vc_ok:.1f} %)")

#     all_samples = data_matrix                       # (N, n_d)
#     vc_samples  = data_matrix[vc_mask]              # (M, n_d)

#     # ── axis ranges from full prior ────────────────────────────────────────
#     ranges = [
#         tuple(np.nanpercentile(all_samples[:, i], list(percentile_range)))
#         for i in range(n_d)
#     ]

#     # ── corner plot: full prior ────────────────────────────────────────────
#     fig = corner.corner(
#         all_samples,
#         labels=labels,
#         range=ranges,
#         bins=bins,
#         smooth=smooth,
#         smooth1d=smooth,
#         color=prior_color,
#         plot_datapoints=False,
#         plot_density=True,
#         fill_contours=False,
#         levels=(0.68, 0.90, 1.0),
#         contourf_kwargs={"alpha": 0.30},
#         contour_kwargs={"linewidths": 1.0},
#         hist_kwargs={"linewidth": 1.4},
#         label_kwargs={"fontsize": 8},
#         tick_kwargs={"labelsize": 6},
#         fig=plt.figure(figsize=(figsize_per_cell * n_d, figsize_per_cell * n_d)),
#     )

#     # ── overlay v_circ-consistent samples ─────────────────────────────────
#     ax_grid = np.array(fig.axes).reshape((n_d, n_d))

#     if vc_samples.shape[0] >= 1:
#         for row_i in range(n_d):
#             # diagonal: dashed histogram
#             ax_diag = ax_grid[row_i, row_i]
#             n_bins_vc = max(5, min(bins, vc_samples.shape[0] // 2))
#             ax_diag.hist(
#                 vc_samples[:, row_i],
#                 bins=n_bins_vc,
#                 range=ranges[row_i],
#                 density=False,
#                 histtype="step",
#                 color=vc_color,
#                 linewidth=2.0,
#                 linestyle="--",
#                 zorder=5,
#             )
#             # lower triangle: scatter
#             for col_i in range(row_i):
#                 ax_grid[row_i, col_i].scatter(
#                     vc_samples[:, col_i],
#                     vc_samples[:, row_i],
#                     s=scatter_size,
#                     c=vc_color,
#                     alpha=scatter_alpha,
#                     zorder=5,
#                     linewidths=0,
#                 )

#     # ── title and legend ───────────────────────────────────────────────────
#     title = (
#         f"Prior parameters  —  "
#         f"$v_{{\\rm circ}}({r_vc_kpc}\\,{{\\rm kpc}}) = "
#         f"{vc_target_kms:.0f} \\pm {100*vc_tolerance:.0f}\\%$ km/s  "
#         f"({n_vc_ok}/{n_samples} samples)"
#     )
#     fig.suptitle(title, fontsize=11, fontweight="bold", y=1.002)

#     fig.legend(
#         handles=[
#             Line2D([0], [0], color=prior_color, linewidth=3,
#                    label="Full prior"),
#             Line2D([0], [0], color=vc_color, linewidth=2.0,
#                    linestyle="--",
#                    label=f"$v_{{\\rm circ}} \\in [{vc_lo:.0f},{vc_hi:.0f}]$ km/s"),
#         ],
#         loc="upper right",
#         bbox_to_anchor=(1.0, 1.0),
#         fontsize=9,
#         framealpha=0.9,
#     )

#     plt.tight_layout()
#     out_path = os.path.join(path_to_save, filename)
#     fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
#     plt.close(fig)

#     if verbose:
#         print(f"\n  → saved {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Add at the top of your file
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
from chainconsumer import ChainConsumer, Chain, PlotConfig


# ─────────────────────────────────────────────────────────────────────────────
# Main plotting function
# ─────────────────────────────────────────────────────────────────────────────

def prior_parameters_corner(
    parameters: dict[str, np.ndarray],
    path_to_save: str,
    # ── v_circ constraint ──────────────────────────────────────────────────
    vc_target_kms: float = 220.0,
    vc_tolerance:  float = 0.10,         # fractional; 0.10 → ±10 %
    r_vc_kpc:      float = 8.0,
    # ── which parameters to display ───────────────────────────────────────
    param_keys:    Optional[list[str]]   = None,
    param_labels:  Optional[dict[str, str]] = None,
    # ── aesthetics ────────────────────────────────────────────────────────
    bins:             int   = 40,
    smooth:           float = 1.0,        # ChainConsumer uses KDE sigma
    figsize_per_cell: float = 2.2,
    prior_color:      str   = "#aec7e8",  # light blue  – all prior samples
    vc_color:         str   = "#d62728",  # red         – v_circ-consistent
    percentile_range: tuple[float, float] = (0.5, 99.5),
    dpi:              int   = 150,
    filename:         str   = "prior_parameters_corner.png",
    verbose:          bool  = True,
) -> None:
    """
    Corner plot of the prior parameter distribution built with ChainConsumer,
    with a second overlaid chain highlighting samples whose circular velocity
    at ``r_vc_kpc`` lies within ``vc_tolerance`` of ``vc_target_kms``.

    Parameters
    ----------
    parameters : dict[str, np.ndarray]
        Flat dict mapping parameter names → 1-D arrays of length N_samples.
        Expected keys (must all be present for the v_circ calculation):
            m_Triaxial_halo, r_Triaxial_halo, q1_Triaxial_halo, q2_Triaxial_halo,
            rho_thin_disk, hr_thin_disk, hz_thin_disk,
            rho_thick_disk, hr_thick_disk, hz_thick_disk,
            m_bulge, r_bulge, alpha_bulge
    param_keys : list[str], optional
        Subset of parameter names to include in the corner axes.
        Defaults to all keys in *parameters*.
    param_labels : dict[str, str], optional
        Mapping from key → LaTeX axis label. Missing keys fall back to the
        key name itself.
    smooth : float
        KDE smoothing bandwidth passed to ChainConsumer. Larger values give
        smoother contours; set to 0 to disable KDE and use a histogram.
    """
    os.makedirs(path_to_save, exist_ok=True)

    # ── parameter selection and labels ────────────────────────────────────
    if param_keys is None:
        param_keys = list(parameters.keys())
    if param_labels is None:
        param_labels = {}

    # Map from storage key → display label (falls back to the key itself)
    col_names  = [param_labels.get(k, k) for k in param_keys]
    n_samples  = len(next(iter(parameters.values())))
    n_d        = len(param_keys)

    # (N, n_d) array; column order matches col_names
    data_matrix = np.column_stack([
        np.asarray(parameters[k]).flatten()[:n_samples]
        for k in param_keys
    ])

    # ── evaluate v_circ for every sample ──────────────────────────────────
    vc_values = np.full(n_samples, np.nan)
    vc_lo     = vc_target_kms * (1.0 - vc_tolerance)
    vc_hi     = vc_target_kms * (1.0 + vc_tolerance)

    if verbose:
        print(f"\nEvaluating v_circ at {r_vc_kpc} kpc for {n_samples:,} samples …")
        print(f"  Accepted band : [{vc_lo:.1f}, {vc_hi:.1f}] km/s")

    for i in range(n_samples):
        if verbose and i % max(1, n_samples // 20) == 0:
            print(f"  {i:>{len(str(n_samples))}}/{n_samples}", end="\r")
        try:
            single       = {k: parameters[k][i] for k in parameters}
            vc_values[i] = _build_potential_and_vcirc(single, r_vc_kpc=r_vc_kpc)
        except Exception as exc:
            if verbose:
                print(f"\n  [warn] sample {i} raised {exc!r} – skipped")

    vc_mask    = (vc_values >= vc_lo) & (vc_values <= vc_hi)
    n_vc_ok    = int(vc_mask.sum())
    frac_vc_ok = n_vc_ok / n_samples if n_samples > 0 else 0.0

    if verbose:
        print(f"\n  v_circ stats  : min={np.nanmin(vc_values):.1f}  "
              f"median={np.nanmedian(vc_values):.1f}  "
              f"max={np.nanmax(vc_values):.1f} km/s")
        print(f"  Accepted      : {n_vc_ok}/{n_samples}  ({100*frac_vc_ok:.1f} %)")

    # ── build DataFrames for ChainConsumer ────────────────────────────────
    # ChainConsumer identifies axes by DataFrame column names, so we use the
    # display labels directly as column names here.
    df_all = pd.DataFrame(data_matrix,              columns=col_names)
    df_vc  = pd.DataFrame(data_matrix[vc_mask],     columns=col_names)

    # ── axis extents (computed from full prior) ───────────────────────────
    # ChainConsumer accepts per-parameter (lo, hi) tuples via the
    # `extents` argument on Chain, keyed by column name.
    extents = {
        col: tuple(np.nanpercentile(data_matrix[:, i], list(percentile_range)))
        for i, col in enumerate(col_names)
    }

    # ── assemble ChainConsumer ─────────────────────────────────────────────
    c = ChainConsumer()

    c.add_chain(
        Chain(
            samples     = df_all,
            name        = "Full prior",
            color       = "#4878CF",   # steel blue
            shade       = True,
            shade_alpha = 0.25,
            bar_shade   = True,
        )
    )

    if n_vc_ok >= 2:
        vc_label = (
            f"$v_{{\\rm circ}}({r_vc_kpc}\\,{{\\rm kpc}}) "
            f"\\in [{vc_lo:.0f},{vc_hi:.0f}]$ km/s  "
            f"({n_vc_ok}/{n_samples})"
        )
        c.add_chain(
            Chain(
                samples     = df_vc,
                name        = vc_label,
                color       = "#E8532A",   # vivid orange-red
                shade       = True,
                shade_alpha = 0.40,
                bar_shade   = True,
            )
        )
    elif verbose:
        print("  [warn] fewer than 2 v_circ-consistent samples – overlay skipped")

    # ── plot config ────────────────────────────────────────────────────────
    c.set_plot_config(
        PlotConfig(
            bins             = bins,
            smooth           = smooth,
            fig_size         = (figsize_per_cell * n_d, figsize_per_cell * n_d),
            label_font_size  = 9,
            tick_font_size   = 6,
            show_legend      = True,          # ← ensures the legend is drawn
            legend_location  = (0, -1),       # top-right corner of the grid
            legend_kwargs    = {
                "fontsize"   : 9,
                "framealpha" : 0.9,
                "title"      : "Chains",
                "title_fontsize": 9,
            },
            extents = extents,
        )
    )

    # ── render and save ────────────────────────────────────────────────────
    out_path = os.path.join(path_to_save, filename)
    fig = c.plotter.plot(filename=out_path, figsize=(figsize_per_cell * n_d,
                                                      figsize_per_cell * n_d))

    title = (
        f"Prior parameters  —  "
        f"$v_{{\\rm circ}}({r_vc_kpc}\\,{{\\rm kpc}}) = "
        f"{vc_target_kms:.0f} \\pm {100 * vc_tolerance:.0f}\\%$ km/s"
    )
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.002)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"\n  → saved {out_path}")


@hydra.main(version_base=None, config_path="config", config_name="eval_config_new_rotationcurve_agama",)
def main(cfg: EvalConfig):
    base_dir               = "/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/"
    training_data_data_dir = "/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/streams/data_agama/"
    observed_data_path     = os.path.join(base_dir, "gaia_observed_streams_6Dwitherrors_cutNGC3201.npz")
    path_to_save           = os.path.join(base_dir, "plots/prior_predictive_check/agama/")

    # ── Training set ─────────────────────────────────────────────────────────
    training_set_loaded = dict(np.load(os.path.join(base_dir, training_data_data_dir, "training_data_local_1000000.npz")))
    training_set = {}
    # for k in ["sim_data_projected", "j"]:
    for k in training_set_loaded.keys():
        training_set[k] = training_set_loaded[k][0:1_000_000]   # ← keep as dict, never overwrite
        print(f"Training set {k} shape: {training_set[k].shape}")
    # ── Observations ─────────────────────────────────────────────────────────
    observations_loaded = np.load(observed_data_path, allow_pickle=True)
    obs_data = {}
    for k in observations_loaded.keys():
        obs_data[k] = observations_loaded[k]                                 # (3, 1) ← drop outer dim
    for k in obs_data:
        print(f"Observations {k} shape: {obs_data[k].shape}")

    #augumentation function for training 
    augmentations_class = AugmentationsClass(cfg)
    augmentations = []
    if "cut_to_300_particles" in cfg.augmentations:
        augmentations.append(augmentations_class.cut_to_300_particles)
    # --- Coordinate transforms (must be first, before any masking) ---
    if "remove_los_velocity" in cfg.augmentations:
        augmentations.append(augmentations_class.remove_los_velocity)
    if "convert_distance_to_parallax" in cfg.augmentations:
        augmentations.append(augmentations_class.convert_distance_to_parallax)

    # --- Observational selection (window → subsample → compact) ---
    if "observational_window" in cfg.augmentations:
        augmentations.append(augmentations_class.observational_window)
    if "observational_window_spline" in cfg.augmentations:
        augmentations.append(augmentations_class.observational_window_spline)
    if "observational_window_random" in cfg.augmentations:
        augmentations.append(augmentations_class.observational_window_random)
    if "observed_n_stars" in cfg.augmentations:
        augmentations.append(augmentations_class.subsampling_to_observed_n_stars)
    if "compact_to_attended" in cfg.augmentations:
        augmentations.append(augmentations_class.compact_to_attended)

    # --- Photometric augmentation (magnitudes → errors → apply) ---
    if "sample_magnitudes" in cfg.augmentations:
        augmentations.append(augmentations_class.sample_magnitudes)
    if "sample_obs_error" in cfg.augmentations:
        augmentations.append(augmentations_class.sample_obs_error)
    if "apply_obs_error" in cfg.augmentations:
        augmentations.append(augmentations_class.apply_obs_error)

    # --- v_los masking (must be after apply_obs_error) ---
    if "mask_vlos" in cfg.augmentations:
        augmentations.append(augmentations_class.mask_vlos)

    # --- Symmetry augmentations ---
    if "flip_dirz" in cfg.augmentations:
        augmentations.append(augmentations_class.flip_dirz)


    training_set[cfg.sim_data] = training_set[cfg.sim_data].reshape(-1, training_set[cfg.sim_data].shape[-2], training_set[cfg.sim_data].shape[-1])
    training_set['j'] = training_set['j'].reshape(-1, 1)
    print('Training data sim shape before augmentation: ', training_set[cfg.sim_data].shape)
    for aug in augmentations:
        training_set = aug(training_set)
    print('Training data sim shape after augmentation: ', training_set[cfg.sim_data].shape)
    


    #Augmentation for the observations
    augmentations = []
    if "remove_los_velocity" in cfg.augmentations:
        augmentations.append(augmentations_class.remove_los_velocity)
    if "sample_obs_error" in cfg.augmentations:
        augmentations.append(augmentations_class.sample_obs_error)
    if "observational_window" in cfg.augmentations:
        augmentations.append(augmentations_class.observational_window)

    #reshape the streams dimensions
    obs_data[cfg.sim_data] = obs_data[cfg.sim_data].reshape(-1, obs_data[cfg.sim_data].shape[-2], obs_data[cfg.sim_data].shape[-1])
    # obs_data[cfg.sim_data][:, :, 3], obs_data[cfg.sim_data][:, :, 4] = obs_data[cfg.sim_data][:, :, 4], obs_data[cfg.sim_data][:, :, 3] #swap the two proper motions to match the training data format
    obs_data['j'] = obs_data['j'].reshape(-1, 1)
    if 'vlos_mask' in obs_data:
        obs_data['vlos_mask'] = obs_data['vlos_mask'].reshape(-1, 1, obs_data[cfg.sim_data].shape[-2])
    for k in [cfg.sim_data, "attention_mask", "magnitudes", "vlos_mask"]:
        print(f"{k} shape before truncation: {obs_data[k].shape}")
        if len(obs_data[k].shape) == 2:
            obs_data[k] = obs_data[k][:, :300]
        elif (k == cfg.sim_data):
            obs_data[k] = obs_data[k][:, :300, :]
        elif (k == "attention_mask")|(k == "vlos_mask"):
            obs_data[k] = obs_data[k][:, :, :300]
        print(f"{k} shape after truncation: {obs_data[k].shape}")
    print('Observed data sim shape before augmentation: ', obs_data[cfg.sim_data].shape)
    for aug in augmentations:
        obs_data = aug(obs_data)
    for k in obs_data.keys():
        print(f"{k} shape after augmentation: {obs_data[k].shape}")

    prior_predictive_check(training_set, obs_data, path_to_save, dims_to_show = [0, 1, 2, 3, 4])
    print("saved at: ", path_to_save)

    # ── load / build parameter dict ──────────────────────────────────────────
    # Adjust the key names to whatever your training .npz actually stores.
    # param_npz = np.load(os.path.join(training_data_data_dir, "training_data_300000.npz"))
    # N = 300_000   # number of samples to use for the corner plot (adjust as needed)

    # parameters = {
    #     "m_Triaxial_halo"  : param_npz["m_Triaxial_halo"]  [:N].flatten(),
    #     "r_Triaxial_halo"  : param_npz["r_Triaxial_halo"]  [:N].flatten(),
    #     # "q1_Triaxial_halo" : param_npz["q1_Triaxial_halo"] [:N].flatten(),
    #     "q2_Triaxial_halo" : param_npz["q2_Triaxial_halo"] [:N].flatten(),
    #     "rho_thin_disk"    : param_npz["rho_thin_disk"]    [:N].flatten(),
    #     "hr_thin_disk"     : param_npz["hr_thin_disk"]     [:N].flatten(),
    #     "hz_thin_disk"     : param_npz["hz_thin_disk"]     [:N].flatten(),
    #     "rho_thick_disk"   : param_npz["rho_thick_disk"]   [:N].flatten(),
    #     "hr_thick_disk"    : param_npz["hr_thick_disk"]    [:N].flatten(),
    #     "hz_thick_disk"    : param_npz["hz_thick_disk"]    [:N].flatten(),
    #     # "m_bulge"          : param_npz["m_bulge"]          [:N].flatten(),
    #     # "r_bulge"          : param_npz["r_bulge"]          [:N].flatten(),
    #     # "alpha_bulge"      : param_npz["alpha_bulge"]      [:N].flatten(),
    # }

    # Nice LaTeX labels for each axis
    # param_labels = {
    #     "m_Triaxial_halo"  : r"$M_{\rm halo}$",
    #     "r_Triaxial_halo"  : r"$r_s$",
    #     # "q1_Triaxial_halo" : r"$q_1$",
    #     "q2_Triaxial_halo" : r"$q_2$",
    #     "rho_thin_disk"    : r"$\rho_{\rm thin}$",
    #     "hr_thin_disk"     : r"$h_{R,\rm thin}$",
    #     "hz_thin_disk"     : r"$h_{z,\rm thin}$",
    #     "rho_thick_disk"   : r"$\rho_{\rm thick}$",
    #     "hr_thick_disk"    : r"$h_{R,\rm thick}$",
    #     "hz_thick_disk"    : r"$h_{z,\rm thick}$",
    #     # "m_bulge"          : r"$M_{\rm bulge}$",
    #     # "r_bulge"          : r"$r_c$",
    #     # "alpha_bulge"      : r"$\alpha_{\rm bulge}$",
    # }

    # prior_parameters_corner(
    #     parameters    = parameters,
    #     path_to_save  = os.path.join(base_dir, "plots/prior_predictive_check/gala/"),
    #     param_labels  = param_labels,
    #     vc_target_kms = 220.0,
    #     vc_tolerance  = 0.10,       # ±10 %
    #     r_vc_kpc      = 8.0,
    #     bins          = 40,
    #     verbose       = True,
    # )



if __name__ == "__main__":
    main()