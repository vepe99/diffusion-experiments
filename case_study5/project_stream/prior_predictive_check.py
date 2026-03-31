from __future__ import annotations

from autocvd import autocvd
autocvd(num_gpus = 1)
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""    
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as mcm
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils.utils_train_jax import AugmentationsClass #we will need to use the augmentations on the test_set
from config.EvalConfig import EvalConfig
import hydra
from hydra.core.config_store import ConfigStore
import corner

cs = ConfigStore.instance()
cs.store(name="eval_config", node=EvalConfig)

"""
prior_predictive_check.py  –  corner-library version
──────────────────────────────────────────────────────
Uses `corner.corner` for the training-prior density, then overlays Gaia
observations as scatter / dashed histograms on the same axes.

Quick usage
-----------
# Default: dims 0-4 (no v_los), all three streams
prior_predictive_check(training_set, obs_data, path_to_save)

# Include v_los
prior_predictive_check(training_set, obs_data, path_to_save,
                       dims_to_show=[0, 1, 2, 3, 4, 5])

# Only RA/Dec, Pal5 only
prior_predictive_check(training_set, obs_data, path_to_save,
                       dims_to_show=[0, 1], stream_indices=[0])
"""

# from __future__ import annotations

# import os
# from typing import Optional

# import corner
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.lines import Line2D

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
    for s_idx in stream_indices:
        sname   = stream_names.get(s_idx, f"stream_{s_idx}")
        cmap    = train_cmaps.get(s_idx, "Blues")
        c_train = _safe_color(cmap, level=0.68)
        c_obs   = obs_colors.get(s_idx, "tomato")

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
        fig = corner.corner(
            train_samples,
            labels=labels,
            range=ranges,
            bins=bins,
            smooth=smooth,
            smooth1d=smooth,
            color=c_train,
            plot_datapoints=False,
            plot_density=True,
            fill_contours=False,
            levels=(0.68, 0.99),
            contourf_kwargs={"alpha": 0.35},
            contour_kwargs={"linewidths": 1.2},
            hist_kwargs={"linewidth": 1.4, },
            label_kwargs={"fontsize": 9},
            tick_kwargs={"labelsize": 7},
            fig=plt.figure(figsize=(figsize_per_cell * n_d, figsize_per_cell * n_d)),
        )
        fig.suptitle(sname, fontsize=14, fontweight="bold", y=1.005)

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
                    density=False,
                    histtype="step",
                    color=c_obs,
                    linewidth=2.4,
                    linestyle="--",
                    zorder=5,
                )

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
        fig.legend(
            handles=[
                Line2D([0], [0], color=c_train, linewidth=3,
                       label="Training prior"),
                Line2D([0], [0], color=c_obs, linewidth=2.5,
                       linestyle="--", label="Observations"),
            ],
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            fontsize=9,
            framealpha=0.9,
        )

        plt.tight_layout()

        dim_tag  = "_".join(map(str, dims_to_show))
        out_path = os.path.join(path_to_save, f"prior_predictive_{sname}_dims{dim_tag}.png")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  → saved {out_path}")


@hydra.main(version_base=None, config_path="config", config_name="eval_config",)
def main(cfg: EvalConfig):
    base_dir               = "/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/"
    training_data_data_dir = "streams/data_galax_1e6/"
    observed_data_path     = os.path.join(base_dir, "gaia_observed_streams_6Dwitherrors_cutNGC3201.npz")
    path_to_save           = os.path.join(base_dir, "plots/prior_predictive_check/2smalldataset/")

    # ── Training set ─────────────────────────────────────────────────────────
    training_set_loaded = dict(np.load(os.path.join(base_dir, training_data_data_dir,
                                               "training_data_local_60000.npz")))
    training_set = {}
    # for k in ["sim_data_projected", "j"]:
    for k in training_set_loaded.keys():
        training_set[k] = training_set_loaded[k][:1_000]   # ← keep as dict, never overwrite
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
    if "remove_los_velocity" in cfg.augmentations:
        augmentations.append(augmentations_class.remove_los_velocity)
    if "convert_distance_to_parallax" in cfg.augmentations:
        augmentations.append(augmentations_class.convert_distance_to_parallax)
    if "sample_magnitudes" in cfg.augmentations:
        augmentations.append(augmentations_class.sample_magnitudes)
    if "sample_obs_error" in cfg.augmentations:
        augmentations.append(augmentations_class.sample_obs_error)
    # if "apply_obs_error" in cfg.augmentations:  
    #     augmentations.append(augmentations_class.apply_obs_error)
    if "observational_window" in cfg.augmentations:
        augmentations.append(augmentations_class.observational_window)
    if "observed_n_stars" in cfg.augmentations:
        augmentations.append(augmentations_class.subsampling_to_observed_n_stars)
    if "mask_vlos" in cfg.augmentations:
        augmentations.append(augmentations_class.mask_vlos)
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


if __name__ == "__main__":
    main()