"""
Build the LaTeX results table (tab:results) of posterior constraints on the
global Milky Way potential parameters.

For every parameter we report:
    * the central value  = mode of the marginal posterior, estimated as the
      argmax of a Gaussian-KDE approximation (this is the "median" placeholder
      in the table caption; the KDE mode is the more robust central estimate);
    * the asymmetric uncertainty given by the 16th and 84th percentiles, i.e.
      value^{+ (p84 - mode)}_{- (mode - p16)}.

The 7 potential parameters are read, per stream, from the *_posterior.npz files
(Pal5, NGC3201, M68) and from global_posterior.npz (the compositional /
"Combined" inference).  The derived virial quantities M200 and R200 are read
from global_M200_R200.pdf.npz and are only available for the Combined column.

Usage
-----
    uv run python case_study5/project_stream/make_results_table.py \
        --posterior <global_posterior.npz> \
        --m200r200  <global_M200_R200.pdf.npz>
"""

import argparse
from math import floor, log10
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.stats import gaussian_kde

# ── observed rotation curve (Zhou et al. 2023), hardcoded as in the pipeline ──
OBS_R = np.array([5.24, 5.74, 6.25, 6.77, 7.23, 7.83, 8.21, 8.78, 9.26, 9.75,
                  10.25, 10.75, 11.25, 11.75, 12.24, 12.74, 13.25, 13.74, 14.23, 14.74,
                  15.23, 15.74, 16.24, 16.74, 17.23, 17.74, 18.35, 18.90, 19.50, 20.41,
                  21.28, 22.39, 23.16, 24.00])          # kpc
OBS_SVC = np.array([0.69, 0.68, 0.62, 0.60, 0.45, 0.29, 0.26, 0.22, 0.17, 0.16,
                    0.17, 0.18, 0.19, 0.20, 0.25, 0.27, 0.27, 0.31, 0.40, 0.43,
                    0.50, 0.68, 0.74, 0.87, 1.02, 1.15, 1.45, 1.58, 1.32, 1.71,
                    1.69, 2.01, 2.50, 4.94])            # km/s
OBS_VC = np.array([225.10, 233.53, 234.30, 233.17, 236.19, 236.00, 233.19, 233.15, 232.15, 231.24,
                   230.34, 230.54, 229.11, 227.48, 226.69, 225.56, 224.90, 223.57, 221.10, 220.19,
                   219.59, 217.36, 216.61, 217.28, 216.25, 213.81, 217.53, 212.10, 210.46, 206.69,
                   207.71, 203.72, 205.20, 200.64])     # km/s

# ── Huang et al. 2016 rotation curve: columns r [kpc], Vc [km/s], sigma_Vc ────
# Three tracer samples (HI, PRCG, HKG) combined for full radial coverage.
_HUANG = np.array([
    [4.60, 231.24, 7.00], [5.08, 230.46, 7.00], [5.58, 230.01, 7.00],
    [6.10, 239.61, 7.00], [6.57, 246.27, 7.00], [7.07, 243.49, 7.00],
    [7.58, 242.71, 7.00], [8.04, 243.23, 7.00],
    [8.34, 239.89, 5.92], [8.65, 237.26, 6.29], [9.20, 235.30, 5.60],
    [9.62, 230.99, 5.49], [10.09, 228.41, 5.62], [10.58, 224.26, 5.87],
    [11.09, 224.94, 7.02], [11.58, 233.57, 7.65], [12.07, 240.02, 6.17],
    [12.73, 242.21, 8.64], [13.72, 261.78, 14.89], [14.95, 259.26, 30.84],
    [15.52, 268.57, 49.67], [16.55, 261.17, 50.91], [17.56, 240.66, 49.91],
    [18.54, 215.31, 24.80], [19.50, 214.99, 24.42], [21.25, 251.68, 19.50],
    [23.78, 259.65, 19.62], [26.22, 242.02, 18.66], [28.71, 224.11, 16.97],
    [31.29, 211.20, 16.43], [33.73, 217.93, 17.66], [36.19, 219.33, 18.44],
    [38.73, 213.31, 17.29], [41.25, 200.05, 17.72], [43.93, 190.15, 18.65],
    [46.43, 198.95, 20.70], [48.71, 192.91, 19.24], [51.56, 198.90, 21.74],
    [57.03, 185.88, 21.56], [62.55, 173.89, 22.87], [69.47, 196.36, 25.89],
    [79.27, 175.05, 22.71], [98.97, 147.72, 23.55],
])
HUANG_R, HUANG_VC, HUANG_SVC = _HUANG.T

# ── default inputs ───────────────────────────────────────────────────────────
_DATA_DIR = Path(
    "/export/data/vgiusepp/latest_bayesflow/diffusion-experiments/"
    "case_study5/project_stream/data/hyperparameter_tuning/agama/"
    "rotationcurve/model_5/gaiastreams"
)
DEFAULT_POSTERIOR = _DATA_DIR / "global_posterior.npz"
DEFAULT_M200R200 = _DATA_DIR / "global_M200_R200.pdf.npz"

# local (per-stream) kinematic posterior: a single file holding all streams
# stacked along axis 1.  Unlike the global potential parameters, these are
# inferred per stream, so they have a value in each stream column but none in
# the compositional "Combined" column.
DEFAULT_LOCAL_POSTERIOR = Path(
    "/export/data/vgiusepp/latest_bayesflow/diffusion-experiments/"
    "case_study5/project_stream/data/hyperparameter_tuning/agama/local/"
    "rotationcurve/model_base_250epochs/global_5/gaia_local_posterior.npz"
)

# ── parameter rows: (npz key, latex label, unit factor) ──────────────────────
# `factor` multiplies the stored value to reach the unit shown in the table.
PARAMS = [
    ("rho_TwoPowerTriaxial_halo", r"$\rho_{\mathrm{NFW}}$ [$\mathrm{M_\odot\,kpc^{-3}}$]", 1.0),
    ("gamma_TwoPowerTriaxial_halo", r"$\gamma_{\mathrm{NFW}}$", 1.0),
    ("a_TwoPowerTriaxial_halo", r"$a_{\mathrm{NFW}}$ [kpc]", 1.0),
    ("q_TwoPowerTriaxial_halo", r"$q_{\mathrm{NFW}}$", 1.0),
    ("r_Disk", r"$r_d$ [kpc]", 1.0),
    ("z_Disk", r"$z_d$ [kpc]", 1.0),
    ("Sigma_Disk", r"$\Sigma_d$ [$\mathrm{M_\odot\,kpc^{-2}}$]", 1.0),  # native agama unit
]

# columns: (header, posterior filename relative to the global one)
STREAM_FILES = {
    "Pal~5": "Pal5_posterior.npz",
    "NGC~3201": "NGC3201_posterior.npz",
    "M68": "M68_posterior.npz",
    "Combined": "global_posterior.npz",
}

# local (per-stream) kinematic parameter rows: (npz key, latex label, factor).
# Read from the single local posterior file, sliced per stream.
LOCAL_PARAMS = [
    ("vr", r"$v_r$ [$\mathrm{km\,s^{-1}}$]", 1.0),
    ("r", r"$R$ [kpc]", 1.0),
    ("mu_ra_cosdec", r"$\mu_{\alpha*}$ [$\mathrm{mas\,yr^{-1}}$]", 1.0),
    ("mu_dec", r"$\mu_{\delta}$ [$\mathrm{mas\,yr^{-1}}$]", 1.0),
]

# stream-axis index (axis 1) of the local posterior for each table column.
# Combined has no local counterpart -> None (rendered as a dash).
STREAM_INDEX = {
    "Pal~5": 0,
    "NGC~3201": 1,
    "M68": 2,
    "Combined": None,
}


def kde_mode(samples):
    """Argmax of a Gaussian-KDE approximation of the marginal posterior."""
    samples = np.asarray(samples, dtype=float).ravel()
    samples = samples[np.isfinite(samples)]
    if samples.size < 2 or np.allclose(samples, samples[0]):
        return float(np.median(samples))
    kde = gaussian_kde(samples)
    grid = np.linspace(samples.min(), samples.max(), 4096)
    return float(grid[np.argmax(kde(grid))])


def summarize(samples, factor=1.0, label=""):
    """Return (central, minus_err, plus_err) scaled by `factor`.

    Central value is the KDE mode. If the mode falls outside the [p16, p84]
    band (typically a marginal piling up against a prior bound), the asymmetric
    errors would turn negative, so we fall back to the median as the central
    value and warn.
    """
    samples = np.asarray(samples, dtype=float).ravel() * factor
    p16, p50, p84 = np.percentile(samples, [16, 50, 84])
    central = kde_mode(samples)
    if not (p16 <= central <= p84):
        print(f"[warn] {label}: KDE mode {central:.4g} outside [{p16:.4g}, "
              f"{p84:.4g}] (boundary pile-up); using median {p50:.4g} instead")
        central = p50
    return central, central - p16, p84 - central


def fmt(stat):
    r"""Format (mode, minus, plus) as a LaTeX $v^{+p}_{-m}$ cell."""
    if stat is None:
        return r"-"
    mode, minus, plus = stat
    scale = max(abs(mode), 1e-300)
    exp = int(floor(log10(scale)))

    if exp >= 4 or exp <= -3:
        # scientific notation with a common exponent
        f = 10.0 ** exp
        m, lo, hi = mode / f, minus / f, plus / f
        return (
            rf"${m:.2f}^{{+{hi:.2f}}}_{{-{lo:.2f}}}\times10^{{{exp}}}$"
        )

    # fixed notation: choose decimals so the smaller error keeps 2 sig figs
    smaller = min(e for e in (abs(minus), abs(plus)) if e > 0) if (minus or plus) else abs(mode)
    if smaller > 0 and np.isfinite(smaller):
        dec = max(0, -(int(floor(log10(smaller))) - 1))
    else:
        dec = 2
    dec = min(dec, 4)
    return rf"${mode:.{dec}f}^{{+{plus:.{dec}f}}}_{{-{minus:.{dec}f}}}$"


# names of the three physical components of the rotation curve, in draw order
COMPONENTS = ("bulge", "halo", "disk")


def _component_dicts(p):
    """Return the agama parameter dict of each component (bulge, halo, disk).

    Identical definitions to agama_vcirc_worker.py; isolating them here lets us
    build either the full composite potential or one component at a time so the
    rotation curve can be split per component.
    """
    return {
        "bulge": dict(type='Spheroid',
                      scaleRadius=75 / 1e3, densityNorm=9.6e10,
                      gamma=0, alpha=1, beta=1.8, cutoffStrength=2,
                      outerCutoffRadius=2.1, axisRatioY=1.0, axisRatioZ=0.5),
        "halo": dict(type='Spheroid',
                     scaleRadius=p['a_TwoPowerTriaxial_halo'],
                     densityNorm=p['rho_TwoPowerTriaxial_halo'],
                     gamma=p['gamma_TwoPowerTriaxial_halo'],
                     alpha=1, beta=3, cutoffStrength=2, outerCutoffRadius=np.inf,
                     axisRatioY=1.0, axisRatioZ=p['q_TwoPowerTriaxial_halo']),
        "disk": dict(type='Disk',
                     scaleRadius=p['r_Disk'], scaleHeight=p['z_Disk'],
                     surfaceDensity=p['Sigma_Disk'],
                     sersicIndex=1, innerCutoffRadius=0),
    }


def _build_full_potential(p):
    """Bulge + halo + disk potential, identical to agama_vcirc_worker.py."""
    import agama
    comps = _component_dicts(p)
    return agama.Potential(*(comps[name] for name in COMPONENTS))


def vcirc_at(p, R):
    """Circular velocity [km/s] at radii R [kpc] for parameter dict p."""
    R = np.asarray(R, dtype=float)
    pot = _build_full_potential(p)
    points = np.column_stack((R, np.zeros_like(R), np.zeros_like(R)))
    v2 = -R * pot.force(points)[:, 0]
    return np.sqrt(v2)


# default spherical radius [kpc] for the total enclosed-mass column
MASS_RADIUS_KPC = 20.0


def enclosed_mass_samples(posterior, R_kpc, n_samples=None, seed=0):
    """Total mass [Msun] inside a sphere of radius ``R_kpc`` for every posterior
    draw of the full (bulge + halo + disk) potential.

    Uses agama's ``Potential.enclosedMass`` (spherical enclosed mass) on the
    same composite potential as the rotation curve, so this is the total
    dynamical mass within ``R_kpc``.  All 7 potential parameters are drawn from
    the posterior (the bulge is fixed).  Non-finite / failed draws are dropped.
    """
    import agama
    agama.setUnits(length=1, velocity=1, mass=1)
    keys = [k for k, _, _ in PARAMS]
    samples = np.column_stack([posterior[k].ravel() for k in keys])  # (Ntot, 7)
    ntot = samples.shape[0]
    n = ntot if n_samples is None else min(n_samples, ntot)
    idx = (np.arange(ntot) if n == ntot
           else np.random.default_rng(seed).choice(ntot, size=n, replace=False))

    out = np.full(n, np.nan)
    for c, i in enumerate(idx):
        p = {k: float(samples[i, j]) for j, k in enumerate(keys)}
        try:
            out[c] = float(_build_full_potential(p).enclosedMass(R_kpc))
        except Exception:
            continue
    out = out[np.isfinite(out)]
    print(f"    M(<{R_kpc:g} kpc): used {out.size}/{n} posterior draws")
    return out


def write_mass_within(mass_samples, R_kpc, out_dir):
    """Persist the total enclosed-mass-within-R distributions per column.

    ``mass_samples`` maps each table column to its array of M(<R) draws (or
    None).  Writes ``mass_within_<R>kpc.txt`` / ``.npz`` into ``out_dir`` with
    the 16/50/84th percentiles and the KDE-mode summary, in 10^11 Msun.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = (16, 50, 84)
    L = []
    L.append(f"Total enclosed mass within a sphere of radius R = {R_kpc:g} kpc")
    L.append("=" * 70)
    L.append("M(<R) = enclosedMass(R) of the full bulge+halo+disk potential,")
    L.append("per inference column (all 7 potential parameters drawn from the")
    L.append("posterior; the bulge is fixed).  Values in 10^11 Msun.")
    L.append("")
    L.append(f"  {'column':<12}{'p16':>12}{'p50':>12}{'p84':>12}"
             f"{'KDE mode':>12}{'+err':>12}{'-err':>12}")
    save = {}
    for col, m in mass_samples.items():
        if m is None or len(m) == 0:
            continue
        p16, p50, p84 = np.percentile(m, levels) / 1e11
        mode, minus, plus = summarize(m, 1.0e-11, f"{col}/M(<{R_kpc:g}kpc)")
        L.append(f"  {col:<12}{p16:>12.4f}{p50:>12.4f}{p84:>12.4f}"
                 f"{mode:>12.4f}{plus:>12.4f}{minus:>12.4f}")
        save[f"M_within_samples_{col}"] = m
    txt = "\n".join(L) + "\n"
    txt_path = out_dir / f"mass_within_{R_kpc:g}kpc.txt"
    txt_path.write_text(txt)
    np.savez(out_dir / f"mass_within_{R_kpc:g}kpc.npz", R_kpc=R_kpc, **save)
    print(txt)
    print(f"Enclosed mass within {R_kpc:g} kpc written to:\n    {txt_path}")


def vcirc_components_at(p, R):
    """Per-component circular velocity [km/s] at radii R [kpc].

    Returns a dict with one curve per component (bulge, halo, disk) plus the
    combined ``total``.  Because forces add linearly, the total equals the
    quadrature sum of the components: v_tot^2 = v_bulge^2 + v_halo^2 + v_disk^2.
    """
    import agama
    R = np.asarray(R, dtype=float)
    points = np.column_stack((R, np.zeros_like(R), np.zeros_like(R)))
    comps = _component_dicts(p)
    out, v2_tot = {}, np.zeros_like(R)
    for name in COMPONENTS:
        pot = agama.Potential(comps[name])
        v2 = -R * pot.force(points)[:, 0]
        v2_tot = v2_tot + v2
        out[name] = np.sqrt(np.clip(v2, 0.0, None))
    out["total"] = np.sqrt(np.clip(v2_tot, 0.0, None))
    return out


def vcirc_ensemble(combined, R_grid, n_samples=None, seed=0):
    """Per-radius (p16, p50, p84) of vcirc over Combined posterior draws.

    n_samples=None uses every available posterior sample.
    """
    keys = [k for k, _, _ in PARAMS]
    samples = np.column_stack([combined[k].ravel() for k in keys])  # (Ntot, 7)
    n = samples.shape[0] if n_samples is None else min(n_samples, samples.shape[0])
    idx = (np.arange(samples.shape[0]) if n == samples.shape[0]
           else np.random.default_rng(seed).choice(samples.shape[0], size=n, replace=False))

    curves = []
    for i in idx:
        p = {k: float(samples[i, j]) for j, k in enumerate(keys)}
        try:
            v = vcirc_at(p, R_grid)
        except Exception:
            continue
        if np.all(np.isfinite(v)):
            curves.append(v)
    curves = np.asarray(curves)
    print(f"    used {len(curves)}/{n} posterior draws for the band")
    p16, p50, p84 = np.percentile(curves, [16, 50, 84], axis=0)
    return p16, p50, p84


def vcirc_components_ensemble(combined, R_grid, n_samples=None, seed=0):
    """Per-radius (p16, p50, p84) of vcirc for each component over draws.

    Returns dict[name] -> (p16, p50, p84) arrays for name in
    bulge/halo/disk/total.  Draws that fail or produce non-finite curves are
    skipped consistently across all components.
    """
    keys = [k for k, _, _ in PARAMS]
    samples = np.column_stack([combined[k].ravel() for k in keys])  # (Ntot, 7)
    n = samples.shape[0] if n_samples is None else min(n_samples, samples.shape[0])
    idx = (np.arange(samples.shape[0]) if n == samples.shape[0]
           else np.random.default_rng(seed).choice(samples.shape[0], size=n, replace=False))

    names = COMPONENTS + ("total",)
    curves = {name: [] for name in names}
    for i in idx:
        p = {k: float(samples[i, j]) for j, k in enumerate(keys)}
        try:
            vc = vcirc_components_at(p, R_grid)
        except Exception:
            continue
        if all(np.all(np.isfinite(vc[name])) for name in names):
            for name in names:
                curves[name].append(vc[name])
    print(f"    used {len(curves['total'])}/{n} posterior draws for the component bands")
    return {name: np.percentile(np.asarray(curves[name]), [16, 50, 84], axis=0)
            for name in names}


# per-component styles for the split rotation curve (the combined/total curve
# keeps the original blue line + band).  Colours chosen to sit nicely next to
# the blue total and the crimson/purple data points; halo and disk also get a
# 16th--84th band from the posterior samples (like the total), bulge stays a
# plain line.
COMPONENT_STYLE = {
    "halo":  dict(color="seagreen",   ls="-", label="Halo",  band=True),
    "disk":  dict(color="darkorange", ls="-", label="Disk",  band=True),
    "bulge": dict(color="0.45",       ls=":", label="Bulge", band=False),
}


def plot_vcirc(combined, out_pdf):
    """Rotation curve (Combined posterior) vs Zhou+2023 and Huang+2016, split
    per component.

    Mirrors the split linear/log two-panel layout of notebooks/paper_plots.py.
    The combined curve is unchanged: solid blue line = per-radius median of the
    rotation-curve ensemble, shaded band = its 16th--84th percentile.  In
    addition the halo, disk and bulge contributions are drawn as their own
    median lines (with a matching 16th--84th band for the halo and disk) so the
    decomposition v_tot^2 = v_halo^2 + v_disk^2 + v_bulge^2 is visible.
    """
    import agama
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    agama.setUnits(length=1, velocity=1, mass=1)

    split = 30.0
    R_grid = np.logspace(np.log10(0.1), np.log10(100.0), 400)
    # per-radius p50 / p16 / p84 of the rotation-curve ensemble over all
    # posterior samples, for the total and for each component (guarantees the
    # median curve sits inside its own 16th--84th band)
    ens = vcirc_components_ensemble(combined, R_grid)
    band_lo, v_med, band_hi = ens["total"]

    fig, (ax_lin, ax_log) = plt.subplots(
        1, 2, sharey=True, figsize=(5, 3),
        gridspec_kw={"width_ratios": [3, 2], "wspace": 0},
    )

    # the model curve is drawn past the split on each side (xlim clips it
    # invisibly) so there is no gap at the linear/log boundary
    map_extend = 35.0       # linear panel draws the curve up to this radius
    map_extend_log = 28.0   # log panel draws the curve down to this radius
    for ax, xmin, xmax, xscale, mlo, mhi in [
        (ax_lin, 0.1, split, "linear", 0.1, map_extend),
        (ax_log, split, 99.9, "log", map_extend_log, 99.9),
    ]:
        m = (R_grid >= mlo) & (R_grid <= mhi)
        ax.plot(R_grid[m], v_med[m], color='blue', lw=1.5,
                label='Global posterior median')
        ax.fill_between(R_grid[m], band_lo[m], band_hi[m],
                        alpha=0.3, color='blue', lw=0,
                        )

        # per-component curves: median line for each, plus a 16th--84th band
        # (from the posterior samples) for halo and disk
        for name, style in COMPONENT_STYLE.items():
            lo, mid, hi = ens[name]
            ax.plot(R_grid[m], mid[m], lw=1.3, color=style["color"],
                    ls=style["ls"], label=style["label"])
            if style["band"]:
                ax.fill_between(R_grid[m], lo[m], hi[m],
                                alpha=0.25, color=style["color"], lw=0)

        mz = (OBS_R >= xmin) & (OBS_R <= xmax)
        ax.errorbar(OBS_R[mz], OBS_VC[mz], yerr=OBS_SVC[mz],
                    fmt='o', color='crimson', ms=1.5, lw=1, capsize=1,
                    label='Zhou et al. 2023')

        # only show Huang points beyond R > 30 kpc
        mh = (HUANG_R >= xmin) & (HUANG_R <= xmax) & (HUANG_R > 30.0)
        ax.errorbar(HUANG_R[mh], HUANG_VC[mh], yerr=HUANG_SVC[mh],
                    fmt='o', color='purple', ms=1.5, lw=1, capsize=1,
                    label='Huang et al. 2016')

        ax.set_xscale(xscale)
        ax.set_xlim(xmin, xmax)
        # lower bound dropped to 0 so the sub-dominant components stay visible
        ax.set_ylim(0, 260)

    # tick / spine cosmetics to match the paper figure
    ax_log.set_xticks([40, 60, 80, 100])
    ax_log.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax_log.xaxis.set_minor_locator(mticker.NullLocator())
    ax_lin.set_xticks([5, 15, 25, 30])
    ax_lin.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax_lin.spines["right"].set_visible(False)
    ax_log.spines["left"].set_visible(False)
    ax_log.tick_params(axis='y', left=False)
    ax_lin.axvline(split, color='k', lw=1.5, ls='--', clip_on=False, zorder=5)

    fig.supxlabel('$R$ [kpc]', fontsize=12, x=0.55, y=0.1)
    ax_lin.set_ylabel('$V_C$ [km/s]', fontsize=12)

    handles, labels = [], []
    for ax in (ax_lin, ax_log):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    # legend in the (now empty) lower-left corner of the linear panel; two
    # columns keep the six entries compact
    ax_lin.legend(handles, labels, loc='lower left',
                  bbox_to_anchor=(0.02, 0.02), bbox_transform=ax_lin.transAxes,
                  fontsize=6, ncol=2, columnspacing=1.0, handlelength=1.6,
                  framealpha=0.85)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0)
    fig.savefig(out_pdf, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nRotation curve written to: {out_pdf}")


# halo parameters M200/R200 depend on (the spherical-overdensity criterion is
# evaluated on the halo-only potential).
HALO_KEYS = ("rho_TwoPowerTriaxial_halo", "a_TwoPowerTriaxial_halo",
             "gamma_TwoPowerTriaxial_halo", "q_TwoPowerTriaxial_halo")


def _import_worker():
    """Import agama_vcirc_worker from this script's directory."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import agama_vcirc_worker as worker
    return worker


# radial grid (kpc) for the ellipsoidal-mass solver, shared across samples
_ELL_GRID = np.logspace(np.log10(1e-2), np.log10(3000.0), 4000)


def m200_r200_spherical(p, H0):
    """Pipeline definition (agama_vcirc_worker._compute_m200): r200 solves
    M_sphere(<r) / (4/3 pi r^3) = 200 rho_crit(H0), with M_sphere the halo mass
    inside a *sphere* of radius r.  Returns (R200, M200)."""
    return _import_worker()._compute_m200(p, H0)


def m200_r200_ellipsoidal(p, rho_crit, c=200.0):
    r"""Virial-mass definition of the stream papers (H0 = 71 km/s/Mpc,
    rho_crit = 3 H0^2 / (8 pi G) = 140 Msun/kpc^3):

        M200 = (4pi/3) r200^3 c rho_crit = 4pi q_h \int_0^r200 s^2 rho_h(s) ds

    so r200 is the major-axis radius at which the mean density of the flattened
    halo equals c rho_crit, and the enclosed mass is the *ellipsoidal* mass of
    the halo (note the q_h prefactor).  rho_h(s) is the halo density along the
    major axis,

        rho_h(s) = rho0 (s/a)^-gamma (1 + s/a)^(gamma-3)     (alpha=1, beta=3,
    no outer cutoff), which matches the agama Spheroid used elsewhere.  Returns
    (R200, M200); NaNs when the overdensity criterion has no root on the grid.
    """
    rho0 = p["rho_TwoPowerTriaxial_halo"]
    a    = p["a_TwoPowerTriaxial_halo"]
    gam  = p["gamma_TwoPowerTriaxial_halo"]
    q    = p["q_TwoPowerTriaxial_halo"]
    S = _ELL_GRID
    rho = rho0 * (S / a) ** (-gam) * (1.0 + S / a) ** (gam - 3.0)
    m_ell = 4.0 * np.pi * q * cumulative_trapezoid(S ** 2 * rho, S, initial=0.0)
    mean = m_ell / (4.0 / 3.0 * np.pi * S ** 3)
    g = mean - c * rho_crit                  # decreasing; first downward crossing
    k = np.where((g[:-1] > 0) & (g[1:] <= 0))[0]
    if k.size == 0:
        return np.nan, np.nan
    i = k[0]
    x0, x1 = np.log(S[i]), np.log(S[i + 1])
    R200 = float(np.exp(x0 + (0.0 - g[i]) * (x1 - x0) / (g[i + 1] - g[i])))
    return R200, float(4.0 / 3.0 * np.pi * R200 ** 3 * c * rho_crit)


def _plot_m200_r200_hist(R200_s, M200_s, out_pdf):
    """Two-panel R200 / M200 posterior histograms (median + 16th/84th vlines).

    Same style as notebooks/paper_plots.py 'M200_R200_final.pdf', except M200
    is shown linearly (in 10^12 Msun) instead of as log10(M200)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M12 = M200_s / 1e12
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
    for ax, data, color, xlabel in [
        (axes[0], R200_s, "turquoise", r"$R_{200}$ [kpc]"),
        (axes[1], M12, "aquamarine", r"$M_{200}$ [$10^{12}\,M_\odot$]"),
    ]:
        ax.hist(data, bins=40, color=color, lw=0.4, density=True)
        ax.axvline(np.median(data), color="k", lw=1.5)
        ax.axvline(np.percentile(data, 16), color="k", lw=1, ls="--")
        ax.axvline(np.percentile(data, 84), color="k", lw=1, ls="--")
        ax.set_xlabel(xlabel, fontsize=13)
    axes[0].set_ylabel("Normalized counts", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"M200/R200 histogram written to: {out_pdf}")


def recompute_m200_r200(combined, out_dir, per_sample, tag, meta,
                        m200r200_file=None, n_samples=None, seed=0,
                        make_plot=False):
    """Recompute M200/R200 over the posterior two independent ways and save it.

    ``per_sample(p)`` maps a halo-parameter dict to (R200, M200) under a chosen
    virial-mass definition (see m200_r200_spherical / m200_r200_ellipsoidal).
    ``meta`` carries the human-readable 'definition', 'H0' and 'rho_crit' for the
    report.  The two propagation methods are:

      * method 1 (sample)     : evaluate (R200, M200) for *every* posterior draw,
        then summarise the resulting distributions.  Statistically correct
        propagation of the posterior.
      * method 2 (percentile) : take the marginal 16/50/84th percentiles of each
        halo parameter, assemble a "p16"/"p50"/"p84" parameter vector and
        evaluate (R200, M200) once for each.  Cheap plug-in approach; comparing
        it with method 1 exposes how much it mis-states the spread, since M200
        depends non-linearly on several correlated halo parameters.

    Writes ``m200_r200_<tag>.npz`` / ``.txt`` into ``out_dir`` (and, if
    ``make_plot``, ``M200_R200_<tag>.pdf``); also returns a dict.
    """
    import agama
    agama.setUnits(length=1, velocity=1, mass=1)   # needed by the spherical fn
    out_dir.mkdir(parents=True, exist_ok=True)

    S = {k: combined[k].ravel() for k in HALO_KEYS}
    ntot = next(iter(S.values())).size
    n = ntot if n_samples is None else min(n_samples, ntot)
    idx = (np.arange(ntot) if n == ntot
           else np.random.default_rng(seed).choice(ntot, size=n, replace=False))

    # ── method 1: per-sample R200 / M200 over the posterior draws ─────────────
    print(f"\nRecomputing M200/R200 [{tag}] -- {meta['definition']}")
    R200_s = np.full(n, np.nan)
    M200_s = np.full(n, np.nan)
    for c, i in enumerate(idx):
        p = {k: float(S[k][i]) for k in HALO_KEYS}
        R200_s[c], M200_s[c] = per_sample(p)
    finite = np.isfinite(R200_s) & np.isfinite(M200_s)
    R200_s, M200_s = R200_s[finite], M200_s[finite]
    print(f"    method 1 (sample): used {finite.sum()}/{n} posterior draws")

    # ── method 2: plug the marginal percentiles of the halo parameters ────────
    levels = (16, 50, 84)
    param_pct = {k: dict(zip(levels, np.percentile(S[k], levels))) for k in HALO_KEYS}
    method2 = {}
    for lvl in levels:
        p = {k: float(param_pct[k][lvl]) for k in HALO_KEYS}
        method2[lvl] = per_sample(p)
    print("    method 2 (percentile): evaluated p16/p50/p84 parameter vectors")

    # ── precomputed values currently used by the table (for reference) ────────
    pre = None
    if m200r200_file is not None and Path(m200r200_file).exists():
        d = np.load(m200r200_file, allow_pickle=True)
        pre = dict(M200=np.percentile(d["M200"], levels),
                   R200=np.percentile(d["R200"], levels))

    # ── persist raw arrays ────────────────────────────────────────────────────
    npz_path = out_dir / f"m200_r200_{tag}.npz"
    np.savez(
        npz_path,
        H0=meta["H0"], rho_crit=meta["rho_crit"], definition=meta["definition"],
        M200_samples=M200_s, R200_samples=R200_s,
        levels=np.array(levels),
        M200_method2=np.array([method2[l][1] for l in levels]),
        R200_method2=np.array([method2[l][0] for l in levels]),
        **{f"param_pct_{k}": np.array([param_pct[k][l] for l in levels])
           for k in HALO_KEYS},
    )

    # ── human-readable summary ────────────────────────────────────────────────
    m_p16, m_p50, m_p84 = np.percentile(M200_s, levels) / 1e12  # 10^12 Msun
    r_p16, r_p50, r_p84 = np.percentile(R200_s, levels)         # kpc
    m_mode, m_minus, m_plus = summarize(M200_s, 1.0e-12, f"{tag}/M200")
    r_mode, r_minus, r_plus = summarize(R200_s, 1.0, f"{tag}/R200")

    L = []
    L.append("Recomputation of M200 and R200 from the Combined (global) posterior")
    L.append("=" * 70)
    L.append(f"tag        : {tag}")
    L.append(f"definition : {meta['definition']}")
    L.append(f"H0 = {meta['H0']:.5g} km/s/kpc ({meta['H0']*1e3:.1f} km/s/Mpc); "
             f"rho_crit = {meta['rho_crit']:.4g} Msun/kpc^3")
    L.append(f"posterior draws used: {finite.sum()}/{n} (out of {ntot} available)")
    L.append("")
    L.append("Marginal percentiles of the halo parameters (inputs to method 2):")
    L.append(f"    {'parameter':<32}{'p16':>14}{'p50':>14}{'p84':>14}")
    for k in HALO_KEYS:
        v16, v50, v84 = (param_pct[k][l] for l in levels)
        L.append(f"    {k:<32}{v16:>14.5g}{v50:>14.5g}{v84:>14.5g}")
    L.append("")
    L.append("M200 [10^12 Msun]:")
    L.append(f"    method 1 (sample)      p16/p50/p84 : "
             f"{m_p16:.4f} / {m_p50:.4f} / {m_p84:.4f}")
    L.append(f"    method 1 (sample)      KDE mode +/- : "
             f"{m_mode:.4f}  (+{m_plus:.4f} / -{m_minus:.4f})")
    L.append(f"    method 2 (percentile)  p16/p50/p84 : "
             f"{method2[16][1]/1e12:.4f} / {method2[50][1]/1e12:.4f} / "
             f"{method2[84][1]/1e12:.4f}")
    if pre is not None:
        L.append(f"    precomputed file       p16/p50/p84 : "
                 f"{pre['M200'][0]/1e12:.4f} / {pre['M200'][1]/1e12:.4f} / "
                 f"{pre['M200'][2]/1e12:.4f}")
    L.append("")
    L.append("R200 [kpc]:")
    L.append(f"    method 1 (sample)      p16/p50/p84 : "
             f"{r_p16:.2f} / {r_p50:.2f} / {r_p84:.2f}")
    L.append(f"    method 1 (sample)      KDE mode +/- : "
             f"{r_mode:.2f}  (+{r_plus:.2f} / -{r_minus:.2f})")
    L.append(f"    method 2 (percentile)  p16/p50/p84 : "
             f"{method2[16][0]:.2f} / {method2[50][0]:.2f} / {method2[84][0]:.2f}")
    if pre is not None:
        L.append(f"    precomputed file       p16/p50/p84 : "
                 f"{pre['R200'][0]:.2f} / {pre['R200'][1]:.2f} / {pre['R200'][2]:.2f}")
    L.append("")
    L.append("(precomputed file = global_M200_R200.pdf.npz, the values the "
             "results table reports; spherical mass, H0 ~ 67.4)")
    txt = "\n".join(L) + "\n"

    txt_path = out_dir / f"m200_r200_{tag}.txt"
    txt_path.write_text(txt)
    print(txt)
    print(f"M200/R200 recomputation written to:\n    {npz_path}\n    {txt_path}")

    if make_plot:
        _plot_m200_r200_hist(R200_s, M200_s, out_dir / f"M200_R200_{tag}.pdf")

    return dict(M200_samples=M200_s, R200_samples=R200_s,
                method2=method2, param_pct=param_pct, meta=meta)


# Galactocentric solar radius [kpc] for the local halo density (ro = 8.122,
# the value adopted in the stream-analysis notebooks).
SOLAR_RADIUS_KPC = 8.122
# 1 Msun/kpc^3 expressed in GeV/cm^3 (the usual local dark-matter-density unit).
MSUN_KPC3_TO_GEV_CM3 = 3.7978e-8
# 1 Msun/kpc^3 expressed in Msun/pc^3.
MSUN_KPC3_TO_MSUN_PC3 = 1.0e-9


def halo_density_major_axis(rho0, a, gamma, s):
    """Halo density [Msun/kpc^3] along the major axis (z=0) at radius s [kpc]:

        rho_h(s) = rho0 (s/a)^-gamma (1 + s/a)^(gamma-3)   (alpha=1, beta=3, no
    cutoff), matching the agama Spheroid used elsewhere.  On the major axis the
    flattening q drops out, and the Sun sits at z ~ 0, so this is the local halo
    (dark-matter) density.  Vectorised over array inputs.
    """
    return rho0 * (s / a) ** (-gamma) * (1.0 + s / a) ** (gamma - 3.0)


def recompute_halo_solar_density(combined, out_dir, R_sun=SOLAR_RADIUS_KPC,
                                 n_samples=None, seed=0):
    """Halo (dark-matter) density at the solar radius, rho_h(R_sun), propagated
    two ways and saved.  It is independent of the M200/R200 virial definition
    (just the density profile), and reported in both Msun/kpc^3 and GeV/cm^3:

      * method 1 (sample)     : rho_h(R_sun) for every posterior draw -> summary.
      * method 2 (percentile) : rho_h(R_sun) from the p16/p50/p84 parameter
        vectors of (rho0, a, gamma).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ("rho_TwoPowerTriaxial_halo", "a_TwoPowerTriaxial_halo",
            "gamma_TwoPowerTriaxial_halo")
    S = {k: combined[k].ravel() for k in keys}
    ntot = next(iter(S.values())).size
    n = ntot if n_samples is None else min(n_samples, ntot)
    idx = (np.arange(ntot) if n == ntot
           else np.random.default_rng(seed).choice(ntot, size=n, replace=False))

    # method 1: full-sample distribution (vectorised)
    rho_sun = halo_density_major_axis(
        S["rho_TwoPowerTriaxial_halo"][idx], S["a_TwoPowerTriaxial_halo"][idx],
        S["gamma_TwoPowerTriaxial_halo"][idx], R_sun)
    rho_sun = rho_sun[np.isfinite(rho_sun)]

    # method 2: percentile plug-in of each parameter
    levels = (16, 50, 84)
    pct = {k: np.percentile(S[k], levels) for k in keys}
    method2 = {lvl: float(halo_density_major_axis(
        pct["rho_TwoPowerTriaxial_halo"][j], pct["a_TwoPowerTriaxial_halo"][j],
        pct["gamma_TwoPowerTriaxial_halo"][j], R_sun))
        for j, lvl in enumerate(levels)}

    p16, p50, p84 = np.percentile(rho_sun, levels)
    mode, minus, plus = summarize(rho_sun, 1.0, "halo/rho_sun")
    pc = MSUN_KPC3_TO_MSUN_PC3      # Msun/kpc^3 -> Msun/pc^3
    g = MSUN_KPC3_TO_GEV_CM3        # Msun/kpc^3 -> GeV/cm^3

    def _row(label, v):
        return f"  {label:<26}{v*pc:>13.5f}{v:>14.4g}{v*g:>12.4f}"

    L = []
    L.append(f"Halo (dark-matter) density at the solar radius R_sun = {R_sun:.3f} kpc")
    L.append("=" * 70)
    L.append("rho_h(R_sun) = rho0 (R/a)^-gamma (1+R/a)^(gamma-3)  [major axis, z=0]")
    L.append(f"posterior draws used: {rho_sun.size}/{n} (out of {ntot} available)")
    L.append("")
    L.append(f"  {'estimate':<26}{'Msun/pc^3':>13}{'Msun/kpc^3':>14}{'GeV/cm^3':>12}")
    L.append(_row("method 1 (sample) p16", p16))
    L.append(_row("method 1 (sample) p50", p50))
    L.append(_row("method 1 (sample) p84", p84))
    L.append(_row("method 1 KDE mode", mode))
    L.append(f"      (KDE asym. err:  +{plus*pc:.5f} / -{minus*pc:.5f} Msun/pc^3,"
             f"   +{plus*g:.4f} / -{minus*g:.4f} GeV/cm^3)")
    L.append(_row("method 2 (pct) p16-params", method2[16]))
    L.append(_row("method 2 (pct) p50-params", method2[50]))
    L.append(_row("method 2 (pct) p84-params", method2[84]))
    L.append("")
    txt = "\n".join(L) + "\n"

    txt_path = out_dir / "halo_solar_density.txt"
    txt_path.write_text(txt)
    np.savez(out_dir / "halo_solar_density.npz",
             R_sun=R_sun, rho_sun_samples=rho_sun,
             levels=np.array(levels),
             rho_sun_sample_pct_msun_kpc3=np.array([p16, p50, p84]),
             rho_sun_method2_msun_kpc3=np.array([method2[l] for l in levels]),
             msun_kpc3_to_msun_pc3=pc, msun_kpc3_to_gev_cm3=g)
    print(txt)
    print(f"Halo solar density written to:\n    {txt_path}")
    return dict(rho_sun_samples=rho_sun, method2=method2, R_sun=R_sun)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--posterior", type=Path, default=DEFAULT_POSTERIOR,
                    help="global_posterior.npz (defines the data directory)")
    ap.add_argument("--m200r200", type=Path, default=DEFAULT_M200R200,
                    help="global_M200_R200.pdf.npz")
    ap.add_argument("--local-posterior", type=Path, default=DEFAULT_LOCAL_POSTERIOR,
                    help="gaia_local_posterior.npz with per-stream local parameters")
    ap.add_argument("--out", type=Path, default=None,
                    help="output .tex path (default: results_table.tex next to posterior)")
    ap.add_argument("--results-subdir", type=str, default="recomputed_results",
                    help="subfolder of the data dir for the split rotation curve "
                         "and the M200/R200 recomputation")
    ap.add_argument("--m200-nsamples", type=int, default=None,
                    help="cap the number of posterior draws used to recompute "
                         "M200/R200 (default: use all)")
    ap.add_argument("--solar-radius", type=float, default=SOLAR_RADIUS_KPC,
                    help="Galactocentric solar radius [kpc] for the local halo "
                         f"density (default: {SOLAR_RADIUS_KPC})")
    ap.add_argument("--mass-radius", type=float, default=MASS_RADIUS_KPC,
                    help="spherical radius [kpc] for the total enclosed-mass "
                         f"row M(<R) (default: {MASS_RADIUS_KPC})")
    ap.add_argument("--mass-nsamples", type=int, default=None,
                    help="cap the number of posterior draws used for the total "
                         "enclosed-mass row (default: use all)")
    args = ap.parse_args()

    data_dir = args.posterior.parent
    out_path = args.out or (data_dir / "results_table.tex")

    columns = list(STREAM_FILES.keys())

    # load every stream's posterior
    posteriors = {}
    for col, fname in STREAM_FILES.items():
        path = data_dir / fname
        if path.exists():
            posteriors[col] = np.load(path, allow_pickle=True)
        else:
            print(f"[warn] missing {path}; column '{col}' left as placeholder")
            posteriors[col] = None

    # collect stats[param_key or 'M200'/'R200'][column] = (mode, minus, plus) or None
    stats = {}
    for key, _, factor in PARAMS:
        stats[key] = {}
        for col in columns:
            d = posteriors[col]
            stats[key][col] = summarize(d[key], factor, f"{col}/{key}") if d is not None else None

    # derived total disk mass of the exponential disk (sersicIndex=1,
    # innerCutoffRadius=0):  M_disk = 2 pi Sigma_0 R_d^2  [M_sun].
    # Computed per stream from each posterior (Sigma_Disk, r_Disk both present).
    stats["Mdisk"] = {}
    for col in columns:
        d = posteriors[col]
        if d is not None:
            mdisk = 2.0 * np.pi * d["Sigma_Disk"].ravel() * d["r_Disk"].ravel() ** 2
            stats["Mdisk"][col] = summarize(mdisk, 1.0e-10, f"{col}/Mdisk")  # 10^10 M_sun
        else:
            stats["Mdisk"][col] = None

    # local (per-stream) kinematic parameters: a single file holds all streams
    # stacked along axis 1 (Pal5=0, NGC3201=1, M68=2).  Values exist per stream
    # but not for the compositional Combined column.
    if args.local_posterior.exists():
        local = np.load(args.local_posterior, allow_pickle=True)
    else:
        print(f"[warn] missing {args.local_posterior}; local rows left as placeholders")
        local = None
    for key, _, factor in LOCAL_PARAMS:
        stats[key] = {}
        for col in columns:
            j = STREAM_INDEX.get(col)
            if local is not None and j is not None and key in local.files:
                stats[key][col] = summarize(local[key][0, j].ravel(), factor,
                                            f"{col}/{key}")
            else:
                stats[key][col] = None

    # M200 / R200: only the Combined (global) inference is available
    m200r200 = np.load(args.m200r200, allow_pickle=True)
    stats["M200"] = {c: None for c in columns}
    stats["R200"] = {c: None for c in columns}
    stats["M200"]["Combined"] = summarize(m200r200["M200"], 1.0e-12, "Combined/M200")  # 10^12 M_sun
    stats["R200"]["Combined"] = summarize(m200r200["R200"], 1.0, "Combined/R200")       # kpc

    # derived total enclosed mass within R = mass_radius kpc of the full
    # (bulge+halo+disk) potential, computed per column from each posterior (all
    # 7 potential params are present in every column).  Sample arrays are kept
    # so the Combined block can write a detailed per-column summary file.
    R_mass = args.mass_radius
    print(f"\nTotal enclosed mass within R = {R_mass:g} kpc (full potential):")
    mass_samples = {}
    stats["Mwithin"] = {}
    for col in columns:
        d = posteriors[col]
        if d is not None:
            ms = enclosed_mass_samples(d, R_mass, n_samples=args.mass_nsamples)
            mass_samples[col] = ms
            stats["Mwithin"][col] = summarize(ms, 1.0e-11, f"{col}/M(<{R_mass:g}kpc)")  # 10^11 M_sun
        else:
            mass_samples[col] = None
            stats["Mwithin"][col] = None

    # ── plain-text summary to stdout ─────────────────────────────────────────
    print(f"\nData directory: {data_dir}\n")
    row_labels = [(k, lbl) for k, lbl, _ in PARAMS] + [
        ("Mdisk", r"Mdisk [10^10 Msun]"),
        ("M200", r"M200 [10^12 Msun]"), ("R200", r"R200 [kpc]"),
        ("Mwithin", f"M(<{R_mass:g} kpc) [10^11 Msun]")] + [
        (k, lbl) for k, lbl, _ in LOCAL_PARAMS]
    header = f"{'parameter':<28}" + "".join(f"{c:>26}" for c in columns)
    print(header)
    print("-" * len(header))
    for key, lbl in row_labels:
        cells = []
        for col in columns:
            s = stats[key][col]
            cells.append(f"{s[0]:.4g} (+{s[2]:.3g}/-{s[1]:.3g})" if s else "--")
        print(f"{lbl:<28}" + "".join(f"{c:>26}" for c in cells))

    # ── LaTeX table ──────────────────────────────────────────────────────────
    def row(label, key):
        return f"            {label:<52}& " + " & ".join(
            fmt(stats[key][c]) for c in columns) + r" \\[4pt]"

    lines = [
        r"\begin{table}",
        r"        \caption{Posterior constraints on the global Milky Way potential",
        r"            parameters from the Gaia data: median and 16th--84th percentile interval",
        r"            of the marginal posterior for each single-stream inference and for the",
        r"            compositional (combined) one. The middle rows report the derived total",
        r"            disk mass and virial quantities; the last rows report the per-stream",
        r"            local (kinematic) parameters, which have no compositional counterpart.}",
        r"        \label{tab:results}",
        r"        \centering",
        r"        \renewcommand{\arraystretch}{1.4}",
        r"        \begin{tabular}{l c c c c}",
        r"            \hline\hline",
        r"            Parameter                                            & Pal~5 & NGC~3201 & M68 & Combined \\[4pt]",
        r"            \hline",
    ]
    for key, lbl, _ in PARAMS:
        lines.append(row(lbl, key))
    lines.append(r"            \hline")
    lines.append(row(r"$M_{\mathrm{disk}}$ [$10^{10}\,\mathrm{M_\odot}$]", "Mdisk"))
    lines.append(row(r"$M_{200}$ [$10^{12}\,\mathrm{M_\odot}$]", "M200"))
    lines.append(row(r"$R_{200}$ [kpc]", "R200"))
    lines.append(row(rf"$M(<{R_mass:g}\,\mathrm{{kpc}})$ [$10^{{11}}\,\mathrm{{M_\odot}}$]",
                     "Mwithin"))
    lines.append(r"            \hline")
    for key, lbl, _ in LOCAL_PARAMS:
        lines.append(row(lbl, key))
    lines += [
        r"            \hline",
        r"        \end{tabular}",
        r"    \end{table}",
    ]
    tex = "\n".join(lines) + "\n"

    out_path.write_text(tex)
    print(f"\nLaTeX table written to: {out_path}\n")
    print(tex)

    # ── requested extra results, saved in a subfolder (table left untouched) ──
    # 1) rotation curve split per component (halo, disk, bulge + combined)
    # 2) recomputation of M200/R200 from the posterior sample and from the
    #    p16/p50/p84 of the parameters, under two virial-mass definitions
    if posteriors["Combined"] is not None:
        results_dir = data_dir / args.results_subdir
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nWriting requested results to: {results_dir}")
        plot_vcirc(posteriors["Combined"], results_dir / "vcirc_components.pdf")

        # paper virial definition (ellipsoidal halo mass, H0 = 71 km/s/Mpc):
        # M200 = (4pi/3) r200^3 * 200 * rho_crit = 4pi q_h int_0^r200 s^2 rho_h ds.
        # This one also produces the requested R200 / M200 histogram (M200 linear).
        rho_crit_71 = _import_worker()._rho_crit(71e-3)
        recompute_m200_r200(
            posteriors["Combined"], results_dir,
            per_sample=lambda p: m200_r200_ellipsoidal(p, rho_crit_71),
            tag="ellipsoidal_H0-71",
            meta=dict(definition="ellipsoidal halo mass "
                                 "4pi q_h int_0^r200 s^2 rho_h(s) ds, c=200 "
                                 "(paper virial definition)",
                      H0=71e-3, rho_crit=rho_crit_71),
            m200r200_file=args.m200r200, n_samples=args.m200_nsamples,
            make_plot=True)

        # same ellipsoidal definition but with the OLD H0 = 67.4 km/s/Mpc
        # (Planck 2018), the value the table's current M200/R200 were made with.
        rho_crit_674 = _import_worker()._rho_crit(67.4e-3)
        recompute_m200_r200(
            posteriors["Combined"], results_dir,
            per_sample=lambda p: m200_r200_ellipsoidal(p, rho_crit_674),
            tag="ellipsoidal_H0-67.4",
            meta=dict(definition="ellipsoidal halo mass "
                                 "4pi q_h int_0^r200 s^2 rho_h(s) ds, c=200 "
                                 "(old H0 = 67.4 km/s/Mpc, Planck 2018)",
                      H0=67.4e-3, rho_crit=rho_crit_674),
            m200r200_file=args.m200r200, n_samples=args.m200_nsamples,
            make_plot=True)

        # pipeline definition (spherical enclosedMass, H0 = 70 km/s/Mpc), kept
        # for comparison with the value the results table currently reports.
        worker = _import_worker()
        recompute_m200_r200(
            posteriors["Combined"], results_dir,
            per_sample=lambda p: m200_r200_spherical(p, worker.H0_DEFAULT),
            tag="spherical_H0-70",
            meta=dict(definition="spherical halo enclosedMass, c=200 "
                                 "(pipeline agama_vcirc_worker._compute_m200)",
                      H0=worker.H0_DEFAULT,
                      rho_crit=worker._rho_crit(worker.H0_DEFAULT)),
            m200r200_file=args.m200r200, n_samples=args.m200_nsamples,
            make_plot=False)

        # local halo (dark-matter) density at the solar radius
        recompute_halo_solar_density(posteriors["Combined"], results_dir,
                                     R_sun=args.solar_radius,
                                     n_samples=args.m200_nsamples)

        # total enclosed mass within R = mass_radius kpc (per-column detail);
        # reuses the sample arrays already computed for the table row
        write_mass_within(mass_samples, R_mass, results_dir)


if __name__ == "__main__":
    main()
