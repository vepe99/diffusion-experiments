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


def _build_full_potential(p):
    """Bulge + halo + disk potential, identical to agama_vcirc_worker.py."""
    import agama
    return agama.Potential(
        dict(type='Spheroid',
             scaleRadius=75 / 1e3, densityNorm=9.6e10,
             gamma=0, alpha=1, beta=1.8, cutoffStrength=2,
             outerCutoffRadius=2.1, axisRatioY=1.0, axisRatioZ=0.5),          # bulge
        dict(type='Spheroid',
             scaleRadius=p['a_TwoPowerTriaxial_halo'],
             densityNorm=p['rho_TwoPowerTriaxial_halo'],
             gamma=p['gamma_TwoPowerTriaxial_halo'],
             alpha=1, beta=3, cutoffStrength=2, outerCutoffRadius=np.inf,
             axisRatioY=1.0, axisRatioZ=p['q_TwoPowerTriaxial_halo']),        # halo
        dict(type='Disk',
             scaleRadius=p['r_Disk'], scaleHeight=p['z_Disk'],
             surfaceDensity=p['Sigma_Disk'],
             sersicIndex=1, innerCutoffRadius=0),                             # disk
    )


def vcirc_at(p, R):
    """Circular velocity [km/s] at radii R [kpc] for parameter dict p."""
    R = np.asarray(R, dtype=float)
    pot = _build_full_potential(p)
    points = np.column_stack((R, np.zeros_like(R), np.zeros_like(R)))
    v2 = -R * pot.force(points)[:, 0]
    return np.sqrt(v2)


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


def plot_vcirc(combined, out_pdf):
    """Rotation curve (Combined posterior) vs Zhou+2023 and Huang+2016.

    Mirrors the split linear/log two-panel layout of notebooks/paper_plots.py.
    Solid line = median of the parameter posterior; shaded band = 16th--84th
    percentile of the ensemble of rotation curves over the posterior samples.
    """
    import agama
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    agama.setUnits(length=1, velocity=1, mass=1)

    split = 30.0
    R_grid = np.logspace(np.log10(0.1), np.log10(100.0), 400)
    # central line and band are the per-radius p50 / p16 / p84 of the
    # rotation-curve ensemble over all posterior samples (guarantees the
    # median curve sits inside its own 16th--84th band)
    band_lo, v_med, band_hi = vcirc_ensemble(combined, R_grid)

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
        ax.set_ylim(100, 250)

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
    # legend anchored around R = 5 kpc on the linear panel
    ax_lin.legend(handles, labels, loc='lower left',
                  bbox_to_anchor=(5.0, 110), bbox_transform=ax_lin.transData,
                  fontsize=7)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0)
    fig.savefig(out_pdf, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nRotation curve written to: {out_pdf}")


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

    # ── plain-text summary to stdout ─────────────────────────────────────────
    print(f"\nData directory: {data_dir}\n")
    row_labels = [(k, lbl) for k, lbl, _ in PARAMS] + [
        ("Mdisk", r"Mdisk [10^10 Msun]"),
        ("M200", r"M200 [10^12 Msun]"), ("R200", r"R200 [kpc]")] + [
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

    # ── rotation curve from the Combined posterior median ────────────────────
    if posteriors["Combined"] is not None:
        plot_vcirc(posteriors["Combined"], data_dir / "new_vcirc.pdf")


if __name__ == "__main__":
    main()
