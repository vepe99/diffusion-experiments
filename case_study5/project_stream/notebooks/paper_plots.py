import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt      
    import matplotlib
    import os
    plt.rcParams["savefig.format"] = 'svg'
    return matplotlib, np, os, plt


@app.cell
def _(np):
    path_to_data = '../data/streams/data_multistream_agama_rotationcurve_posterior_predictive_check/rotationacurve/model_5_modelbase250_new_Huang/'
    data_rotcurve = dict(np.load(path_to_data + '/rotation_curve_data.npz', allow_pickle=True))

    print(data_rotcurve.keys())

    radial_distance = data_rotcurve['radial_distance']
    mean_rot_curve = data_rotcurve['mean_rot_curve']
    std_rot_curve = data_rotcurve['std_rot_curve']
    return mean_rot_curve, path_to_data, radial_distance, std_rot_curve


@app.cell
def _(
    matplotlib,
    mean_rot_curve,
    np,
    os,
    path_to_data,
    plt,
    radial_distance,
    std_rot_curve,
):
    obs_R_plot   = np.array([5.24,5.74,6.25,6.77,7.23,7.83,8.21,8.78,9.26,9.75,
                                 10.25,10.75,11.25,11.75,12.24,12.74,13.25,13.74,14.23,14.74,
                                 15.23,15.74,16.24,16.74,17.23,17.74,18.35,18.90,19.50,20.41,
                                 21.28,22.39,23.16,24.00])
    obs_sVc_plot = np.array([0.69,0.68,0.62,0.60,0.45,0.29,0.26,0.22,0.17,0.16,
                             0.17,0.18,0.19,0.20,0.25,0.27,0.27,0.31,0.40,0.43,
                             0.50,0.68,0.74,0.87,1.02,1.15,1.45,1.58,1.32,1.71,
                             1.69,2.01,2.50,4.94])
    obs_Vc_plot  = np.array([225.10,233.53,234.30,233.17,236.19,236.00,233.19,233.15,232.15,231.24,
                             230.34,230.54,229.11,227.48,226.69,225.56,224.90,223.57,221.10,220.19,
                             219.59,217.36,216.61,217.28,216.25,213.81,217.53,212.10,210.46,206.69,
                             207.71,203.72,205.20,200.64])

    # Columns: r (kpc), Vc (km/s), sigma_Vc (km/s)
    #Huang 2016
    HI = np.array([
        [4.60,  231.24, 7.00],
        [5.08,  230.46, 7.00],
        [5.58,  230.01, 7.00],
        [6.10,  239.61, 7.00],
        [6.57,  246.27, 7.00],
        [7.07,  243.49, 7.00],
        [7.58,  242.71, 7.00],
        [8.04,  243.23, 7.00],
    ])

    PRCG = np.array([
        [ 8.34, 239.89,  5.92],
        [ 8.65, 237.26,  6.29],
        [ 9.20, 235.30,  5.60],
        [ 9.62, 230.99,  5.49],
        [10.09, 228.41,  5.62],
        [10.58, 224.26,  5.87],
        [11.09, 224.94,  7.02],
        [11.58, 233.57,  7.65],
        [12.07, 240.02,  6.17],
        [12.73, 242.21,  8.64],
        [13.72, 261.78, 14.89],
        [14.95, 259.26, 30.84],
    ])

    HKG = np.array([
        [15.52, 268.57, 49.67],
        [16.55, 261.17, 50.91],
        [17.56, 240.66, 49.91],
        [18.54, 215.31, 24.80],
        [19.50, 214.99, 24.42],
        [21.25, 251.68, 19.50],
        [23.78, 259.65, 19.62],
        [26.22, 242.02, 18.66],
        [28.71, 224.11, 16.97],
        [31.29, 211.20, 16.43],
        [33.73, 217.93, 17.66],
        [36.19, 219.33, 18.44],
        [38.73, 213.31, 17.29],
        [41.25, 200.05, 17.72],
        [43.93, 190.15, 18.65],
        [46.43, 198.95, 20.70],
        [48.71, 192.91, 19.24],
        [51.56, 198.90, 21.74],
        [57.03, 185.88, 21.56],
        [62.55, 173.89, 22.87],
        [69.47, 196.36, 25.89],
        [79.27, 175.05, 22.71],
        [98.97, 147.72, 23.55],
    ])

    # Unpack for convenience
    r_HI,   Vc_HI,   sVc_HI   = HI.T
    r_PRCG, Vc_PRCG, sVc_PRCG = PRCG.T
    r_HKG,  Vc_HKG,  sVc_HKG  = HKG.T

    #PLOTTING
    split = 30.0
    map_extend     = 35.0  # linear panel extends past split to the right
    map_extend_log = 28.0  # log panel extends past split to the left

    fig, (ax_lin, ax_log) = plt.subplots(
        1, 2,
        sharey=True,
        figsize=(5, 3),
        gridspec_kw={"width_ratios": [3, 2], "wspace": 0}
    )

    # --- MAP on linear panel with extended mask ---
    mask_map_lin = radial_distance <= map_extend
    ax_lin.plot(radial_distance[mask_map_lin], mean_rot_curve[mask_map_lin],
                color='blue', )
    ax_lin.fill_between(
        radial_distance[mask_map_lin],
        mean_rot_curve[mask_map_lin] - 3 * std_rot_curve[mask_map_lin],
        mean_rot_curve[mask_map_lin] + 3 * std_rot_curve[mask_map_lin],
        alpha=0.3, color='blue',
    )

    # --- MAP on log panel with extended mask ---
    mask_map_log = radial_distance >= map_extend_log
    ax_log.plot(radial_distance[mask_map_log], mean_rot_curve[mask_map_log],
                color='blue', )
    ax_log.fill_between(
        radial_distance[mask_map_log],
        mean_rot_curve[mask_map_log] - 3 * std_rot_curve[mask_map_log],
        mean_rot_curve[mask_map_log] + 3 * std_rot_curve[mask_map_log],
        alpha=0.3, color='blue',
    )

    for ax, xmin, xmax, xscale in [
        (ax_lin, 0.1,  split, "linear"),
        (ax_log, split, 99.9,  "log"),
    ]:
        mask_zhou = (obs_R_plot >= xmin) & (obs_R_plot <= xmax)
        ax.errorbar(obs_R_plot[mask_zhou], obs_Vc_plot[mask_zhou], yerr=obs_sVc_plot[mask_zhou],
                    fmt='o', color='crimson', ms=1.5, lw=1, capsize=1, label='Zhou et al. 2023')

        mask_hkg = (r_HKG >= xmin) & (r_HKG <= xmax)
        ax.errorbar(r_HKG[mask_hkg], Vc_HKG[mask_hkg], yerr=sVc_HKG[mask_hkg],
                    fmt='o', color='purple', ms=1.5, lw=1, capsize=1, label='Huang et al. 2016')

        ax.set_xscale(xscale)
        ax.set_xlim(xmin, xmax)  # xlim clips the extensions invisibly
        ax.set_ylim(100, 350)

    
    # --- fix overlapping tick at 30: remove it from the log axis ---
    ax_log.set_xticks([40, 60, 80, 100])
    ax_log.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_log.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    ax_lin.set_xticks([5, 15, 25, 30])
    ax_lin.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # --- remove inner spines ---
    ax_lin.spines["right"].set_visible(False)
    ax_log.spines["left"].set_visible(False)
    ax_log.tick_params(axis='y', left=False)

    # --- single centered xlabel via fig.supxlabel ---
    fig.supxlabel('$R$ [kpc]', fontsize=12, x=+0.55, y=0.1)
    ax_lin.set_ylabel('$V_C$ [km/s]', fontsize=12)

    # --- vertical dashed line at the split ---
    ax_lin.axvline(split, color='k', lw=1.5, ls='--', clip_on=False, zorder=5)

    # --- deduplicated legend ---
    handles, labels = [], []
    for ax in (ax_lin, ax_log):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    ax_log.legend(handles, labels, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.00)

    fig.savefig(os.path.join(path_to_data, 'rotation_curve_sample.pdf'),
                dpi=150, bbox_inches='tight')
    plt.show()
    return


@app.cell
def _(np, os, plt):
    path_to_dir_data_M200 = '../data/hyperparameter_tuning/agama/rotationcurve/model_5/gaiastreams/'
    path_to_data_M200 = os.path.join(path_to_dir_data_M200, 'global_M200_R200.pdf.npz')
    data_M200_R200 = dict(np.load(path_to_data_M200, allow_pickle=True))
    M200_ok = data_M200_R200['M200']
    logM200 = np.log10(M200_ok)
    R200_ok = data_M200_R200['R200']

    fig_m200, axes = plt.subplots(1, 2, figsize=(5, 2.5))
    axes[0].hist(R200_ok, bins=40, color='turquoise', lw=0.4, density=True)
    axes[0].axvline(np.median(R200_ok), color='k',      lw=1.5, )
    axes[0].axvline(np.percentile(R200_ok, 16), color='k', lw=1, ls='--')
    axes[0].axvline(np.percentile(R200_ok, 84), color='k', lw=1, ls='--', )
    axes[0].set_xlabel('$R_{200}$ [kpc]', fontsize=13)
    axes[0].set_ylabel('Normalized counts', fontsize=13)

    axes[1].hist(logM200, bins=40, color='aquamarine',  lw=0.4, density=True)
    axes[1].axvline(np.median(logM200), color='k',      lw=1.5, )
    axes[1].axvline(np.percentile(logM200, 16), color='k', lw=1, ls='--')
    axes[1].axvline(np.percentile(logM200, 84), color='k', lw=1, ls='--', )
    axes[1].set_xlabel('$\log_{10}(M_{200})$ [$M_\odot$]', fontsize=13)

    fig_m200.tight_layout()

    fig_m200.savefig(os.path.join(path_to_dir_data_M200, 'M200_R200_final.pdf'),
                dpi=150, bbox_inches='tight')
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
