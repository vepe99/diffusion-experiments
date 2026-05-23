import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import corner


    return corner, np, plt


@app.cell
def _(np):

    # ── Load data ────────────────────────────────────────────────────────────────
    data = dict(np.load('../data/streams/data_agama_new/training_data_local_300000.npz'))
    data_rotcurve = dict(np.load('../data/plots/agama_rotcurv_new/rotation_curves.npz'))
    data['vcirc_kms'] = data_rotcurve['vcirc_kms'][:, 5]
    return (data,)


@app.cell
def _(corner, data, np, plt):


    # ── Mask ─────────────────────────────────────────────────────────────────────
    mask = (data['vcirc_kms'] < 240.0) & (data['vcirc_kms'] > 210.0)

    # ── Configure keys here ───────────────────────────────────────────────────────
    KEYS = [
        'a_TwoPowerTriaxial_halo',
        'rho_TwoPowerTriaxial_halo',
        'q_TwoPowerTriaxial_halo',
        'gamma_TwoPowerTriaxial_halo',
        'r_Disk',
        'z_Disk',
        'Sigma_Disk'
    ]

    # Optional: nicer axis labels (set to None to just use key names)
    LABELS = {
        'a_TwoPowerTriaxial_halo':   r'$a$',
        'rho_TwoPowerTriaxial_halo': r'$\rho_0$',
        'q_TwoPowerTriaxial_halo':   r'$q$',
        'gamma_TwoPowerTriaxial_halo': r'$\gamma$',
        'r_Disk': 'r_Disk',
        'z_Disk': 'z_Disk',
        'Sigma_Disk': 'Sigma_Disk'
    }

    # ── Build sample array ────────────────────────────────────────────────────────
    samples = np.column_stack([
        data[k].flatten()[mask] for k in KEYS
    ])

    labels = [LABELS.get(k, k) for k in KEYS]

    # ── Corner plot ───────────────────────────────────────────────────────────────
    fig = corner.corner(
        samples,
        labels=labels,
        show_titles=True,
        title_kwargs={"fontsize": 11},
        label_kwargs={"fontsize": 12},
        plot_density=True,
        plot_contours=True,
        fill_contours=True,
        bins=40,
        smooth=1.0,
        quantiles=[0.16, 0.5, 0.84],
        title_fmt='.3g',
    )

    fig.suptitle(
        rf'$v_{{circ}} \in [210, 250]$ km/s  —  $N={mask.sum():,}$ samples',
        y=1.01, fontsize=13
    )
    plt.tight_layout()
    plt.savefig('cornerplot_halo_params.pdf', bbox_inches='tight', dpi=150)
    plt.show()
    return


@app.cell
def _():
    # fig = plt.figure(figsize=(15, 5))
    # ax = fig.add_subplot(131)
    # ax.scatter( data['a_TwoPowerTriaxial_halo'].flatten()[mask],
    #             data['rho_TwoPowerTriaxial_halo'].flatten()[mask],
    #            c=data['vcirc_kms'][mask],
    #            rasterized=True,
    #            s=1)
    # ax = fig.add_subplot(132)
    # ax.scatter(data['q_TwoPowerTriaxial_halo'].flatten()[mask],
    #             data['rho_TwoPowerTriaxial_halo'].flatten()[mask], 
    #            c=data['vcirc_kms'][mask],
    #            rasterized=True,
    #            s=1)
    # ax = fig.add_subplot(133)
    # ax.scatter(data['gamma_TwoPowerTriaxial_halo'].flatten()[mask],
    #             data['rho_TwoPowerTriaxial_halo'].flatten()[mask], 
    #            c=data['vcirc_kms'][mask],
    #            rasterized=True,
    #            s=1)
    # plt.show()
    return


if __name__ == "__main__":
    app.run()
