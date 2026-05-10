import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from astropy.io import ascii
    import matplotlib
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    import galstreams

    return ascii, galstreams, gaussian_kde, np, pd, plt


@app.cell
def _(ascii, pd):
    tbl_data = ascii.read(
        "../data/apjad382dt1_mrt.txt",
        format="cds"   
    )

    tbl_ids = pd.read_csv("../data/gaia_stream_id.csv", 
                          sep='\t')

    print(tbl_data.colnames)
    print(tbl_ids.columns)

    tbl_ids[(tbl_ids['Name'] == 'Fjorm') | (tbl_ids['Name'] == 'Gjoll') | (tbl_ids['Name'] == 'Pal-5')]
    return tbl_data, tbl_ids


@app.cell
def _(gaussian_kde, np, plt, tbl_data, tbl_ids):
    name_to_plot = ['Pal5', 'NGC3201', 'M68']
    name_for_id = ['Pal-5', 'Gjoll', 'Fjorm']
    particles = {}
    for name_id, name_plot in zip(name_for_id, name_to_plot):
        source_id = tbl_ids.loc[tbl_ids['Name'] == name_id, 's_ID'].values[0]
        tbl_subset = tbl_data[tbl_data['Stream'] == source_id]
        print(f'{name_plot}: {len(tbl_subset)} stars')
        particles[name_plot] = {'Ra': np.array([a for a in tbl_subset['RAdeg']]), 'Dec': np.array([a for a in tbl_subset['DEdeg']]), 'Plx': np.array([a for a in tbl_subset['plx']]), 'PmRA': np.array([a for a in tbl_subset['pmRA']]), 'PmDE': np.array([a for a in tbl_subset['pmDE']]), 'VHel': np.array([a for a in tbl_subset['VHel']]), 'e_VHel': np.array([a for a in tbl_subset['e_VHel']]), 'Gmag': np.array([a for a in tbl_subset['Gmag']]), 'r_VHel': np.array([a for a in tbl_subset['r_VHel']])}
    _fig = plt.figure(figsize=(15, 5))
    for _i, _name in enumerate(name_to_plot):
        _ax = _fig.add_subplot(1, 3, _i + 1)
        _ax.scatter(particles[_name]['Ra'], particles[_name]['Dec'], s=1, label=_name)
        _ax.set_xlabel('RA [deg]')
        if _i == 0:
            _ax.set_ylabel('Dec [deg]')
        _ax.set_title(_name)
    _fig = plt.figure(figsize=(15, 5))
    for _i, _name in enumerate(name_to_plot):
        _ax = _fig.add_subplot(1, 3, _i + 1)
        _ax.scatter(particles[_name]['Ra'], particles[_name]['Plx'], s=1, label=_name)
        _ax.set_xlabel('RA [deg]')
        if _i == 0:
            _ax.set_ylabel('Plx [mas]')
        _ax.set_title(_name)
    _fig = plt.figure(figsize=(15, 5))
    for _i, _name in enumerate(name_to_plot):
        _ax = _fig.add_subplot(1, 3, _i + 1)
        _ax.scatter(particles[_name]['Ra'], particles[_name]['PmRA'], s=1, label=_name)
        _ax.set_xlabel('RA [deg]')
        if _i == 0:
            _ax.set_ylabel('PmRA [mas/yr]')
        _ax.set_title(_name)
    _fig = plt.figure(figsize=(15, 5))
    for _i, _name in enumerate(name_to_plot):
        _ax = _fig.add_subplot(1, 3, _i + 1)
        _ax.scatter(particles[_name]['Ra'], particles[_name]['PmDE'], s=1, label=_name)
        _ax.set_xlabel('RA [deg]')
        if _i == 0:
            _ax.set_ylabel('PmDE [mas/yr]')
        _ax.set_title(_name)
    _fig = plt.figure(figsize=(15, 5))
    for _i, _name in enumerate(name_to_plot):
        _ax = _fig.add_subplot(1, 3, _i + 1)
        _ax.errorbar(particles[_name]['Ra'], particles[_name]['VHel'], yerr=particles[_name]['e_VHel'], fmt='o', markersize=1, label=_name)
        _ax.set_xlabel('RA [deg]')
        if _i == 0:
            _ax.set_ylabel('VHel [km/s]')
        _ax.set_title(_name)
    _fig = plt.figure(figsize=(15, 5))
    for _i, _name in enumerate(name_to_plot):
        _ax = _fig.add_subplot(1, 3, _i + 1)
        _data = particles[_name]['Gmag']
        _ax.hist(_data, bins=30, label=_name, alpha=0.5, density=True)
        kde = gaussian_kde(_data)
        x_grid = np.linspace(np.min(_data), np.max(_data), 1000)
        _ax.plot(x_grid, kde(x_grid), color='k', label='KDE')
        samples = kde.resample((len(_data * 10),)).reshape(len(_data))
        samples = np.clip(samples, np.min(_data), np.max(_data))
        _ax.hist(samples, bins=30, density=True, histtype='step', color='r', lw=2, label='KDE samples')
        _ax.set_xlabel('G [mag]')
        if _i == 0:
            _ax.set_ylabel('Density')
        _ax.set_title(_name)
        _ax.legend()
    plt.tight_layout()  # KDE
    plt.show()
    return name_to_plot, particles


@app.cell
def _(name_to_plot, np, particles):
    surveyid_to_survey = {0: 'no $v_{los}$', 1: 'APOGEE', 2: 'GALAH', 3: 'Gaia RVS', 4: 'LAMOST', 5: 'S5', 6: 'SDSS', 7: 'BOSS', 8: 'ESPaDOnS (this work)', 9: 'AAOmega (from Ibata et al. 2017a [2017ApJ...842..120I])', 10: 'FLAMES (from Ibata et al. 2017a [2017ApJ...842..120I])', 11: 'UVES (from Odenkirchen et al. 2009 [2009AJ....137.3378O])', 12: 'EFOSC (this work)', 13: 'UVES (this work)', 14: 'INT (this work)', 15: 'Yuan et al. 2022a [2022MNRAS.514.1664Y]', 16: 'Li et al. 2021 [2021ApJ...911..149L]', 17: 'Caldwell et al. 2020 [2020AJ....159..287C]', 18: 'Li et al. 2018b [2018ApJ...866...22L]', 19: 'Koposov et al. 2018 [2018MNRAS.479.5343K]', 20: 'Li et al. 2018a [2018ApJ...869..122L]', 21: 'Simon et al. 2020 [2020ApJ...892..137S]', 22: 'Walker et al. 2015 [2015ApJ...808..108W]', 23: 'VIZIER', 24: 'GES', 25: 'DESI', 47: 'not known'}
    for _name in name_to_plot:
        print(f'{_name} survey and counts:')
        r_vhel = particles[_name]['r_VHel']
        survey_id, count = np.unique(r_vhel, return_counts=True)
        print('survey_id: ', [surveyid_to_survey[s] for s in survey_id])
        print('survey count: ', count)
    from collections import defaultdict
    survey_counts = defaultdict(int)
    for _name in name_to_plot:
        r_vhel = particles[_name]['r_VHel']
        survey_ids, counts = np.unique(r_vhel, return_counts=True)
        for sid, c in zip(survey_ids, counts):
            survey_name = surveyid_to_survey.get(int(sid), f'survey_{int(sid)}')
            survey_counts[survey_name] = survey_counts[survey_name] + int(c)
    survey_counts = dict(survey_counts)
    print(survey_counts)
    return (surveyid_to_survey,)


@app.cell
def _(name_to_plot, particles, plt):
    # We want to know from where the Vhel are coming from 
    _fig = plt.figure(figsize=(15, 15))
    for _name in name_to_plot:
        v_los = particles[_name]['VHel']
        mask_vlos = v_los == 0.0
        e_vhel = particles[_name]['e_VHel']
        magnitude = particles[_name]['Gmag']
        _ax = _fig.add_subplot(3, 3, name_to_plot.index(_name) + 1)
        _ax.errorbar(magnitude[~mask_vlos], v_los[~mask_vlos], yerr=e_vhel[~mask_vlos], fmt='.')
        _ax.set_xlabel('G [mag]')
        _ax.set_ylabel('v_los [km/s]')
        _ax.set_title(_name)
    for _name in name_to_plot:
        v_los = particles[_name]['VHel']
        mask_vlos = v_los == 0.0
        e_vhel = particles[_name]['e_VHel']
        magnitude = particles[_name]['Gmag']
        _ax = _fig.add_subplot(3, 3, name_to_plot.index(_name) + 4)
        _ax.scatter(magnitude[~mask_vlos], e_vhel[~mask_vlos])
        _ax.set_xlabel('G [mag]')
        _ax.set_ylabel('e_vhel [km/s]')
        _ax.set_title(_name)
    for _name in name_to_plot:
        v_los = particles[_name]['VHel']
        mask_vlos = v_los == 0.0
        e_vhel = particles[_name]['e_VHel']
        _ax = _fig.add_subplot(3, 3, name_to_plot.index(_name) + 7)
        _ax.hist(e_vhel[~mask_vlos])
        _ax.set_xlabel('e_vhel [km/s]')
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Looking for $err_{v_{los}}(mag)$
    """)
    return


@app.cell
def _(tbl_data_pandas):
    tbl_data_pandas['Gmag'].values.flatten()
    return


@app.cell
def _(tbl_data):
    tbl_data_pandas = tbl_data.to_pandas()

    tbl_data_pandas = tbl_data_pandas[tbl_data_pandas['r_VHel']!=0]
    tbl_data_pandas.head()

    # ...existing code...
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)

    # for r_vhel_id, group in tbl_data_pandas.groupby('r_VHel'):
    #     ax.scatter(group['Gmag'], group['e_VHel'], label=str(r_vhel_id))

    # ax.set_xlabel('Gmag')
    # ax.set_ylabel('e_Vhel')
    # ax.legend(title='r_VHel')
    # ...existing code...
    return (tbl_data_pandas,)


@app.cell
def _(plt, surveyid_to_survey, tbl_data_pandas):
    unique_labels = tbl_data_pandas['r_VHel'].unique()
    n_labels = len(unique_labels)

    fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(18, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, r_vhel_id in zip(axes, unique_labels):
        group = tbl_data_pandas[tbl_data_pandas['r_VHel'] == r_vhel_id]
        ax.scatter(group['Gmag'], group['e_VHel'])
        ax.set_title(f'{surveyid_to_survey[r_vhel_id]}')
        ax.set_xlabel('Gmag')
        ax.set_ylabel('e_VHel')

    # Hide any unused axes if n_labels < 30
    for ax in axes[n_labels:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(np, pd, tbl_data_pandas):


    mag_bins = np.array([9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    tbl_data_pandas['mag_bin'] = pd.cut(tbl_data_pandas['Gmag'], bins=mag_bins, right=False)

    median_e_VHel_per_bin = tbl_data_pandas.groupby('mag_bin')['e_VHel'].median()
    std_e_VHel_per_bin = tbl_data_pandas.groupby('mag_bin')['e_VHel'].std()

    print('Median e_VHel')
    print(median_e_VHel_per_bin)
    print('Std e_VHel')
    print(std_e_VHel_per_bin)
    return median_e_VHel_per_bin, std_e_VHel_per_bin


@app.cell
def _(median_e_VHel_per_bin, np, plt, std_e_VHel_per_bin, tbl_data_pandas):
    # Get bin centers for plotting
    bin_centers = [interval.left + (interval.right - interval.left)/2 for interval in median_e_VHel_per_bin.index]

    fig1, ax1 = plt.subplots(figsize=(8, 6))

    # Scatter plot of all data
    ax1.scatter(tbl_data_pandas['Gmag'], tbl_data_pandas['e_VHel'], alpha=0.3, label='stars')

    # Plot median line
    ax1.plot(bin_centers, median_e_VHel_per_bin.values, color='red', marker='o', label='median $err_{V_R}$')

    # Plot filled error bars (standard deviation)
    lower = np.maximum(median_e_VHel_per_bin.values - std_e_VHel_per_bin.values, 0)
    upper = median_e_VHel_per_bin.values + std_e_VHel_per_bin.values

    ax1.fill_between(
        bin_centers,
        lower,
        upper,
        color='red',
        alpha=0.2,
        # label='$\\sigma$'
    )

    ax1.set_xlabel('Gmag', fontsize=20)
    ax1.set_ylabel('$err_{V_R}$', fontsize=20)
    # ax1.set_yscale('log')
    ax1.legend(fontsize=25)
    ax1.set_ylim(-1, 60)
    ax1.tick_params(axis="both", labelsize=15)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Galstreams
    """)
    return


@app.cell
def _(galstreams):
    mws = galstreams.MWStreams(verbose=False, implement_Off=True, print_topcat_friendly_files=False);
    return (mws,)


@app.cell
def _(mws, name_to_plot, particles, plt):
    _track_name_Pal5 = mws.get_track_names_for_stream('Pal5', On_only=True)
    _track_name_NGC3201 = mws.get_track_names_for_stream('NGC3201', On_only=True)
    _track_name_M68 = mws.get_track_names_for_stream('M68', On_only=True)
    print('Track name for Pal 5:', _track_name_Pal5)
    print('Track name for NGC 3201:', _track_name_NGC3201)
    print('Track name for M68:', _track_name_M68)
    _fig = plt.figure(figsize=(15, 5), tight_layout=True)
    _ax = _fig.add_subplot(131)
    _ax.set_xlabel('RA (deg)')
    _ax.set_ylabel('Dec (deg)')
    _ax2 = _fig.add_subplot(132)
    _ax2.set_xlabel('RA (deg)')
    _ax2.set_ylabel('Dec (deg)')
    _ax3 = _fig.add_subplot(133)
    _ax3.set_xlabel('RA (deg)')
    _ax3.set_ylabel('Dec (deg)')
    for _name in _track_name_Pal5:
        _track = mws[_name].track
        _ax.plot(_track.ra, _track.dec, label=f'{_name} Stream Track')
    for _name in _track_name_NGC3201:
        _track = mws[_name].track
        _ax2.plot(_track.ra, _track.dec, label=f'{_name} Stream Track')
    for _name in _track_name_M68:
        _track = mws[_name].track
        _ax3.plot(_track.ra, _track.dec, label=f'{_name} Stream Track')
    for _i, _name in enumerate(name_to_plot):
        if _i == 0:
    # track_ngc3201 = mws[track_name_NGC3201[3]].track
    # ax2.plot(track_ngc3201.ra, track_ngc3201.dec, label='NGC 3201 Stream Track', color='orange')
            _ax.scatter(particles[_name]['Ra'], particles[_name]['Dec'], s=1, label=_name)
            _ax.set_title(_name)
        elif _i == 1:
            _ax2.scatter(particles[_name]['Ra'], particles[_name]['Dec'], s=1, label=_name)
    # track_m68 = mws[track_name_M68[0]].track
    # ax3.plot(track_m68.ra, track_m68.dec, label='M68 Stream Track', color='green')
            _ax2.set_title(_name)
        else:
            _ax3.scatter(particles[_name]['Ra'], particles[_name]['Dec'], s=1, label=_name)
            _ax3.set_title(_name)
    _ax.legend()
    _ax2.legend()
    _ax3.legend()
    return


@app.cell
def _(mws, name_to_plot, particles, plt):
    _track_name_Pal5 = mws.get_track_names_for_stream('Pal5', On_only=True)
    _track_name_NGC3201 = mws.get_track_names_for_stream('NGC3201', On_only=True)
    _track_name_M68 = mws.get_track_names_for_stream('M68', On_only=True)
    print('Track name for Pal 5:', _track_name_Pal5)
    print('Track name for NGC 3201:', _track_name_NGC3201)
    print('Track name for M68:', _track_name_M68)
    _fig = plt.figure(figsize=(15, 5), tight_layout=True)
    _ax = _fig.add_subplot(131)
    _ax.set_xlabel('RA (deg)')
    _ax.set_ylabel('Plx')
    _ax2 = _fig.add_subplot(132)
    _ax2.set_xlabel('RA (deg)')
    _ax2.set_ylabel('Plx')
    _ax3 = _fig.add_subplot(133)
    _ax3.set_xlabel('RA (deg)')
    _ax3.set_ylabel('Plx')
    for _name in _track_name_Pal5:
        _track = mws[_name].track
        _ax.plot(_track.ra, 1 / _track.distance, label=f'{_name} Stream Track')
    for _name in _track_name_NGC3201:
        _track = mws[_name].track
        _ax2.plot(_track.ra, 1 / _track.distance, label=f'{_name} Stream Track')
    for _name in _track_name_M68:
        _track = mws[_name].track
        _ax3.plot(_track.ra, 1 / _track.distance, label=f'{_name} Stream Track')
    for _i, _name in enumerate(name_to_plot):
        if _i == 0:
    # track_ngc3201 = mws[track_name_NGC3201[3]].track
    # ax2.plot(track_ngc3201.ra, track_ngc3201.dec, label='NGC 3201 Stream Track', color='orange')
            _ax.scatter(particles[_name]['Ra'], particles[_name]['Plx'], s=1, label=_name)
            _ax.set_title(_name)
        elif _i == 1:
            _ax2.scatter(particles[_name]['Ra'], particles[_name]['Plx'], s=1, label=_name)
    # track_m68 = mws[track_name_M68[0]].track
    # ax3.plot(track_m68.ra, track_m68.dec, label='M68 Stream Track', color='green')
            _ax2.set_title(_name)
        else:
            _ax3.scatter(particles[_name]['Ra'], particles[_name]['Plx'], s=1, label=_name)
            _ax3.set_title(_name)
    _ax.legend()
    _ax2.legend()
    _ax3.legend()
    return


@app.cell
def _(mws, name_to_plot, particles, plt):
    _track_name_Pal5 = mws.get_track_names_for_stream('Pal5', On_only=True)
    _track_name_NGC3201 = mws.get_track_names_for_stream('NGC3201', On_only=True)
    _track_name_M68 = mws.get_track_names_for_stream('M68', On_only=True)
    print('Track name for Pal 5:', _track_name_Pal5)
    print('Track name for NGC 3201:', _track_name_NGC3201)
    print('Track name for M68:', _track_name_M68)
    _fig = plt.figure(figsize=(15, 5), tight_layout=True)
    _ax = _fig.add_subplot(131)
    _ax.set_xlabel('RA (deg)')
    _ax.set_ylabel('PmRA')
    _ax2 = _fig.add_subplot(132)
    _ax2.set_xlabel('RA (deg)')
    _ax2.set_ylabel('PmRA')
    _ax3 = _fig.add_subplot(133)
    _ax3.set_xlabel('RA (deg)')
    _ax3.set_ylabel('PmRA')
    for _name in _track_name_Pal5:
        _track = mws[_name].track
        _ax.plot(_track.ra, _track.pm_ra_cosdec, label=f'{_name} Stream Track')
    for _name in _track_name_NGC3201:
        _track = mws[_name].track
        _ax2.plot(_track.ra, _track.pm_ra_cosdec, label=f'{_name} Stream Track')
    for _name in _track_name_M68:
        _track = mws[_name].track
        _ax3.plot(_track.ra, _track.pm_ra_cosdec, label=f'{_name} Stream Track')
    for _i, _name in enumerate(name_to_plot):
        if _i == 0:
    # track_ngc3201 = mws[track_name_NGC3201[3]].track
    # ax2.plot(track_ngc3201.ra, track_ngc3201.dec, label='NGC 3201 Stream Track', color='orange')
            _ax.scatter(particles[_name]['Ra'], particles[_name]['PmRA'], s=1, label=_name)
            _ax.set_title(_name)
        elif _i == 1:
            _ax2.scatter(particles[_name]['Ra'], particles[_name]['PmRA'], s=1, label=_name)
    # track_m68 = mws[track_name_M68[0]].track
    # ax3.plot(track_m68.ra, track_m68.dec, label='M68 Stream Track', color='green')
            _ax2.set_title(_name)
        else:
            _ax3.scatter(particles[_name]['Ra'], particles[_name]['PmRA'], s=1, label=_name)
            _ax3.set_title(_name)
    _ax.legend()
    _ax2.legend()
    _ax3.legend()
    return


@app.cell
def _(mws, name_to_plot, particles, plt):
    _track_name_Pal5 = mws.get_track_names_for_stream('Pal5', On_only=True)
    _track_name_NGC3201 = mws.get_track_names_for_stream('NGC3201', On_only=True)
    _track_name_M68 = mws.get_track_names_for_stream('M68', On_only=True)
    print('Track name for Pal 5:', _track_name_Pal5)
    print('Track name for NGC 3201:', _track_name_NGC3201)
    print('Track name for M68:', _track_name_M68)
    _fig = plt.figure(figsize=(15, 5), tight_layout=True)
    _ax = _fig.add_subplot(131)
    _ax.set_xlabel('RA (deg)')
    _ax.set_ylabel('PmDE')
    _ax2 = _fig.add_subplot(132)
    _ax2.set_xlabel('RA (deg)')
    _ax2.set_ylabel('PmDE')
    _ax3 = _fig.add_subplot(133)
    _ax3.set_xlabel('RA (deg)')
    _ax3.set_ylabel('PmDE')
    for _name in _track_name_Pal5:
        _track = mws[_name].track
        _ax.plot(_track.ra, _track.pm_dec, label=f'{_name} Stream Track')
    for _name in _track_name_NGC3201:
        _track = mws[_name].track
        _ax2.plot(_track.ra, _track.pm_dec, label=f'{_name} Stream Track')
    for _name in _track_name_M68:
        _track = mws[_name].track
        _ax3.plot(_track.ra, _track.pm_dec, label=f'{_name} Stream Track')
    for _i, _name in enumerate(name_to_plot):
        if _i == 0:
    # track_ngc3201 = mws[track_name_NGC3201[3]].track
    # ax2.plot(track_ngc3201.ra, track_ngc3201.dec, label='NGC 3201 Stream Track', color='orange')
            _ax.scatter(particles[_name]['Ra'], particles[_name]['PmDE'], s=1, label=_name)
            _ax.set_title(_name)
        elif _i == 1:
            _ax2.scatter(particles[_name]['Ra'], particles[_name]['PmDE'], s=1, label=_name)
    # track_m68 = mws[track_name_M68[0]].track
    # ax3.plot(track_m68.ra, track_m68.dec, label='M68 Stream Track', color='green')
            _ax2.set_title(_name)
        else:
            _ax3.scatter(particles[_name]['Ra'], particles[_name]['PmDE'], s=1, label=_name)
            _ax3.set_title(_name)
    _ax.legend()
    _ax2.legend()
    _ax3.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Extract Gaia stream data
    """)
    return


@app.cell
def _(np, particles):
    name_to_plot_1 = ['Pal5', 'NGC3201', 'M68']
    N_max = 1000
    n_features = 6
    sim_data_projected = np.zeros((1, len(name_to_plot_1), N_max, n_features))
    _attention_mask = np.zeros((len(name_to_plot_1), N_max))
    for _i, _name in enumerate(name_to_plot_1):
        p = particles[_name]
        n_stars = len(p['Ra'])
        print(f'{_name}: {n_stars} stars (padding to {N_max})')
        _data = np.column_stack([p['Ra'], p['Dec'], p['Plx'], p['PmRA'], p['PmDE'], p['VHel']])
        n_fill = min(n_stars, N_max)
        sim_data_projected[0, _i, :n_fill, :] = _data[:n_fill]
        _attention_mask[_i, :n_fill] = 1.0
    _attention_mask = _attention_mask[:, np.newaxis, :]
    _j = np.array([0, 1, 2]).reshape(1, len(name_to_plot_1), 1)
    print(f'sim_data_projected shape: {sim_data_projected.shape}')
    print(f'attention_mask shape: {_attention_mask.shape}')
    print(f'j: {_j}')
    np.savez('../data/gaia_observed_streams.npz', j=_j, sim_data_projected=sim_data_projected, attention_mask=_attention_mask)
    print('Saved to ../data/gaia_observed_streams.npz')
    return


@app.cell
def _(np, plt):
    _data = np.load('../data/gaia_observed_streams.npz')
    _sim_data = _data['sim_data_projected']
    _j = _data['j']
    name_to_plot_2 = ['Pal5', 'NGC3201', 'M68']
    print(f'Loaded sim_data shape: {_sim_data.shape}')
    print(f'Loaded j shape: {_j.shape}')
    _attention_mask = _data['attention_mask'].astype(bool)
    for _i in range(_sim_data.shape[1]):
        plt.scatter(_sim_data[0, _i, _attention_mask[_i, 0, :], 0], _sim_data[0, _i, _attention_mask[_i, 0, :], 1], s=1, label=f'Stream {_j[0, _i, 0]}')
    plt.legend()
    plt.xlabel('RA [deg]')
    plt.ylabel('Dec [deg]')
    return


@app.cell
def _(np, plt):
    _data = np.load('../data/gaia_observed_streams.npz')
    _sim_data = _data['sim_data_projected']
    _j = _data['j']
    name_to_plot_3 = ['Pal5', 'NGC3201', 'M68']
    print(f'Loaded sim_data shape: {_sim_data.shape}')
    print(f'Loaded j shape: {_j.shape}')
    _attention_mask = _data['attention_mask'].astype(bool)
    _fig = plt.figure(figsize=(7, 5))
    for _i in range(_sim_data.shape[1]):
        plt.scatter(_sim_data[0, _i, _attention_mask[_i, 0, :], 0], _sim_data[0, _i, _attention_mask[_i, 0, :], 1], s=1, label=f'{name_to_plot_3[_i]}')
    plt.legend()
    plt.xlabel('$\\alpha$ ', fontsize=20)
    plt.ylabel('$\\delta$ ', fontsize=20)
    return


if __name__ == "__main__":
    app.run()
