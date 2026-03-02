import marimo

__generated_with = "0.19.5"
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
        particles[name_plot] = {'Ra': np.array([a for a in tbl_subset['RAdeg']]), 'Dec': np.array([a for a in tbl_subset['DEdeg']]), 'Plx': np.array([a for a in tbl_subset['plx']]), 'PmRA': np.array([a for a in tbl_subset['pmRA']]), 'PmDE': np.array([a for a in tbl_subset['pmDE']]), 'VHel': np.array([a for a in tbl_subset['VHel']]), 'e_VHel': np.array([a for a in tbl_subset['e_VHel']]), 'Gmag': np.array([a for a in tbl_subset['Gmag']])}
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
    return name_to_plot, particles


@app.cell
def _(name_to_plot, np, particles, plt):
    fig = plt.figure(figsize=(15,3))
    for i, name in enumerate(name_to_plot):
        p = particles[name]['VHel']
        mask = (p==0.0)
        ax = fig.add_subplot(2, 3, i+1)
        ax.hist(p[~mask])
        ax.set_xlabel('$V_r$')
        ax.set_title(f'{name}')

    for i, name in enumerate(name_to_plot):
        p = particles[name]['VHel']
        err_p = particles[name]['e_VHel']
        mask = (p==0.0)
        p = p[~mask]
        err_p = err_p[~mask]
        sorted_index = np.argsort(p)
        ax = fig.add_subplot(2, 3, i+4)
        ax.errorbar(range(len(p[sorted_index])), p[sorted_index], err_p[sorted_index])
        ax.set_xlabel('$V_r$')
    
    plt.show()
    
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
