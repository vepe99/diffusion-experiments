# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.20.2",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    # from autocvd import autocvd

    # autocvd(num_gpus=1, interval=1)
    import os
    os.environ['CUDA_VISIBILE_DEVICES'] = ""
    import numpy as np
    import matplotlib.pyplot as plt

    if "KERAS_BACKEND" not in os.environ: 
        os.environ["KERAS_BACKEND"] = "jax"

    import bayesflow as bf


    return bf, np, os


@app.cell
def _(np, os):
    param_names_gloabl_pretty = ["$M_{NFW}$", 
    "$r_{NFW}$",
    "$q_{NFW}$",
    "$\\rho_t$",
    "$h_t$",
    "$z_t$",
    "$\\rho_k$",
    "$h_k$",
    "$z_k$"
    ]

    param_name_global = [
    "m_Triaxial_halo",
    "r_Triaxial_halo",
    "q2_Triaxial_halo",
    "rho_thin_disk",
    "hr_thin_disk",
    "hz_thin_disk",
    "rho_thick_disk",
    "hr_thick_disk",
    "hz_thick_disk"
    ]


    N=333
    base_dir = '/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/plots/'
    posterior_sample_dir = [f'plots_galax6D_1e6_concat_fulldataset_500_hyper40_cutNGC3201_nocomposition_{N}tests', 
                            # f'plots_galax6D_1e6_concat_fulldataset_500_hyper61_cutNGC3201_nocomposition_{N}tests',
                            f"plots_galax6D_1e6_concat_fulldataset_500_hyper61_cutNGC3201_nocomposition_{N}tests", ]
    posterior_path = [os.path.join(base_dir, p, 'posterior.npz') for p in posterior_sample_dir]

    posteriors = [dict(np.load(p)) for p in posterior_path]
    ensable_posterior = {}
    for k in posteriors[0].keys():
        ensable_posterior[k] = np.concatenate(tuple([posteriors[i][k] for i in range(len(posteriors))]),axis=1)
    for k in ensable_posterior:
        print(f'{k} shape: {ensable_posterior[k].shape}')

    test_data = dict(np.load(f'/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/streams/data_multistream_galax/simulation_multistream_{N}.npz'))

    test_data_new = {}
    for k in param_name_global:
            test_data_new[k] = np.repeat(test_data[k], 3, axis=0).reshape(-1, 1)
    # for k in param_name_global:
    #     test_data_new[k] = test_data[k]

    print('Test data keys')
    for k in test_data_new:
        print(f'{k} shape: {test_data_new[k].shape}')
    return ensable_posterior, param_names_gloabl_pretty, test_data_new


@app.cell
def _(bf, ensable_posterior, param_names_gloabl_pretty, test_data_new):
    fig = bf.diagnostics.calibration_ecdf(
            estimates=ensable_posterior,
            targets=test_data_new,
            difference=False,
            variable_names=param_names_gloabl_pretty
            # variable_names = param_names_global
        )
    return (fig,)


@app.cell
def _(fig):
    fig.savefig('./calibration_ensamble.pdf')
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
