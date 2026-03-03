# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.20.2",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    from autocvd import autocvd

    autocvd(num_gpus=1, interval=1)
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    if "KERAS_BACKEND" not in os.environ: 
        os.environ["KERAS_BACKEND"] = "jax"

    import bayesflow as bf


    return bf, np, os


@app.cell
def _(ditc, np, os):
    base_dir = '/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/plots'
    posterior_sample_dir = ['plots_streamax_concatenation_500_hyper_nocomposition',
                            'plots_streamax_concatenate_500_hyper2nd_nocomposition',
                            'plots_streamax_concatenation_500_highestcalibration_nocomposition', ]
    posterior_path = [os.path.join(base_dir, p, 'posterior.npz') for p in posterior_sample_dir]

    posteriors = [dict(np.load(p)) for p in posterior_path]
    ensable_posterior = {}
    for k in posteriors[0].keys():
        ensable_posterior[k] = np.concatenate(tuple([posteriors[i][k] for i in range(len(posteriors))]),axis=1)
    for k in ensable_posterior:
        print(f'{k} shape: {ensable_posterior[k].shape}')

    test_data = ditc(np.load('/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/data_multistream_streamax/simulation_multistream_1000'))

    for k in test_data:
        print(f'{k} shape: {test_data[k].shape}')
    return ensable_posterior, test_data


@app.cell
def _(bf, cfg, ensable_posterior, test_data):
    # fig = bf.diagnostics.calibration_ecdf(
    #         estimates=ensable_posterior,
    #         targets=test_data,
    #         difference=True,
    #         variable_names=cfg.paramater_global_pretty
    #         # variable_names = param_names_global
    #     )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
