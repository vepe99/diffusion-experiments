import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import os
    os.environ['CUDA_VISIBILE_DEVICES'] = ""
    import numpy as np
    import matplotlib.pyplot as plt

    if "KERAS_BACKEND" not in os.environ: 
        os.environ["KERAS_BACKEND"] = "jax"

    import bayesflow as bf

    return np, os


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


    base_dir = '/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/plots'
    posterior_sample_dir = ['plots_galax6D_1e6_concat_fulldataset_500_hyper61_gaia_cutNGC3201', "plots_galax6D_1e6_concat_fulldataset_500_hyper40_gaia_cutNGC3201", ]
    posterior_sample_dir = [os.path.join(base_dir, d) for d in posterior_sample_dir]
    posterior_names = ['global_posterior.npz', 'Pal5_posterior.npz', 'NGC3201_posterior.npz', 'M68_posterior.npz']


    ensemble_posteriors = {}

    for post_name in posterior_names:
        print(f"\n--- Processing {post_name} ---")

        # Gather the paths for this specific posterior across all sample directories
        posterior_paths = [os.path.join(d, post_name) for d in posterior_sample_dir]

        # Load the posteriors for this name
        loaded_posteriors = [dict(np.load(p)) for p in posterior_paths]

        combined_posterior = {}
        for k in loaded_posteriors[0].keys():
            # Store combined arrays
            combined_posterior[k] = np.concatenate([p[k] for p in loaded_posteriors], axis=1)

        ensemble_posteriors[post_name] = combined_posterior

        # Print shapes to verify
        for k, v in combined_posterior.items():
            print(f'{k} shape: {v.shape}')
    return (ensemble_posteriors,)


@app.cell
def _(np):
    true_data_path  = '/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz'
    print('Loading test data from ', true_data_path)
    true_data = dict(np.load(true_data_path, allow_pickle=True))
    true_data = {k: true_data[k] for k in ["sim_data_projected", "j", "attention_mask", "magnitudes"] }
    return


@app.cell
def _(ensemble_posteriors):
    for name in ensemble_posteriors.keys():
        for a in ensemble_posteriors[name].keys():
            ensemble_posteriors[name][a] = ensemble_posteriors[name][a].reshape(-1)
    return


@app.cell
def _(ensemble_posteriors):
    ensemble_posteriors.keys()
    return


@app.cell
def _(ensemble_posteriors):
    import pandas as pd

    df_global = pd.DataFrame(ensemble_posteriors['global_posterior.npz'])
    df_Pal5 = pd.DataFrame(ensemble_posteriors['Pal5_posterior.npz'])
    df_NGC3201 = pd.DataFrame(ensemble_posteriors['NGC3201_posterior.npz'])
    df_M68 = pd.DataFrame(ensemble_posteriors['M68_posterior.npz'])
    return df_M68, df_NGC3201, df_Pal5, df_global


@app.cell
def _(df_M68, df_NGC3201, df_Pal5, df_global):
    from chainconsumer import Chain, ChainConsumer, ChainConfig

    c = ChainConsumer()
    c.add_chain(Chain(samples=df_global, name="Global"))
    c.add_chain(Chain(samples=df_Pal5, name="Pal5"))
    c.add_chain(Chain(samples=df_NGC3201, name="NGC3201"))
    c.add_chain(Chain(samples=df_M68, name="M68"))


    c.set_override(ChainConfig(shade=False))
    fig = c.plotter.plot()
    fig.savefig('./global_cornerplot_gaia_ensambled.pdf')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
