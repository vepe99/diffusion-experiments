#%% md
# # PEtab benchmark model
import os
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "tensorflow"
else:
    print(f"Using '{os.environ['KERAS_BACKEND']}' backend")

import numpy as np
import pandas as pd
import pickle
import pypesto.petab

import logging
pypesto.logging.log(level=logging.ERROR, name="pypesto.petab", console=True)

from case_study2.model_settings import MODELS
from helper_visualize import plot_model_comparison_grid

problem_name = "Beer_MolBioSystems2014"
mcmc_path = f'models/mcmc_samples_{problem_name}.pkl'


if __name__ == "__main__":
    if os.path.exists(f'metrics/{problem_name}_mcmc_metrics.csv'):
        with open(f'metrics/{problem_name}_mcmc_metrics.csv', 'rb') as f:
            mcmc_df = pd.read_csv(f, index_col=0)
    else:
        print(f'No MCMC metrics found for {problem_name}')
        mcmc_df = None

    #%%
    for fusion_transformer_summary in [False, True]:
        metrics = []
        for i in range(len(MODELS)):
            model_name = list(MODELS.keys())[i]
            if fusion_transformer_summary and not 'ft' in model_name:
                continue
            elif not fusion_transformer_summary and 'ft' in model_name:
                continue
            if os.path.exists(f'metrics/{problem_name}_metrics_{model_name}.pkl'):
                with open(f'metrics/{problem_name}_metrics_{model_name}.pkl', 'rb') as f:
                    metric = pickle.load(f)
                metrics += metric
            else:
                print(f"Metrics for model {model_name} not found")

        # df, all columns to float beside model and sampler
        metrics_df = pd.DataFrame(metrics, index=None)
        metrics_df.index.name = None
        for col in ['nrmse', 'posterior_contraction', 'posterior_calibration_error', 'c2st']:
            metrics_df[col] = metrics_df[col].astype(float)
            metrics_df.loc[metrics_df[col].isna(), col] = 1
        metrics_df = pd.concat([metrics_df, mcmc_df], ignore_index=True)

        metrics_df.loc[metrics_df['nrmse'] == 1.0, 'posterior_contraction'] = np.nan  # samples were nan
        metrics_df.loc[metrics_df['nrmse'] == 1.0, 'posterior_contraction_mad'] = np.nan  # samples were nan
        metrics_df.loc[metrics_df['nrmse'] == 1.0, 'nrmse'] = np.nan  # samples were nan
        metrics_df.loc[metrics_df['nrmse'] == 1.0, 'nrmse_mad'] = np.nan  # samples were nan
        metrics_df = metrics_df[metrics_df['sampler'] != 'sde-pc']
        metrics_df['c2st'] = 0.5+np.abs(metrics_df['c2st']-0.5)
        metrics_df['rank'] = metrics_df['nrmse'] + metrics_df['posterior_calibration_error']

        if fusion_transformer_summary:
            metrics_df_2 = metrics_df.copy()
        else:
            metrics_df_1 = metrics_df.copy()
            metrics_df_joint = metrics_df.copy()

    #%%
    # order from df1
    order_map = (
        metrics_df_1[["model", "sampler"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .assign(order=lambda x: range(len(x)))
        .rename(columns={"model": "model_key"})
    )

    # add model_key
    df1 = metrics_df_1.copy()
    df1["model_key"] = df1["model"]

    df2 = metrics_df_2.copy()
    df2["model_key"] = (
        df2["model"]
        .str.replace(r"_ft_ema$", "_ema", regex=True)
        .str.replace(r"_ft$", "", regex=True)
    )

    # combine
    merged = pd.concat([df1, df2], ignore_index=True, sort=False)

    # aggregate means ignoring NaN
    agg = (
        merged.groupby(["model_key", "sampler"], dropna=False, as_index=False)
        .mean(numeric_only=True)
    )

    # choose display model name
    display_names = df1[["model_key", "sampler", "model"]].drop_duplicates()
    metrics_df_joint = agg.merge(display_names, on=["model_key", "sampler"], how="left")
    metrics_df_joint["model"] = metrics_df_joint["model"].fillna(metrics_df_joint["model_key"])

    # apply df1 order, unseen pairs go last
    metrics_df_joint = (
        metrics_df_joint.merge(order_map, on=["model_key", "sampler"], how="left")
              .sort_values(["order"], na_position="last", kind="stable")
              .drop(columns=["order", "model_key"])
              .reset_index(drop=True)
    )
    # reorder columns so model and sampler come first
    cols = ["model", "sampler"] + [c for c in metrics_df_joint.columns if c not in ["model", "sampler"]]
    metrics_df_joint = metrics_df_joint[cols]
    metrics_df_joint['rank'] = metrics_df_joint['nrmse'] + metrics_df_joint['posterior_calibration_error']

    metrics_df_joint['family'] = metrics_df_joint['model'].apply(
        lambda x: 'Flow Matching' if 'flow_matching' in x else (
            'Consistency Model' if 'consistency' in x else (
                'Diffusion' if 'diffusion' in x else (
                    'MCMC'
                )
            )
        )
    )

    metrics_df_joint['rank'] = metrics_df_joint['rank'].rank()
    metrics_df_joint.sort_values(['family', 'rank'], inplace=True)
    metrics_df_joint.to_csv("plots/metrics.csv", index=False)
    print(np.corrcoef(metrics_df_joint[['posterior_calibration_error', 'c2st']].values.T))
    plot_model_comparison_grid(metrics_df_joint, save_path='plots', plot_shade=True)
