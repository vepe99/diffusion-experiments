import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    os.environ["KERAS_BACKEND"] = "torch"

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    import sbibm

    from model_settings_benchmark import SAMPLER_SETTINGS
    return SAMPLER_SETTINGS, np, pd, plt, sbibm


@app.cell
def _():
    def create_model_config():
        colors = {
            # Flow Matching family
            "flow_matching": "#1B9E77",
            "ot_flow_matching": "#33A02C",
            "flow_matching_edm": "#66C2A5",

            # Consistency family
            "consistency_model": "#D95F02",
            "stable_consistency_model": "#E6AB02",

            # EDM family
            "diffusion_edm_vp": "#E7298A",
            "diffusion_edm_vp_ema": "#AD1457",
            "diffusion_edm_ve": "#FB9A99",

            # Cosine family
            "diffusion_cosine_F": "#7570B3",
            "diffusion_cosine_v": "#9E9AC8",
            "diffusion_cosine_noise": "#54278F",
        }
        return {"visualization": {"colors": colors}}

    def model_name(k):
        names = {
            "flow_matching":r"Flow Matching$\,$",
            "ot_flow_matching":r"Flow Matching (OT)$\,$",
            "flow_matching_edm":r"Flow Matching ($\rho = -0.6$)$\,$",
            "consistency_model":r"Discrete Consistency Model$\,$",
            "stable_consistency_model":r"Continuous Consistency Model$\,$",
            "diffusion_edm_vp":r"VP EDM$\,$",
            "diffusion_edm_ve":r"VE EDM$\,$",
            "diffusion_edm_vp_ema":r"VP EDM EMA$\,$",
            "diffusion_cosine_F":r"Cosine $\boldsymbol{F}$-pred.$\,$",
            "diffusion_cosine_v":r"Cosine $\boldsymbol{v}$-pred.$\,$",
            "diffusion_cosine_noise":r"Cosine $\boldsymbol{\epsilon}$-pred.$\,$",
            "MCMC":"MCMC",
        }
        return names.get(k, k.replace("_"," ").title())
    return create_model_config, model_name


@app.cell
def _(pd):
    results_lueckmann = pd.read_csv('lueckmann_results.csv')
    results_lueckmann = results_lueckmann[results_lueckmann['num_simulations'] == max(results_lueckmann['num_simulations'])]
    results_lueckmann = results_lueckmann[results_lueckmann['algorithm'] == 'NPE']
    results_lueckmann = results_lueckmann.groupby('task')['C2ST'].agg(['mean']).reset_index()

    # Load the dataset
    results = pd.read_csv('c2st_benchmark_results.csv')
    results.head()
    return results, results_lueckmann


@app.cell
def _(SAMPLER_SETTINGS):
    all_samplers = ['best', 'all'] + [k for k in SAMPLER_SETTINGS.keys()]
    SHOW_SAMPLER = all_samplers[0]
    print(SHOW_SAMPLER)
    return (SHOW_SAMPLER,)


@app.cell
def _(SHOW_SAMPLER, create_model_config, np, pd, results, sbibm):
    # Melt the dataframe into long form
    value_vars = [col for col in results.columns if not col.endswith('_std') and col not in ['Model-Sampler', 'sampler', 'model'] and ('time' not in col)]
    std_vars = [col for col in results.columns if col.endswith('_std') and 'time' not in col]
    time_vars = [col for col in results.columns if not col.endswith('_std') and 'time' in col]
    time_std_vars = [col for col in results.columns if col.endswith('_std') and 'time' in col]
    long_df = results.melt(id_vars=['Model-Sampler'], value_vars=value_vars, var_name='problem', value_name='cs2t')
    std_long = results.melt(id_vars=['Model-Sampler'], value_vars=std_vars, var_name='problem_std', value_name='std')
    std_long['problem'] = std_long['problem_std'].str.replace('_std', '', regex=False)
    long_df['std'] = std_long['std']
    # Merge corresponding std values
    time_long = results.melt(id_vars=['Model-Sampler'], value_vars=time_vars, var_name='problem_time', value_name='time')
    time_long['problem'] = time_long['problem_time'].str.replace('_std', '', regex=False)
    long_df['time'] = time_long['time']
    time_long = results.melt(id_vars=['Model-Sampler'], value_vars=time_std_vars, var_name='problem_time', value_name='time_std')
    time_long['problem'] = time_long['problem_time'].str.replace('_std', '', regex=False)
    long_df['time_std'] = time_long['time_std']
    long_df['error'] = np.abs(long_df['cs2t'] - 0.5)
    long_df['rank'] = long_df.groupby('problem')['error'].rank(method='min', ascending=True)
    long_df['model'] = long_df['Model-Sampler'].apply(lambda x: x.split('-')[0])
    long_df['sampler'] = long_df['Model-Sampler'].apply(lambda x: '-'.join(x.split('-')[1:]))
    long_df_copy = long_df.copy()

    # Rank models within each problem (lower error = better rank)
    if SHOW_SAMPLER == 'best':
        mean_ranks = long_df.groupby(['model', 'sampler'])['rank'].mean().reset_index()
        best_samplers = mean_ranks.loc[mean_ranks.groupby('model')['rank'].idxmin(), ['model', 'sampler']]
        long_df = long_df.merge(best_samplers, on=['model', 'sampler'])
    elif SHOW_SAMPLER == 'all':
        long_df_time = (
            long_df[["model", "problem", "time", "time_std"]].groupby(["model", "problem"])
              .sum()
        ).reset_index()
        long_df = (
            long_df[["model", "problem", "error", "std"]].groupby(["model", "problem"])
              .mean()                    # mean across samplers
        ).reset_index()
        long_df['Model-Sampler'] = long_df['model']
        long_df['sampler'] = 'all'
        long_df['time'] = long_df_time['time']
        long_df['time_std'] = long_df_time['time_std']
    else:
        long_df = long_df[long_df['sampler'] == SHOW_SAMPLER]
    long_df['rank'] = long_df.groupby('problem')['error'].rank(method='min', ascending=True)

    config = create_model_config()
    colors = config['visualization']['colors']
    problem_names = np.array([sbibm.get_task(long_df.problem.unique()[_i]).name_display for _i in range(10)])
    problem_dim = [sbibm.get_task(long_df.problem.unique()[_i]).dim_parameters for _i in range(10)]  
    data_dim = [sbibm.get_task(long_df.problem.unique()[_i]).dim_data for _i in range(10)]
    problem_names = np.array([f'{n}\n' + '$\\text{dim}(\\boldsymbol{\\theta})$' + f'$={p_dim}$' + ', $\\text{dim}(\\boldsymbol{y})$' + f'$={d_dim}$' for n, p_dim, d_dim in zip(problem_names, problem_dim, data_dim)])
    problem_order = np.lexsort((data_dim, problem_dim))

    model_order = list(colors.keys())
    long_df["model"] = pd.Categorical(long_df["model"], categories=model_order, ordered=True)
    long_df = long_df.sort_values("model")

    long_df.head()
    return (
        colors,
        long_df,
        long_df_copy,
        model_order,
        problem_names,
        problem_order,
    )


@app.cell
def _(
    SHOW_SAMPLER,
    colors,
    long_df,
    model_name,
    np,
    plt,
    problem_names,
    problem_order,
    results_lueckmann,
):
    # Boxplot showing distribution of model performances per problem
    _fig, _axes = plt.subplots(2, 5, figsize=(12, 4), sharex=True, sharey=True, layout='constrained')
    _axes = _axes.flatten()
    for _idx, _problem_idx in enumerate(problem_order):
        _problem = long_df['problem'].unique()[_problem_idx]
        _subset = long_df[long_df['problem'] == _problem]
        _data_to_plot = []
        _std_to_plot = []
        _labels = []
        _colors_list = []
        for _model in _subset['Model-Sampler'].unique():
            _model_data = _subset[_subset['Model-Sampler'] == _model]
            _base_name = _model_data['model'].values[0]
            _sampler = _model_data['sampler'].values[0]
            error = _model_data['error'].values[0]
            _data_to_plot.append(error)
            _std_to_plot.append(_model_data['std'].values[0])
            _label = model_name(_base_name)
            if not 'consistency' in _model and SHOW_SAMPLER != 'all':
                _label += f' ({_sampler.title()})'
            _labels.append(_label)
            _colors_list.append(colors.get(_base_name, 'gray'))
        _x_positions = np.arange(len(_data_to_plot))
        for _i, _color in enumerate(_colors_list):
            _axes[_idx].errorbar(_x_positions[_i], _data_to_plot[_i], yerr=_std_to_plot[_i],
                                 fmt='o', markersize=6, capsize=3, color=_color, markeredgewidth=0.5, label=_labels[_i])
        _axes[_idx].set_title(problem_names[_problem_idx], fontsize=11)
        _axes[_idx].spines['right'].set_visible(False)
        _axes[_idx].spines['top'].set_visible(False)
        _axes[_idx].grid(True)
        _axes[_idx].set_ylim(0, 0.55)
        _axes[_idx].set_xticks([])
        ref_val = results_lueckmann.loc[results_lueckmann['task'] == _problem, 'mean'].item()
        _axes[_idx].axhline(y=np.abs(0.5 - ref_val), color='black', linestyle='--', linewidth=1,
                            label='Lueckmann et. al. NPE', zorder=-1)
    _axes[0].set_ylabel('$\\vert 0.5-\\text{C2ST}\\vert$', fontsize=11)
    _axes[5].set_ylabel('$\\vert 0.5-\\text{C2ST}\\vert$', fontsize=11)
    _handles = _fig.axes[0].get_legend_handles_labels()[0]
    _fig.legend(labels=_labels + ['Lueckmann et. al. NPE'], handles=_handles[1:] + _handles[:1], loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3, fancybox=False, fontsize=11)
    _fig.savefig(f"plots/c2st_benchmark_boxplot_{SHOW_SAMPLER}.pdf", bbox_inches='tight')
    plt.show()
    return


@app.cell
def _(
    SHOW_SAMPLER,
    colors,
    long_df,
    model_name,
    np,
    plt,
    problem_names,
    problem_order,
):
    # Boxplot showing distribution of model sampling duration per problem
    _fig, _axes = plt.subplots(2, 5, figsize=(12, 4), sharex=True, sharey=True, layout='constrained')
    _axes = _axes.flatten()
    for _idx, _problem_idx in enumerate(problem_order):
        _problem = long_df['problem'].unique()[_problem_idx]
        _subset = long_df[long_df['problem'] == _problem]
        _data_to_plot = []
        _std_to_plot = []
        _labels = []
        _colors_list = []
        for _model in _subset['Model-Sampler'].unique():
            _model_data = _subset[_subset['Model-Sampler'] == _model]
            _base_name = _model_data['model'].values[0]
            _sampler = _model_data['sampler'].values[0]
            timing = _model_data['time'].values[0]
            _data_to_plot.append(timing)
            _std_to_plot.append(_model_data['time_std'].values[0])
            _label = model_name(_base_name)
            if not 'consistency' in _model and SHOW_SAMPLER != 'all':
                _label += f' ({_sampler.title()})'
            _labels.append(_label)
            _colors_list.append(colors.get(_base_name, 'gray'))
        _x_positions = np.arange(len(_data_to_plot))
        for _i, _color in enumerate(_colors_list):
            _axes[_idx].errorbar(_x_positions[_i], _data_to_plot[_i], yerr=_std_to_plot[_i], fmt='o',
                                 markersize=6, capsize=3, color=_color, markeredgewidth=0.5, label=_labels[_i])
        _axes[_idx].set_title(problem_names[_problem_idx], fontsize=11)
        _axes[_idx].spines['right'].set_visible(False)
        _axes[_idx].spines['top'].set_visible(False)
        _axes[_idx].grid(True)
        _axes[_idx].set_xticks([])
        #_axes[_idx].set_ylim([0, 200])
    _axes[0].set_ylabel('Time [s]', fontsize=11)
    _axes[5].set_ylabel('Time [s]', fontsize=11)
    _handles = _fig.axes[0].get_legend_handles_labels()[0]
    _fig.legend(labels=_labels, handles=_handles, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3, fancybox=False, fontsize=11)
    _fig.savefig(f"plots/time_benchmark_boxplot_{SHOW_SAMPLER}.pdf", bbox_inches='tight')
    plt.show()
    return


@app.cell
def _(
    SAMPLER_SETTINGS,
    colors,
    long_df_copy,
    model_name,
    model_order,
    plt,
    problem_names,
):
    # Order samplers: use SAMPLER_SETTINGS order, then any remaining
    sampler_order = [
            s for s in SAMPLER_SETTINGS.keys() if s in long_df_copy["sampler"].unique()
        ]
    sampler_order += [
            s
            for s in sorted(long_df_copy["sampler"].unique())
            if s not in sampler_order
        ]

    problems_unique = long_df_copy["problem"].unique()

    # Dictionary for nicer titles
    problem_to_title = {
        p: t for p, t in zip(problems_unique, problem_names)
    }

    # One plot per problem: x axis = sampler, points colored by model
    for problem in problems_unique:
        subset = long_df_copy[long_df_copy["problem"] == problem]

        # Only keep samplers that appear in this problem
        samplers_here = [
                s for s in sampler_order if s in subset["sampler"].unique()
        ]
        if not samplers_here:
            continue

        models_here = [
            m for m in model_order if m in subset["model"].unique()
        ]
        n_models = max(len(models_here), 1)
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
        for i_s, sampler in enumerate(samplers_here):
            base_x = i_s
            sub_s = subset[subset["sampler"] == sampler]
            if sub_s.empty:
                continue

            for i_m, model in enumerate(models_here):
                row = sub_s[sub_s["model"] == model]
                if row.empty:
                    continue

                x = base_x - 0.4 + width / 2 + i_m * width
                y = row["error"].values[0]
                yerr = row["std"].values[0]

                ax.errorbar(
                        x,
                        y,
                        yerr=yerr,
                        fmt="o",
                        markersize=5,
                        capsize=3,
                        color=colors.get(model, "gray"),
                        markeredgewidth=0.5,
                    )

        ax.set_xticks(range(len(samplers_here)))
        ax.set_xticklabels(
                [s.replace("_", " ").title() for s in samplers_here],
                rotation=45,
                ha="right",
            )
        ax.set_ylabel("$\\vert 0.5-\\text{C2ST}\\vert$")
        ax.set_title(problem_to_title.get(problem, problem), fontsize=11)
        ax.grid(True)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Legend by model, keeping colors
        handles = []
        labels = []
        for model in models_here:
            handle = plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markeredgewidth=0.5,
                    color=colors.get(model, "gray"),
                )
            handles.append(handle)
            labels.append(model_name(model))
        ax.legend(
                handles,
                labels,
                title="Model",
                bbox_to_anchor=(1.02, 1.0),
                loc="upper left",
                borderaxespad=0.0,
                fontsize=9,
            )

        fig.savefig(f"plots/c2st_benchmark_all_samplers_{problem}.pdf", bbox_inches="tight")
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
        for i_s, sampler in enumerate(samplers_here):
            base_x = i_s
            sub_s = subset[subset["sampler"] == sampler]
            if sub_s.empty:
                continue

            for i_m, model in enumerate(models_here):
                row = sub_s[sub_s["model"] == model]
                if row.empty:
                    continue

                x = base_x - 0.4 + width / 2 + i_m * width
                y = row["time"].values[0]
                yerr = row["time_std"].values[0]

                ax.errorbar(
                        x,
                        y,
                        yerr=yerr,
                        fmt="o",
                        markersize=5,
                        capsize=3,
                        color=colors.get(model, "gray"),
                        markeredgewidth=0.5,
                    )

        ax.set_xticks(range(len(samplers_here)))
        ax.set_xticklabels(
                [s.replace("_", " ").title() for s in samplers_here],
                rotation=45,
                ha="right",
            )
        ax.set_ylabel("Time [s]")
        ax.set_title(problem_to_title.get(problem, problem), fontsize=11)
        ax.grid(True)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Legend by model, keeping colors
        handles = []
        labels = []
        for model in models_here:
            handle = plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markeredgewidth=0.5,
                    color=colors.get(model, "gray"),
                )
            handles.append(handle)
            labels.append(model_name(model))
        ax.legend(
                handles,
                labels,
                title="Model",
                bbox_to_anchor=(1.02, 1.0),
                loc="upper left",
                borderaxespad=0.0,
                fontsize=9,
            )

        fig.savefig(f"plots/time_benchmark_all_samplers_{problem}.pdf", bbox_inches="tight")
        plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
