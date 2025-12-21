import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    os.environ["KERAS_BACKEND"] = "tensorflow"

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    import sbibm
    import sys

    BASE = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE.parent 
    sys.path.append(str(PROJECT_ROOT))

    from case_study1.model_settings_benchmark import SAMPLER_SETTINGS
    return BASE, SAMPLER_SETTINGS, np, pd, plt, sbibm


@app.cell
def _():
    def create_model_config():
        colors = {
            # Flow Matching family
            "flow_matching": "#1B9E77",
            "ot_flow_matching": "#2E7D32",           # dark medium green
            "ot_flow_matching_offline": "#388E3C",   # medium green
            "ot_partial_flow_matching": "#43A047",   # balanced green
            "cot_flow_matching": "#4CAF50",          # standard green
            "cot_partial_flow_matching": "#66BB6A",  # light green
            "flow_matching_edm": "#81C784",          # pale green

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
            "ot_flow_matching_offline":r"Flow Matching (OT, offline)$\,$",
            "ot_partial_flow_matching":r"Flow Matching (POT)$\,$",
            "cot_flow_matching":r"Flow Matching (COT)$\,$",
            "cot_partial_flow_matching":r"Flow Matching (CPOT)$\,$",
            "flow_matching_edm":r"Flow Matching ($\rho = -0.6$)$\,$",
            "consistency_model":r"Discrete Consistency Model$\,$",
            "stable_consistency_model":r"Continuous Consistency Model$\,$",
            "diffusion_edm_vp":r"VP EDM$\,$",
            "diffusion_edm_ve":r"VE EDM$\,$",
            "diffusion_edm_vp_ema":r"VP EDM EMA$\,$",
            "diffusion_cosine_F":r"Cosine $\mathbf{F}$-pred.$\,$",
            "diffusion_cosine_v":r"Cosine $\boldsymbol{v}$-pred.$\,$",
            "diffusion_cosine_noise":r"Cosine $\boldsymbol{\epsilon}$-pred.$\,$",
            "MCMC":"MCMC",
        }
        return names.get(k, k.replace("_"," ").title())

    def sampler_name(k):
        names = {
           'ode-rk45': 'ODE-RK45',
           'ode-rk45-adaptive': 'ODE-RK45-Adaptive', 
           'ode-tsit5': 'ODE-TSIT5',
           'ode-tsit5-adaptive': 'ODE-TSIT5-Adaptive',
           'ode-euler': 'ODE-Euler', 
           'sde-euler-adaptive': 'SDE-EulerM-Adaptive', 
           'sde-euler': 'SDE-EulerM' ,
           'sde-euler-pc': 'SDE-EulerM-PC',
           'sde-langevin': 'SDE-Langevin', 
           'sde-langevin-pc': 'SDE-Langevin-PC',
           'sde-sea': 'SDE-SEA',
           'sde-shark': 'SDE-ShARK',
           'sde-two_step-adaptive': 'SDE-2Step-Adaptive'
        }
        return names.get(k, k.replace("_"," ").title())
    return create_model_config, model_name, sampler_name


@app.cell
def _(BASE, pd):
    results_lueckmann = pd.read_csv(BASE / 'plots' / 'lueckmann_results.csv')
    results_lueckmann = results_lueckmann.loc[results_lueckmann['num_simulations'] == max(results_lueckmann['num_simulations']),
        ['task', 'algorithm', 'C2ST']]
    results_lueckmann = results_lueckmann[results_lueckmann['algorithm'] == 'NPE']
    results_lueckmann = results_lueckmann.groupby('task')['C2ST'].mean().reset_index()
    #results_lueckmann = results_lueckmann.groupby(['task', 'algorithm']).mean().groupby(['task']).min().reset_index()

    # Load the dataset
    results = pd.read_csv(BASE / 'plots' / 'c2st_benchmark_results.csv')
    results.head()
    return results, results_lueckmann


@app.cell
def _(SAMPLER_SETTINGS):
    all_samplers = ['best', 'merge_problems'] + [k for k in SAMPLER_SETTINGS.keys()]
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
    long_df['error'] = long_df['cs2t'] #np.abs(long_df['cs2t'] - 0.5)
    long_df['model'] = long_df['Model-Sampler'].apply(lambda x: x.split('-')[0])
    long_df['sampler'] = long_df['Model-Sampler'].apply(lambda x: '-'.join(x.split('-')[1:]))
    long_df_copy = long_df.copy()

    # Rank models within each problem (lower error = better rank)
    if SHOW_SAMPLER == 'best':
        mean_error = long_df.groupby(['model', 'sampler'])['error'].mean().reset_index()
        best_samplers = mean_error.loc[mean_error.groupby('model')['error'].idxmin(), ['model', 'sampler']]
        long_df = long_df.merge(best_samplers, on=['model', 'sampler'])
    elif SHOW_SAMPLER == 'merge_problems':
        long_df = (
            long_df[["model", "sampler", "error", "std", "time", "time_std"]].groupby(["model", "sampler"])
              .sum()                    # mean across problems
        ).reset_index()
        long_df['problem'] = 'all'
        long_df['time'] = long_df['time'] / 10
        long_df['error'] = long_df['error'] / 10
    else:
        long_df = long_df[long_df['sampler'] == SHOW_SAMPLER]

    config = create_model_config()
    colors = config['visualization']['colors']
    if long_df.problem[0] != 'all':
        problem_names = np.array([sbibm.get_task(long_df.problem.unique()[_i]).name for _i in range(10)])
        problem_names_nice = np.array([sbibm.get_task(long_df.problem.unique()[_i]).name_display for _i in range(10)])
        problem_dim = [sbibm.get_task(long_df.problem.unique()[_i]).dim_parameters for _i in range(10)]  
        data_dim = [sbibm.get_task(long_df.problem.unique()[_i]).dim_data for _i in range(10)]
        problem_names_nice = np.array([f'{n}\n' + '$\\mathrm{dim}(\\boldsymbol{\\theta})$' + f'$={p_dim}$' + ', $\\mathrm{dim}(\\mathbf{y})$' + f'$={d_dim}$' for n, p_dim, d_dim in zip(problem_names_nice, problem_dim, data_dim)])
        problem_order = np.lexsort((data_dim, problem_dim))

    model_order = list(colors.keys())
    long_df["model"] = pd.Categorical(long_df["model"], categories=model_order, ordered=True)
    long_df = long_df.sort_values("model")

    # just for which OT works better
    #long_df = long_df[long_df["model"].str.contains('flow_matching')].reset_index()
    if SHOW_SAMPLER == 'best':
        long_df = long_df[(long_df["model"].str.contains('diffusion')) | (long_df["model"].str.contains('consistency')) | (long_df["model"] == 'flow_matching') | (long_df["model"] == 'cot_flow_matching')| (long_df["model"] == 'flow_matching_edm')].reset_index()

    #long_df = long_df.dropna()

    long_df.head()
    return (
        colors,
        long_df,
        long_df_copy,
        model_order,
        problem_names,
        problem_names_nice,
        problem_order,
    )


@app.cell
def _(
    BASE,
    SHOW_SAMPLER,
    colors,
    long_df,
    model_name,
    np,
    plt,
    problem_names,
    problem_names_nice,
    problem_order,
    results_lueckmann,
    sampler_name,
):
    if long_df.problem[0] != 'all':
        # Boxplot showing distribution of model performances per problem
        _fig, _axes = plt.subplots(2, 5, figsize=(12, 4), sharex=True, sharey=True, layout='constrained')
        _axes = _axes.flatten()
        for _idx, _problem_idx in enumerate(problem_order):
            _subset = long_df[long_df['problem'] == problem_names[_problem_idx]]
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
                    _label += f' ({sampler_name(_sampler)})'
                _labels.append(_label)
                _colors_list.append(colors.get(_base_name, 'gray'))
            _x_positions = np.arange(len(_data_to_plot))
            for _i, _color in enumerate(_colors_list):
                _axes[_idx].errorbar(_x_positions[_i], _data_to_plot[_i], yerr=_std_to_plot[_i],
                                     fmt='o', markersize=6, capsize=3, color=_color, markeredgewidth=0.5, label=_labels[_i])
            _axes[_idx].set_title(problem_names_nice[_problem_idx], fontsize=12)
            _axes[_idx].spines['right'].set_visible(False)
            _axes[_idx].spines['top'].set_visible(False)
            _axes[_idx].grid(True)
            _axes[_idx].set_ylim(0.48, 1)
            _axes[_idx].set_xticks([])
            #ref_val = results_lueckmann.loc[results_lueckmann['task'] == problem_names[_problem_idx], 'mean'].item()
            ref_val = results_lueckmann.loc[results_lueckmann['task'] == problem_names[_problem_idx], 'C2ST'].item()
            _axes[_idx].axhline(y=ref_val, color='black', linestyle='--', linewidth=1,
                                label='Lueckmann et. al. NPE', zorder=-1)
        _axes[0].set_ylabel(r'$\mathrm{C2ST}$', fontsize=12)
        _axes[5].set_ylabel(r'$\mathrm{C2ST}$', fontsize=12)
        _handles = _fig.axes[0].get_legend_handles_labels()[0]
        _fig.legend(labels=_labels + ['Lueckmann et al. NPE'], handles=_handles[1:] + _handles[:1], loc='lower center',
                    bbox_to_anchor=(0.5, -0.32), ncol=3, fancybox=False, fontsize=12)
        _fig.savefig(BASE / 'plots' / f"c2st_benchmark_boxplot_{SHOW_SAMPLER}.pdf", bbox_inches='tight')
        plt.show()
    return


@app.cell
def _(BASE, SHOW_SAMPLER, colors, long_df, model_name, np, plt, sampler_name):
    if long_df.problem[0] == 'all':
        # Boxplot showing distribution of model performances per sampler
        _fig, _axes = plt.subplots(2, len(long_df.sampler.unique())//2+1, figsize=(12, 4), sharex=True, sharey=True, layout='constrained')
        _axes = _axes.flatten()
        for _idx, _sampler in enumerate(long_df.sampler.unique()):
            _subset = long_df[long_df['sampler'] == _sampler]
            _data_to_plot = []
            _std_to_plot = []
            if _idx == 0:
                _labels = []
            _colors_list = []

            for _model in _subset['model'].unique():
                _model_data = _subset[_subset['model'] == _model]
                _base_name = _model_data['model'].values[0]
                _sampler = _model_data['sampler'].values[0]
                _val = _model_data['time'].values[0]
                _data_to_plot.append(_val)
                _std_to_plot.append(_model_data['time_std'].values[0])
                _label = model_name(_base_name)
                if _idx == 0:
                    _labels.append(_label)
                _colors_list.append(colors.get(_base_name, 'gray'))
            _x_positions = np.arange(len(_data_to_plot))
            if len(_x_positions) != len(long_df.model.unique()):
                _x_positions += 5
            for _i, _color in enumerate(_colors_list):
                if _sampler != 'ode-euler' and 'Consistency' in _labels[_i]:
                    continue  # consistency models
                _yerr = np.array([np.minimum(_data_to_plot[_i], _std_to_plot[_i]), _std_to_plot[_i]])[:, None]
                _axes[_idx].errorbar(_x_positions[_i], _data_to_plot[_i], yerr=_yerr, #_std_to_plot[_i],
                                     fmt='o', markersize=6, capsize=3, color=_color, markeredgewidth=0.5, label=_labels[_i])
            _axes[_idx].set_title(sampler_name(_sampler), fontsize=11)
            _axes[_idx].spines['right'].set_visible(False)
            _axes[_idx].spines['top'].set_visible(False)
            _axes[_idx].grid(True)
            _axes[_idx].set_xticks([])
        _axes[0].set_ylabel(r'Time [s]', fontsize=11)
        _axes[len(long_df.sampler.unique())//2+1].set_ylabel(r'Time [s]', fontsize=11)
        _axes[-1].set_visible(False)
        #_axes[0].set_yscale('log')
        _axes[0].set_ylim(0.1, 150)
        _handles = _fig.axes[4].get_legend_handles_labels()[0]
        _fig.legend(labels=_labels, handles=_handles, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=4, fancybox=False, fontsize=11)
        _fig.savefig(BASE / 'plots' / f"time_benchmark_boxplot_{SHOW_SAMPLER}.pdf", bbox_inches='tight')
        plt.show()

        # Boxplot showing distribution of model performances per sampler
        _fig, _axes = plt.subplots(2, len(long_df.sampler.unique())//2+1, figsize=(12, 4), sharex=True, sharey=True, layout='constrained')
        _axes = _axes.flatten()
        for _idx, _sampler in enumerate(long_df.sampler.unique()):
            _subset = long_df[long_df['sampler'] == _sampler]
            _data_to_plot = []
            _std_to_plot = []
            if _idx == 0:
                _labels = []
            _colors_list = []
            for _model in _subset['model'].unique():
                _model_data = _subset[_subset['model'] == _model]
                _base_name = _model_data['model'].values[0]
                _sampler = _model_data['sampler'].values[0]
                _val = _model_data['error'].values[0]
                _data_to_plot.append(_val)
                _std_to_plot.append(_model_data['std'].values[0])
                _label = model_name(_base_name)
                if _idx == 0:
                    _labels.append(_label)
                _colors_list.append(colors.get(_base_name, 'gray'))
            _x_positions = np.arange(len(_data_to_plot))
            if len(_x_positions) != len(long_df.model.unique()):
                _x_positions += 5
            for _i, _color in enumerate(_colors_list):
                if _sampler != 'ode-euler' and 'Consistency' in _labels[_i]:
                    continue  # consistency models
                _axes[_idx].errorbar(_x_positions[_i], _data_to_plot[_i], yerr=_std_to_plot[_i],
                                     fmt='o', markersize=6, capsize=3, color=_color, markeredgewidth=0.5, label=_labels[_i])
            _axes[_idx].set_title(sampler_name(_sampler), fontsize=11)
            _axes[_idx].spines['right'].set_visible(False)
            _axes[_idx].spines['top'].set_visible(False)
            _axes[_idx].grid(True)
            _axes[_idx].set_xticks([])
        _axes[0].set_ylabel(r'$\mathrm{C2ST}$', fontsize=11)
        _axes[len(long_df.sampler.unique())//2+1].set_ylabel(r'$\mathrm{C2ST}$', fontsize=11)
        _axes[-1].set_visible(False)
        _handles = _fig.axes[4].get_legend_handles_labels()[0]
        _fig.legend(labels=_labels, handles=_handles, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=4, fancybox=False, fontsize=11)
        _fig.savefig(BASE / 'plots' / f"c2st_benchmark_boxplot_{SHOW_SAMPLER}.pdf", bbox_inches='tight')
        plt.show()
    return


@app.cell
def _(long_df):
    vp_edm_df = long_df[long_df['model'] == 'diffusion_edm_vp']
    vp_edm_df[['sampler', 'error', 'time']].sort_values(by='error')
    return


@app.cell
def _(long_df):
    fm_df = long_df[long_df['model'] == 'flow_matching']
    fm_df[['sampler', 'error', 'time']].sort_values(by='error')
    return


@app.cell
def _(
    SAMPLER_SETTINGS,
    colors,
    long_df_copy,
    model_name,
    model_order,
    plt,
    problem_names_nice,
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
        p: t for p, t in zip(problems_unique, problem_names_nice)
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
        ax.set_ylabel(r"$\mathrm{C2ST}$")
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

        #fig.savefig(BASE / 'plots' / f"c2st_benchmark_all_samplers_{problem}.pdf", bbox_inches="tight")
        plt.show()
    return


@app.cell
def _():
    import bayesflow as bf

    from model_settings_benchmark import load_model, MODELS
    return MODELS, bf, load_model


@app.cell
def _(MODELS, sbibm):
    model_i = 4
    task_name = sbibm.get_available_tasks()[0]
    task = sbibm.get_task(task_name)
    conf_tuple = list(MODELS.values())[model_i]
    model_name_test = list(MODELS.keys())[model_i]
    priors = task.get_prior()(1_000).numpy()
    print(model_name_test)
    print(task_name)
    storage = ''
    return conf_tuple, model_name_test, priors, storage, task, task_name


@app.cell
def _(conf_tuple, load_model, model_name_test, storage, task_name):
    workflow = load_model(conf_tuple=conf_tuple,
                          training_data=None,
                          simulator=None,
                          storage=storage,
                          problem_name=task_name, model_name=model_name_test,
                          use_ema=True)
    return (workflow,)


@app.cell
def _(np, task, task_name):
    test_data = {
        'observables': np.concatenate([task.get_observation(num_observation=i).numpy() 
                                       for i in range(1, 11)]),
        'parameters': np.concatenate([task.get_true_parameters(num_observation=i).numpy()
                                      for i in range(1, 11)])
    }

    if task_name == 'lotka_volterra':
        new_data = []
        for obs in test_data['observables']:
            new_data.append(np.array([obs[:10], obs[10:]]).T.flatten()[np.newaxis])
        test_data['observables'] = np.concatenate(new_data)
    return (test_data,)


@app.cell
def _(plt, test_data, workflow):
    diagnostics = workflow.plot_default_diagnostics(test_data=test_data)
    plt.show()
    return


@app.cell
def _(np, task, workflow):
    num_observation = 1
    observation = task.get_observation(num_observation=num_observation).numpy()
    observation = np.array([observation[0, :10], observation[0, 10:]]).T.flatten()[np.newaxis]
    reference_samples = task.get_reference_posterior_samples(num_observation=num_observation).numpy()[:1000]
    posterior_samples_dict = workflow.sample(conditions={'observables': observation}, num_samples=1_000)
    posterior_samples = posterior_samples_dict['parameters'][0]

    param_names = task.get_labels_parameters()
    true_params = task.get_true_parameters(num_observation=num_observation).numpy()[0]
    return param_names, posterior_samples, reference_samples, true_params


@app.cell
def _(bf, param_names, plt, posterior_samples, priors, true_params):
    bf.diagnostics.pairs_posterior(
        estimates=posterior_samples,
        priors=priors,
        targets=true_params,
        variable_names=param_names
    )
    plt.show()
    return


@app.cell
def _(bf, param_names, plt, priors, reference_samples, true_params):
    bf.diagnostics.pairs_posterior(
        estimates=reference_samples,
        priors=priors,
        targets=true_params,
        variable_names=param_names
    )
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return


if __name__ == "__main__":
    app.run()
