import os

import matplotlib.pyplot as plt
import numpy as np

# metrics
METRICS = ['NRMSE', 'Posterior Contraction', 'Calibration Error', 'c2st']
METRIC_LABELS = {
    'NRMSE':r'NRMSE',
    'Posterior Contraction':r'$1-$Contraction',
    'Calibration Error':'Calibration\nError',
    'c2st':r'$\vert \text{C2ST}-0.5\vert$'}

# colors and styles
def _create_model_config():
    colors = {
        "flow_matching":"#1f77b4","ot_flow_matching":"#ff7f0e","flow_matching_edm":"#1abc9c",
        "consistency_model":"#2ca02c","stable_consistency_model":"#d62728",
        "diffusion_edm_vp":"#9467bd","diffusion_edm_ve":"#8c564b",
        "diffusion_cosine_F":"#e377c2","diffusion_cosine_v":"#7f7f7f","diffusion_cosine_noise":"#bcbd22",
        "ode":"#17becf","sde":"#ffbb78","sde-pc":"#98df8a","MCMC":"#000000",
        "flow_matching_ft": "#1f77b4", "ot_flow_matching_ft": "#ff7f0e", "flow_matching_edm_ft": "#1abc9c",
        "consistency_model_ft": "#2ca02c", "stable_consistency_model_ft": "#d62728",
        "diffusion_edm_vp_ft": "#9467bd", "diffusion_edm_ve_ft": "#8c564b",
        "diffusion_cosine_F_ft": "#e377c2", "diffusion_cosine_v_ft": "#7f7f7f", "diffusion_cosine_noise_ft": "#bcbd22"
    }
    return {"visualization":{"colors":colors}}

def _create_sampler_linestyles():
    return {'ode':'-','sde':'--','sde-pc':'-.'}

def _get_model_display_name(k):
    if k[-2:] == 'ft':
        k = k[:-2]
    names = {
        "flow_matching":"Flow Matching","ot_flow_matching":"Flow Matching (OT)","flow_matching_edm":"Flow Matching (EDM)",
        "consistency_model":"Discrete Consistency","stable_consistency_model":"Stable Consistency",
        "diffusion_edm_vp":"VP-EDM","diffusion_edm_ve":"VE-EDM",
        "diffusion_cosine_F":r"Cosine $\boldsymbol{F}$-pred.","diffusion_cosine_v":r"Cosine $\boldsymbol{v}$-pred.","diffusion_cosine_noise":r"Cosine $\boldsymbol{\epsilon}$-pred.",
        "MCMC":"MCMC"
    }
    return names.get(k, k.replace("_"," ").title())

def _get_method_groups(df, key):
    if key[-2:] == 'ft':
        key = key[:-2]
    groups = {}
    flow = ['flow_matching','ot_flow_matching','flow_matching_edm']
    cons = ['consistency_model','stable_consistency_model']
    diff = ['diffusion_edm_vp','diffusion_edm_ve','diffusion_cosine_F','diffusion_cosine_v','diffusion_cosine_noise']

    if key == 'flow': groups['Flow Methods'] = df[df['model'].isin(flow)]
    elif key == 'consistency': groups['Consistency Methods'] = df[df['model'].isin(cons)]
    elif key == 'diffusion': groups['Diffusion Methods'] = df[df['model'].isin(diff)]
    elif key == 'all':
        groups['Flow Methods'] = df[df['model'].isin(flow)]
        groups['Consistency Methods'] = df[df['model'].isin(cons)]
        groups['Diffusion Methods'] = df[df['model'].isin(diff)]
    elif key == 'edm_vs_cosine':
        groups['EDM Diffusion'] = df[df['model'].isin(['diffusion_edm_vp','diffusion_edm_ve'])]
        groups['Cosine Diffusion'] = df[df['model'].isin(['diffusion_cosine_F','diffusion_cosine_v','diffusion_cosine_noise'])]
    elif key == 'prediction_type':
        groups['F parameterization'] = df[df['model'].isin(['diffusion_edm_vp','diffusion_edm_ve','diffusion_cosine_F'])]
        groups['v parameterization'] = df[df['model'].isin(['diffusion_cosine_v'])]
        groups['eps parameterization'] = df[df['model'].isin(['diffusion_cosine_noise'])]

    return {k:v for k,v in groups.items() if not v.empty}

def _get_sampler_display_name(k):
    names = {'ode':'(ODE RK45)','sde':'(SDE Euler)','sde-pc':'(SDE PC Euler)', 'MCMC':''}
    return names.get(k, k.upper())

# helpers
def _angles(n, rot):
    a = [i / float(n) * 2 * np.pi + rot for i in range(n)]
    return a + a[:1]

def _plot(ax, angles, vals, label, color, ls, marker, alpha):
    v = vals + vals[:1]
    ax.plot(angles, v, label=label, color=color, linestyle=ls, marker=marker, linewidth=2, alpha=alpha)
    ax.fill(angles, v, color=color, alpha=0.05)

def _prep_metrics(df):
    d = df.copy()
    if 'posterior_contraction' in d.columns:
        d['Posterior Contraction'] = 1 - d['posterior_contraction']
    if 'c2st' in d.columns:
        d['c2st'] = np.abs(d['c2st'] - 0.5)
    if 'nrmse' in d.columns:
        d['NRMSE'] = d['nrmse']
    if 'posterior_calibration_error' in d.columns:
        d['Calibration Error'] = d['posterior_calibration_error']
    return d

# model comparison radar
def plot_model_comparison_radar(
    metrics_df,
    save_path=None, alpha=0.7, suptitle=False,
    group_by_sampler=False, group_by_method=None, rotation_angle=0.0
):
    config = _create_model_config()
    d = _prep_metrics(metrics_df)

    if group_by_method:
        groups = _get_method_groups(d, group_by_method)
        for name, data in groups.items():
            if data.empty:
                continue
            if group_by_sampler and 'sampler' in data.columns:
                for sp in data['sampler'].unique():
                    if sp == 'MCMC':
                        continue
                    print(sp)
                    sd = data[(data['sampler']==sp) | (data['sampler']=='MCMC')]
                    _plot_single_radar(sd, config, f"{name} {_get_sampler_display_name(sp)}",
                                       save_path, alpha, suptitle, rotation_angle, include_sampler=False)
            else:
                _plot_single_radar(data, config, f"{name}",
                                   save_path, alpha, suptitle, rotation_angle, include_sampler=True)
    elif group_by_sampler and 'sampler' in d.columns:
        for sp in d['sampler'].unique():
            if sp == 'MCMC':
                continue
            print(sp)
            sd = d[(d['sampler']==sp) | (d['sampler']=='MCMC')]
            _plot_single_radar(sd, config, f"{_get_sampler_display_name(sp)}",
                               save_path, alpha, suptitle, rotation_angle, include_sampler=False)
    else:
        _plot_single_radar(d, config, 'all', save_path, alpha, suptitle, rotation_angle, include_sampler=True)

def _plot_single_radar(df, config, title, save_path, alpha, suptitle, rotation_angle, include_sampler=False):
    metrics = [m for m in ['NRMSE','Posterior Contraction','Calibration Error','c2st'] if m in df.columns]
    if not metrics:
        return
    maxv = {m: 1 for m in metrics} #float(df[m].max()) for m in metrics}
    angles = _angles(len(metrics), rotation_angle)

    fig, ax = plt.subplots(figsize=(5,4), subplot_kw=dict(polar=True), layout='constrained')
    colors = config["visualization"]["colors"]
    ls_map = _create_sampler_linestyles()

    for model in df['model'].unique() if 'model' in df.columns else ['All']:
        if include_sampler and 'sampler' in df.columns:
            for sp in df['sampler'].unique():
                sub = df[(df['model']==model) & (df['sampler']==sp)]
                if sub.empty:
                    continue
                vals = [float(sub[m].mean())/max(maxv[m],1e-12) for m in metrics]
                label = f"{_get_model_display_name(model)} {_get_sampler_display_name(sp)}"
                _plot(ax, angles, vals, label, colors.get(model, "#000000"), ls_map.get(sp,'-'), 'o', alpha)
        else:
            sub = df[df['model']==model] if model != 'All' else df
            if sub.empty:
                continue
            vals = [float(sub[m].mean())/max(maxv[m],1e-12) for m in metrics]
            label = _get_model_display_name(model) if model != 'All' else 'All'
            _plot(ax, angles, vals, label, colors.get(model, "#000000"), '-', 'o', alpha)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.set_ylim(0, 1)
    ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])        # show scale
    ax.set_rlabel_position(rotation_angle or 0)     # place radial labels
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True)

    for i, m in enumerate(metrics):
        if m == 'NRMSE' or m == 'Calibration Error':
            y = 1.4
        else:
            y = 1.15
        ax.text(angles[i], y, f"{METRIC_LABELS.get(m,m)}", ha='center', size=11, color='black')  # {maxv[m]:.3f

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0),
                   ncol=min(len(handles),2), fontsize=11)

    if suptitle:
        fig.suptitle(title, fontsize=18, fontweight="bold", y=0.95)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fname = f"model_comparison_{title.lower().replace(' ','_')}"
        fig.savefig(os.path.join(save_path, f"{fname}.pdf"), bbox_inches='tight')

    plt.show()
