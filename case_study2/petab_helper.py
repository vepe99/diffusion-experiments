import numpy as np


def scale_values(x, scale):
    if np.isscalar(x):
        return_scalar = True
    else:
        return_scalar = False
    x = np.atleast_1d(x)
    scale = np.atleast_1d(scale)

    if scale.size == 1:
        scale = np.repeat(scale.item(), x.size)

    out = np.empty_like(x, dtype=float)
    for i, (val, sc) in enumerate(zip(x, scale)):
        if sc == 'log10':
            out[i] = np.log10(val)
        elif sc == 'log':
            out[i] = np.log(val)
        elif sc == 'lin':
            out[i] = val
        else:
            raise ValueError(f"Unknown scale: {sc}")
    if return_scalar:
        return out.item()
    return out

def values_to_linear_scale(x, scale):
    if np.isscalar(x):
        return_scalar = True
    else:
        return_scalar = False
    x = np.atleast_1d(x)
    scale = np.atleast_1d(scale)

    if scale.size == 1:
        scale = np.repeat(scale.item(), x.size)

    out = np.empty_like(x, dtype=float)
    for i, (val, sc) in enumerate(zip(x, scale)):
        if sc == 'log10':
            out[i] = np.power(10, val)
        elif sc == 'log':
            out[i] = np.exp(val)
        elif sc == 'lin':
            out[i] = val
        else:
            raise ValueError(f"Unknown scale: {sc}")
    if return_scalar:
        return out.item()
    return out


def apply_noise_to_data(sim_df, params, field, pypesto_problem, petab_problem):
    # apply noise parameters to simulation
    for obs_id in sim_df['observableId'].unique():
        for cond_id in sim_df['simulationConditionId'].unique():
            obs_cond_index = (sim_df['observableId']==obs_id) & (sim_df['simulationConditionId']==cond_id)
            if obs_cond_index.sum() == 0:
                continue
            # get noise parameter for this observable
            obs_noise_param = sim_df.loc[obs_cond_index, 'noiseParameters'].unique()
            if len(obs_noise_param) > 1:
                print(obs_noise_param)
                raise ValueError(f"Multiple noise parameters for observable {obs_id}")

            # scale simulation according to observable transformation
            obs_transformation = petab_problem.observable_df.loc[obs_id, 'observableTransformation']
            scaled_sim = scale_values(sim_df.loc[obs_cond_index, field].values, obs_transformation)

            # scale parameters according to their transformation
            params_scaled = values_to_linear_scale(params, petab_problem.parameter_df['parameterScale'].values)

            # get noise parameter value
            obs_noise_param = obs_noise_param[0]
            obs_distribution = petab_problem.observable_df.loc[obs_id, 'noiseDistribution']
            obs_formula = petab_problem.observable_df.loc[obs_id, 'noiseFormula']
            if len(obs_formula.split('+')) == 2:
                obs_noise_params = obs_noise_param.split(';')
                if obs_noise_params[1] == '0':
                    noise_param_index = pypesto_problem.x_names.index(obs_noise_params[0])
                    noise_param_value = params_scaled[noise_param_index]
                else:
                    noise_param_index1 = pypesto_problem.x_names.index(obs_noise_params[0])
                    noise_param_index2 = pypesto_problem.x_names.index(obs_noise_params[1])
                    noise_param_value =  params_scaled[noise_param_index1] + params_scaled[noise_param_index2]
            elif isinstance(obs_noise_param, str):  # is a parameter
                noise_param_index = pypesto_problem.x_names.index(obs_noise_param)
                noise_param_value = params_scaled[noise_param_index]
            else:
                noise_param_value = float(obs_noise_param)  # is a nominal value

            # apply noise model
            if obs_distribution == 'normal':
                sim_df.loc[obs_cond_index, field] = scaled_sim + np.random.normal(0, 1, size=scaled_sim.shape) * noise_param_value
                sim_df.loc[obs_cond_index, field] = np.maximum(sim_df.loc[obs_cond_index, field], 0)  # avoid negative values in biological quantities
            else:
                raise ValueError(f"Unknown observable distribution: {obs_distribution}")
            sim_df.loc[obs_cond_index, field] = values_to_linear_scale(sim_df.loc[obs_cond_index, field].values,
                                                                       obs_transformation)
    return sim_df


def amici_pred_to_array(pred, params, factory, pypesto_problem, petab_problem):
    """
    Convert a

    Parameters
    ----------
    pred : pypesto.predict.PredictionResult
        The predictor object containing simulation results.
    params: array-like
        The parameter values used for the simulation.
    factory:
        pypesto factory object
    pypesto_problem:
        pypesto problem object
    petab_problem:
        petab problem object
    Returns
    -------
    array: a 2D numpy array of shape (n_timepoints, n_series) with NaNs for missing values.
          Each column corresponds to a unique (simulationConditionId, observableId) pair.
    """
    failed = False
    try:
        sim_df = factory.prediction_to_petab_simulation_df(pred)
        sim_df = apply_noise_to_data(sim_df, params, field='simulation',
                                     pypesto_problem=pypesto_problem, petab_problem=petab_problem)
    except ValueError as e:
        print("Simulation failed:", e)
        sim_df = petab_problem.measurement_df.copy()
        sim_df["simulation"] = 0  # will be set to nan later
        failed = True

    # all unique time points
    timepoints = np.sort(sim_df["time"].unique())

    # pivot to wide format, index=series, columns=time
    wide = sim_df.pivot_table(
        index=['simulationConditionId', 'observableId'],
        columns="time",
        values="simulation",
        aggfunc="first",  # just in case duplicates exist
        sort=True
    )
    if failed:
        wide = wide * np.nan  # set all to nan if simulation failed

    # reindex columns to include all time points
    wide = wide.reindex(columns=timepoints)

    arr = wide.to_numpy(dtype=float).T  # shape (n_timepoints, n_series)

    # inf to nan
    arr[np.isinf(arr)] = np.nan
    return arr, failed
