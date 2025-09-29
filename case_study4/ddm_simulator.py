import numpy as np
from scipy.stats import beta as beta_dist, norm as norm_dist


def beta_from_normal(z, a, b):
    u = norm_dist.cdf(z)
    x = beta_dist.ppf(u, a, b)  # Beta inverse CDF
    return x

# ---------------------------
# DDM simulator
# ---------------------------
def simulate_ddm_trial(nu, alpha, t0, beta, dt=1e-3, scale=1.0, max_time=10.0):
    """
    Simulates one realization of the diffusion process given
    a set of parameters and a step size `dt`.

    Parameters:
    -----------
    nu         : float
        The drift rate (rate of information uptake)
    alpha         : float
        The boundary separation (decision threshold).
    t0       : float
        Non-decision time (additive constant)
    beta      : float in [0, 1]
        Relative starting point (prior option preferences)
    dt        : float, optional (default: 1e-3 = 0.001)
        The step size for the Euler algorithm.
    scale     : float, optional (default: 1.0)
        The scale (sqrt(var)) of the Wiener process. Not considered
        a parameter and typically fixed to either 1.0 or 0.1.
    max_time  : float, optional (default: .10)
        The maximum number of seconds before forced termination.

    Returns:
    --------
    (x, c) - a tuple of response time (y - float) and a
        binary decision (c - int)
    """

    # Inits (process starts at relative starting point)
    y = beta * alpha
    rt = t0
    const = scale * np.sqrt(dt)

    # Loop through process and check boundary conditions
    while (alpha >= y >= 0) and rt <= max_time:
        # Perform diffusion equation
        z = np.random.randn()
        y += nu * dt + const * z

        # Increment step counter
        rt += dt

    if y >= alpha:
        c = 1.0
    else:
        c = 0.0
    return c, rt


def simulate_ddm(nu, alpha, t0, beta, n_subjects=1, n_trials=1):
    if isinstance(nu, (float, int)):
        nu = np.ones((n_subjects,)) * nu
        alpha = np.ones((n_subjects,)) * alpha
        t0 = np.ones((n_subjects,)) * t0
    data = np.zeros((n_subjects, n_trials, 2))
    for j_subject in range(n_subjects):
        for i_trial in range(n_trials):
            data[j_subject, i_trial] = simulate_ddm_trial(nu[j_subject], alpha[j_subject], t0[j_subject], beta)
    if n_subjects == 1 and n_trials == 1:
        data = data[0, 0]
    elif n_subjects == 1:
        data = data[0]
    elif n_trials == 1:
        data = data[:, 0]
    return dict(sim_data=data)
