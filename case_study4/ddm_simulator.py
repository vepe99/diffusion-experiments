import numpy as np
from scipy.stats import beta as beta_dist, norm as norm_dist


def beta_from_normal(z, a, b):
    u = norm_dist.cdf(z)
    x = beta_dist.ppf(u, a, b)  # Beta inverse CDF
    return x

# ---------------------------
# DDM simulator
# ---------------------------
def simulate_ddm_trial(nu, alpha, t0, beta, max_t, dt=0.001, s=1.0):
    """
    One trial from a standard drift diffusion process with two absorbing bounds at 0 and alpha.
    Starting point is beta * alpha. Nondecision time t0 is used to delay the start of the process.
    Returns (choice, rt). choice in {0, 1}. rt in seconds. If no hit by max_t, returns (-1, max_t).
    """
    x = beta * alpha
    t = t0

    while t < max_t:
        x += nu * dt + s * np.sqrt(dt) * np.random.normal()
        t += dt
        if x >= alpha:
            return 1, t
        if x <= 0.0:
            return 0, t

    return -1, max_t


def simulate_ddm(nu, alpha, t0, beta, n_subjects=1, n_trials=1, max_t=1.0):
    if isinstance(nu, (float, int)):
        nu = np.ones((n_subjects,)) * nu
        alpha = np.ones((n_subjects,)) * alpha
        t0 = np.ones((n_subjects,)) * t0
    data = np.zeros((n_subjects, n_trials, 2))
    for j_subject in range(n_subjects):
        for i_trial in range(n_trials):
            data[j_subject, i_trial] = simulate_ddm_trial(nu[j_subject], alpha[j_subject], t0[j_subject], beta, max_t=max_t)
    if n_subjects == 1 and n_trials == 1:
        data = data[0, 0]
    elif n_subjects == 1:
        data = data[0]
    elif n_trials == 1:
        data = data[:, 0]
    return dict(sim_data=data)
