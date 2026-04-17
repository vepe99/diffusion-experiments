import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from gala.potential import scf
from gala import potential 
from scipy.special import hyp2f1, gamma


path_to_plot = '../data/plots/galascf_test'
os.makedirs(path_to_plot, exist_ok=True)
# ── density definitions ──────────────────────────────────────────────────────

def halo_density(s, rho0, a, alpha, beta,):
    return rho0 * (s / a) ** (-alpha) * (1 + (s / a)) ** (alpha - beta)

def halo_density_flattened(x, y, z, rho0, a, alpha, beta, q):
    s = np.sqrt(x**2 + y**2 + (z / q)**2)
    return halo_density(s, rho0, a, alpha, beta)


# ── prior sampler ─────────────────────────────────────────────────────────────

def sample_halo_prior(rng=None):
    """Draw one sample from the uniform priors defined in the YAML."""
    if rng is None:
        rng = np.random.default_rng()
    rho0  = rng.uniform(1.0e5,  1.5e8)
    alpha = rng.uniform(-2.0,   2.0)
    a     = rng.uniform(1.0,    100.0)
    beta  = 3.0                          # identity prior
    q     = rng.uniform(0.5,    1.5)
    return dict(rho0=rho0, alpha=alpha, a=a, beta=beta, q=q)


# ── main diagnostic function ──────────────────────────────────────────────────

def plot_halo_scf_comparison(
    params=None,
    nmax=8,
    lmax=6,
    seed=42,
    r_min=2,
    r_max=100.0,
    n_points=128,
):
    """
    1. Draw one sample from the halo prior (or use supplied params).
    2. Compute SCF coefficients with M=1, r_s=1.
    3. Plot true density vs SCF approximation along x-axis and diagonal.

    Parameters
    ----------
    params : dict or None
        If None, a sample is drawn from the prior.
    nmax, lmax : int
        Expansion order. Increase for better accuracy (slower).
    seed : int
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    if params is None:
        params = sample_halo_prior(rng)

    params['rho0'] = 1
    rho0  = params["rho0"]
    params['a'] = 1
    a     = params["a"]
    alpha = params["alpha"]
    beta  = params["beta"]
    q     = params["q"]
    R = params["a"] 
    mass = (4.0 *np.pi
            * params["a"]**params["alpha"]
            * R ** (3.0 - params["alpha"])
            / (3.0 - params["alpha"])
            * params["beta"]
            * params["q"]
            * hyp2f1(
                3.0 - params["alpha"], params["beta"] - params["alpha"], 4.0 - params["alpha"], -R / params["a"]))
    mass = 1.0
    params["mass"] = mass
    print("Halo parameters:")
    for k, v in params.items():
        print(f"  {k:6s} = {v:.4g}")

    # ── compute SCF coefficients ──────────────────────────────────────────────
    # M=1, r_s=1 as requested; flattened → axisymmetric → skip_m=True
    # beta=3 (NFW-like) can cause slow convergence at large r, but is fine here
    print(f"\nComputing SCF coefficients (nmax={nmax}, lmax={lmax}) ...")

    (S, Serr), _ = scf.compute_coeffs(
        halo_density_flattened,
        nmax=nmax,
        lmax=lmax,
        M=params["mass"],
        r_s=params["a"],
        args=(params["rho0"], params["a"], params["alpha"], params["beta"], params["q"]),
        S_only=True,       # T coefficients vanish for this symmetric profile
        skip_m=True,       # axisymmetric: no phi dependence
        progress=True,
    )

    # ── build SCF potential object ────────────────────────────────────────────
    pot = scf.SCFPotential(Snlm=S, Tnlm=np.zeros_like(S), m=params["mass"], r_s=params["a"])

    # ── evaluation grids ──────────────────────────────────────────────────────
    r = np.logspace(np.log10(r_min), np.log10(r_max), n_points)

    # along x-axis: (r, 0, 0)
    xyz_x = np.zeros((3, n_points))
    xyz_x[0] = r

    # along diagonal: (r, 0, r) — probes the flattening in z
    xyz_d = np.zeros((3, n_points))
    xyz_d[0] = r
    xyz_d[2] = r

    true_x = halo_density_flattened(r, 0, 0,  rho0, a, alpha, beta, q)
    true_d = halo_density_flattened(r, 0, r,  rho0, a, alpha, beta, q)

    scf_x  = pot.density(xyz_x).value   # returns an Astropy Quantity
    scf_d  = pot.density(xyz_d).value

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, true, approx, label in zip(
        axes,
        [true_x, true_d],
        [scf_x,  scf_d],
        ["Along x-axis  (y=0, z=0)", "Diagonal  (y=0, z=x)"],
    ):
        ax.plot(r, true,   lw=2,          label="True density")
        ax.plot(r, approx, lw=2, ls="--", label=f"SCF  (nmax={nmax}, lmax={lmax})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("r  [arbitrary units]")
        ax.set_ylabel(r"$\rho$  [arbitrary units]")
        ax.set_title(label)
        ax.legend()

    title = (
        rf"Halo:  $\rho_0$={rho0:.2e},  $a$={a:.2f},  "
        rf"$\alpha$={alpha:.2f},  $\beta$={beta:.1f},  $q$={q:.2f}"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(path_to_plot, 'halo_scf_comparison.png'), dpi=300)
    print(f"\nSaved plot to '{os.path.join(path_to_plot, 'halo_scf_comparison.png')}'.")
    plt.show()

    return params, S, pot


def plot_herquist_comparison(
    params=None,
    nmax=8,
    lmax=6,
    seed=42,
    r_min=0.1,
    r_max=100.0,
    n_points=128,
):
    """
    Compare SCF approximation to the Hernquist profile, which has an analytic SCF expansion.

    Parameters
    ----------
    nmax, lmax : int
        Expansion order. Increase for better accuracy (slower).
    seed : int
        RNG seed for reproducibility.
    """
    from gala.potential import HernquistPotential
    rng = np.random.default_rng(seed)

    if params is None:
        params = sample_halo_prior(rng)

    params['rho0'] = 1
    rho0  = params["rho0"]
    params['a'] = 1
    a     = params["a"]
    alpha = params["alpha"]
    beta  = params["beta"]
    q     = params["q"]
    R = params["a"] 
    mass = (4.0 *np.pi
            * params["a"]**params["alpha"]
            * R ** (3.0 - params["alpha"])
            / (3.0 - params["alpha"])
            * params["beta"]
            * params["q"]
            * hyp2f1(
                3.0 - params["alpha"], params["beta"] - params["alpha"], 4.0 - params["alpha"], -R / params["a"]))
    mass = 1.0
    params["mass"] = mass
    print("Halo parameters:")
    for k, v in params.items():
        print(f"  {k:6s} = {v:.4g}")
    
    hern = potential.HernquistPotential(m=1, c=1)

    # ── evaluation grids ──────────────────────────────────────────────────────
    r = np.logspace(np.log10(r_min), np.log10(r_max), n_points)

    # along x-axis: (r, 0, 0)
    xyz_x = np.zeros((3, n_points))
    xyz_x[0] = r

    # along diagonal: (r, 0, r) — probes the flattening in z
    xyz_d = np.zeros((3, n_points))
    xyz_d[0] = r
    xyz_d[2] = r

    true_x = halo_density_flattened(r, 0, 0,  rho0, a, alpha, beta, q)
    true_d = halo_density_flattened(r, 0, r,  rho0, a, alpha, beta, q)

    hern_x  = hern.density(xyz_x).value   # returns an Astropy Quantity
    hern_d  = hern.density(xyz_d).value

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, true, approx, label in zip(
        axes,
        [true_x, true_d],
        [hern_x,  hern_d],
        ["Along x-axis  (y=0, z=0)", "Diagonal  (y=0, z=x)"],
    ):
        ax.plot(r, true,   lw=2,          label="True density")
        ax.plot(r, approx, lw=2, ls="--", label=f"Hernquist ")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("r  [arbitrary units]")
        ax.set_ylabel(r"$\rho$  [arbitrary units]")
        ax.set_title(label)
        ax.legend()

    title = (
        rf"Halo:  $\rho_0$={rho0:.2e},  $a$={a:.2f},  "
        rf"$\alpha$={alpha:.2f},  $\beta$={beta:.1f},  $q$={q:.2f}"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(path_to_plot, 'halo_hernquist_comparison.png'), dpi=300)
    print(f"\nSaved plot to '{os.path.join(path_to_plot, 'halo_hernquist_comparison.png')}'.")
    plt.show()





# ── run it ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # params, S, pot = plot_halo_scf_comparison(seed=42)
    plot_herquist_comparison(seed=42)
