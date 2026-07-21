"""Compositional TARP coverage test for the joint global posterior.

Runs the TARP (Tests of Accuracy with Random Points, Lemos et al. 2023) coverage
diagnostic on the *joint* posterior over the global (halo + disk) parameters,
using the compositional posterior samples produced by the rotation-curve eval.

The posterior file only stores the posterior samples, so the ground-truth
parameter values are reconstructed here exactly as the eval did:
  * load the multistream truth simulation,
  * drop simulations whose cartesian sim_data contains NaNs (333 -> 299),
  * derive the disk mass  M_Disk = 4*pi * Sigma_Disk * r_Disk**2 * z_Disk.

Usage:
    .venv/bin/python case_study5/project_stream/tarp_global.py \
        --posterior case_study5/project_stream/data/hyperparameter_tuning/agama/rotationcurve/model_5/333test/posterior.npz
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import tarp

# Order of the joint parameter vector: must match the keys stored in posterior.npz.
# The first 7 come straight from the truth simulation; M_Disk is derived below.
PARAM_ORDER = [
    "rho_TwoPowerTriaxial_halo",
    "gamma_TwoPowerTriaxial_halo",
    "a_TwoPowerTriaxial_halo",
    "q_TwoPowerTriaxial_halo",
    "r_Disk",
    "z_Disk",
    "Sigma_Disk",
    "$M_Disk$",
]
PARAM_LABELS = [
    r"$\rho_{NFW}$", r"$\gamma_{NFW}$", r"$a_{NFW}$", r"$q_{NFW}$",
    r"$r_{D}$", r"$z_{D}$", r"$\Sigma_{D}$", r"$M_{D}$",
]


def reconstruct_truth(truth_path):
    """Load the multistream truth and reproduce the eval's NaN filtering + M_Disk."""
    td = dict(np.load(truth_path, allow_pickle=True))
    # Same mask the eval uses: keep simulations with no NaN in the cartesian sim_data.
    valid_mask = ~np.isnan(td["sim_data_carthesian"]).any(axis=(-1, -2, -3))
    td = {k: v[valid_mask] for k, v in td.items()}
    td["$M_Disk$"] = 4 * np.pi * td["Sigma_Disk"] * td["r_Disk"] ** 2 * td["z_Disk"]
    print(f"Truth simulations kept after NaN filter: {int(valid_mask.sum())} / {valid_mask.size}")
    return td


def build_arrays(posterior_path, truth_path, repeat=None):
    """Return (samples, theta) with shapes (n_samples, n_sims, n_dims) / (n_sims, n_dims).

    ``repeat`` handles the non-compositional case: there each of the ``N`` valid
    multistream simulations is evaluated once per stream, so the posterior has
    ``N * n_streams`` datasets. The eval expands the shared global truth with
    ``np.repeat(param, n_streams, axis=0)``, which we mirror here. If ``repeat`` is
    None it is inferred from the posterior/truth simulation counts.
    """
    ps = dict(np.load(posterior_path, allow_pickle=True))
    td = reconstruct_truth(truth_path)

    missing = [k for k in PARAM_ORDER if k not in ps or k not in td]
    if missing:
        raise KeyError(f"Missing parameters in posterior/truth: {missing}")

    n_post = ps[PARAM_ORDER[0]].shape[0]
    n_truth = td[PARAM_ORDER[0]].shape[0]
    if repeat is None:
        if n_post % n_truth != 0:
            raise ValueError(
                f"Cannot align: posterior has {n_post} sims, truth has {n_truth} "
                f"(not a multiple). Pass --repeat explicitly."
            )
        repeat = n_post // n_truth
    if repeat != 1:
        # Mirror the non-compositional eval: np.repeat expands each sim's shared
        # global truth across its `repeat` streams (sim0,sim0,sim0, sim1,...).
        td = {k: np.repeat(v, repeat, axis=0) for k, v in td.items()}
        print(f"Non-compositional: repeated truth x{repeat} -> {td[PARAM_ORDER[0]].shape[0]} datasets")

    # posterior samples are stored as (n_sims, n_samples, 1) -> transpose to (n_samples, n_sims)
    samples = np.stack([ps[k][..., 0].T for k in PARAM_ORDER], axis=-1)
    theta = np.stack([td[k][..., 0] for k in PARAM_ORDER], axis=-1)

    if samples.shape[1] != theta.shape[0]:
        raise ValueError(
            f"Simulation count mismatch: posterior has {samples.shape[1]} sims, "
            f"truth has {theta.shape[0]}. TARP requires them aligned."
        )
    print(f"samples shape {samples.shape}, theta shape {theta.shape}")
    return samples, theta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_post = (
        "case_study5/project_stream/data/hyperparameter_tuning/agama/"
        "rotationcurve/model_5/333test/posterior.npz"
    )
    default_truth = (
        "case_study5/project_stream/data/streams/data_multistream_agama/"
        "simulation_multistream_333.npz"
    )
    parser.add_argument("--posterior", default=default_post, help="path to posterior.npz")
    parser.add_argument("--truth", default=default_truth, help="path to truth simulation npz")
    parser.add_argument("--output", default=None, help="output pdf (default: <posterior dir>/global_tarp.pdf)")
    parser.add_argument("--repeat", type=int, default=None,
                        help="repeat truth per sim (non-compositional: n_streams). Default: inferred.")
    parser.add_argument("--num-bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_path = args.output or os.path.join(os.path.dirname(args.posterior), "global_tarp.pdf")

    samples, theta = build_arrays(args.posterior, args.truth, repeat=args.repeat)

    # norm=True standardizes each dimension by the sample mean/std, which is essential
    # here because the parameters span very different scales (e.g. q ~ O(1) vs M_Disk).
    #
    # NOTE: we do NOT use tarp's built-in bootstrap=True. In tarp 0.1.1 it swaps a single
    # simulation per iteration and mutates the arrays in place cumulatively (drp.py), so the
    # curves barely differ and the error bars collapse to ~1e-4. Instead we run a proper
    # non-parametric bootstrap: resample all sims with replacement each iteration.
    n_sims = theta.shape[0]
    rng = np.random.default_rng(args.seed)
    boot_ecp = []
    for _ in range(args.num_bootstrap):
        idx = rng.integers(0, n_sims, n_sims)
        ecp_b, alpha = tarp.get_tarp_coverage(
            samples[:, idx, :], theta[idx],
            references="random", metric="euclidean", norm=True, bootstrap=False,
        )
        boot_ecp.append(ecp_b)
    ecp = np.asarray(boot_ecp)

    ecp_mean = ecp.mean(axis=0)
    ecp_std = ecp.std(axis=0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], ls="--", color="k", label="Ideal")
    ax.plot(alpha, ecp_mean, color="C0", label="TARP (joint global posterior)")
    ax.fill_between(alpha, ecp_mean - ecp_std, ecp_mean + ecp_std, color="C0", alpha=0.3)
    ax.fill_between(alpha, ecp_mean - 2 * ecp_std, ecp_mean + 2 * ecp_std, color="C0", alpha=0.15)
    ax.set_xlabel("Credibility level")
    ax.set_ylabel("Expected coverage probability")
    ax.set_title(f"Compositional TARP — {samples.shape[1]} sims, {len(PARAM_ORDER)} params")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(False)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved TARP coverage plot to {out_path}")

    # Also save the raw curves for later reuse.
    npz_out = os.path.splitext(out_path)[0] + ".npz"
    np.savez(npz_out, alpha=alpha, ecp=ecp, ecp_mean=ecp_mean, ecp_std=ecp_std,
             param_order=np.array(PARAM_ORDER))
    print(f"Saved TARP curves to {npz_out}")


if __name__ == "__main__":
    main()
