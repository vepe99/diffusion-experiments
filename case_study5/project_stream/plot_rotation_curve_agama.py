# """
# Standalone rotation curve plotter using agama.
# Follows the same Hydra config structure as the eval script.
# Agama is run in a subprocess to avoid XLA/JAX signal handler conflicts.

# Usage:
#     uv run plot_rotation_curve_agama.py
#     (picks up the same config as eval_config_gaia_new_rotationcurve_agama)
# """
# import os
# os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# import subprocess
# import sys
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import hydra
# from hydra.core.config_store import ConfigStore

# # JAX-dependent imports — these are fine here since agama runs in a subprocess
# from config.EvalConfig import EvalConfig
# from utils.utils_train_jax_new_rotationcurve import AugmentationsClass

# cs = ConfigStore.instance()
# cs.store(name="eval_config", node=EvalConfig)

# WORKER = os.path.join(os.path.dirname(__file__), 'agama_vcirc_worker.py')


# def compute_vcirc_subprocess(p, obs_R_np, timeout=60):
#     """Call agama in a clean subprocess with no JAX loaded."""
#     payload = json.dumps({'p': p, 'obs_R': obs_R_np.tolist()})
#     try:
#         result = subprocess.run(
#             [sys.executable, WORKER],
#             input=payload,
#             capture_output=True,
#             text=True,
#             timeout=timeout,
#         )
#         if result.returncode == 0:
#             return np.array(json.loads(result.stdout))
#         else:
#             print(f"  agama worker failed (exit {result.returncode}):\n{result.stderr[:300]}")
#             return None
#     except subprocess.TimeoutExpired:
#         print("  agama worker timed out")
#         return None


# @hydra.main(
#     version_base=None,
#     config_path="config",
#     config_name="eval_config_gaia_new_rotationcurve_agama",
# )
# def main(cfg: EvalConfig):

#     # ── paths ─────────────────────────────────────────────────────────────────
#     posterior_path = os.path.join(cfg.base_dir, cfg.results_dir, 'global_posterior.npz')
#     output_path    = os.path.join(cfg.base_dir, cfg.results_dir, 'global_rotation_curve.pdf')
#     print(f"Loading posterior from: {posterior_path}")
#     print(f"Output will be saved to: {output_path}")

#     # ── observed rotation curve (same mask as training/eval) ──────────────────
#     obs_R   = np.array([5.24,5.74,6.25,6.77,7.23,7.83,8.21,8.78,9.26,9.75,
#                     10.25,10.75,11.25,11.75,12.24,12.74,13.25,13.74,14.23,14.74,
#                     15.23,15.74,16.24,16.74,17.23,17.74,18.35,18.90,19.50,20.41,
#                     21.28,22.39,23.16,24.00])          # kpc
#     obs_sVc = np.array([0.69,0.68,0.62,0.60,0.45,0.29,0.26,0.22,0.17,0.16,
#                         0.17,0.18,0.19,0.20,0.25,0.27,0.27,0.31,0.40,0.43,
#                         0.50,0.68,0.74,0.87,1.02,1.15,1.45,1.58,1.32,1.71,
#                         1.69,2.01,2.50,4.94])  
#     obs_Vc  = np.array([225.10,233.53,234.30,233.17,236.19,236.00,233.19,233.15,232.15,231.24,
#                 230.34,230.54,229.11,227.48,226.69,225.56,224.90,223.57,221.10,220.19,
#                 219.59,217.36,216.61,217.28,216.25,213.81,217.53,212.10,210.46,206.69,
#                 207.71,203.72,205.20,200.64])       # km/s
#     mask_r_kpc = (obs_R >5.5)

#     obs_Vc = obs_Vc[mask_r_kpc]
#     obs_sVc = obs_sVc[mask_r_kpc]
#     obs_R_np = np.concatenate((np.linspace(0.1, 5.2, 30), obs_R[mask_r_kpc]))

#     N_obs     = len(obs_R_np)
#     print(f"Observed rotation curve: {N_obs} radial points, "
#           f"R in [{obs_R_np.min():.1f}, {obs_R_np.max():.1f}] kpc")

#     # ── posterior samples ─────────────────────────────────────────────────────
#     ps     = dict(np.load(posterior_path))
#     params = [
#         'rho_TwoPowerTriaxial_halo',
#         'gamma_TwoPowerTriaxial_halo',
#         'a_TwoPowerTriaxial_halo',
#         'q_TwoPowerTriaxial_halo',
#         'r_Disk',
#         'z_Disk',
#         'Sigma_Disk',
#     ]
#     for k in params:
#         print(f"  {k}: shape {ps[k].shape}")

#     # flatten (1, 5000, 1) → (5000,)
#     ps_flat     = {k: ps[k].reshape(-1) for k in params}
#     n_available = ps_flat[params[0]].shape[0]
#     N_sample    = min(1000, n_available)

#     rng     = np.random.default_rng(42)
#     indices = rng.choice(n_available, size=N_sample, replace=False)

#     # ── compute rotation curves via agama subprocess ───────────────────────────
#     all_vcirc = np.full((N_sample, N_obs), np.nan)

#     for j, idx in enumerate(tqdm(indices, desc="Computing v_circ")):
#         p = {k: float(ps_flat[k][idx]) for k in params}
#         if j == 0:
#             print(f"\nFirst sample params: {p}")
#         result = compute_vcirc_subprocess(p, obs_R_np)
#         if result is not None:
#             all_vcirc[j] = result

#     valid     = ~np.isnan(all_vcirc[:, 0])
#     all_vcirc = all_vcirc[valid]
#     print(f"\n{valid.sum()}/{N_sample} samples succeeded")

#     # if valid.sum() == 0:
#     #     print("No valid samples — cannot plot. Exiting.")
#     #     return

#     # ── plot ──────────────────────────────────────────────────────────────────
#     fig, ax = plt.subplots(figsize=(10, 6))

#     for row in all_vcirc:
#         ax.plot(obs_R_np, row, color='grey', alpha=0.4, lw=0.4)

#     mean_v = np.mean(all_vcirc, axis=0)
#     std_v  = np.std(all_vcirc,  axis=0)

#     ax.plot(obs_R_np, mean_v, color='steelblue', lw=2, label='Posterior mean')
#     ax.fill_between(
#         obs_R_np,
#         mean_v - 3 * std_v,
#         mean_v + 3 * std_v,
#         color='steelblue', alpha=0.3,
#         label='Posterior ±3σ',
#     )
#     ax.errorbar(
#         obs_R[mask_r_kpc], obs_Vc, yerr=3 * obs_sVc,
#         fmt='o', color='crimson', ms=4, lw=1, capsize=3,
#         label='Zhou et al. (2023) ±3σ', zorder=5,
#     )

#     ax.set_xlabel('Radius [kpc]')
#     ax.set_ylabel('Circular velocity [km/s]')
#     ax.legend()
#     fig.tight_layout()
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     fig.savefig(output_path)
#     print(f"Saved to {output_path}")
#     plt.close(fig)


# if __name__ == '__main__':
#     main()



import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import subprocess
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import hydra
from hydra.core.config_store import ConfigStore

from config.EvalConfig import EvalConfig
from utils.utils_train_jax_new_rotationcurve import AugmentationsClass

cs = ConfigStore.instance()
cs.store(name="eval_config", node=EvalConfig)

WORKER = os.path.join(os.path.dirname(__file__), 'agama_vcirc_worker.py')


# CHANGED: accepts an optional `compute` argument; always returns a dict or None
def call_worker(p, obs_R_np=None, compute='vcirc', H0=67.4e-3, timeout=60):
    """Generic subprocess wrapper for the agama worker."""
    payload = {'p': p, 'compute': compute, 'H0': H0}
    if obs_R_np is not None:
        payload['obs_R'] = obs_R_np.tolist()
    try:
        result = subprocess.run(
            [sys.executable, WORKER],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)   # always a dict now
        else:
            print(f"  worker failed (exit {result.returncode}):\n{result.stderr[:300]}")
            return None
    except subprocess.TimeoutExpired:
        print("  worker timed out")
        return None


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="eval_config_gaia_new_rotationcurve_agama",
)
def main(cfg: EvalConfig):

    # ── paths ──────────────────────────────────────────────────────────────────
    posterior_path  = os.path.join(cfg.base_dir, cfg.results_dir, 'global_posterior.npz')
    output_vcirc    = os.path.join(cfg.base_dir, cfg.results_dir, 'global_rotation_curve.pdf')
    output_m200     = os.path.join(cfg.base_dir, cfg.results_dir, 'global_M200_R200.pdf')  # NEW
    print(f"Loading posterior from: {posterior_path}")

    # ── observed rotation curve ────────────────────────────────────────────────
    obs_R   = np.array([5.24,5.74,6.25,6.77,7.23,7.83,8.21,8.78,9.26,9.75,
                    10.25,10.75,11.25,11.75,12.24,12.74,13.25,13.74,14.23,14.74,
                    15.23,15.74,16.24,16.74,17.23,17.74,18.35,18.90,19.50,20.41,
                    21.28,22.39,23.16,24.00])
    obs_sVc = np.array([0.69,0.68,0.62,0.60,0.45,0.29,0.26,0.22,0.17,0.16,
                        0.17,0.18,0.19,0.20,0.25,0.27,0.27,0.31,0.40,0.43,
                        0.50,0.68,0.74,0.87,1.02,1.15,1.45,1.58,1.32,1.71,
                        1.69,2.01,2.50,4.94])
    obs_Vc  = np.array([225.10,233.53,234.30,233.17,236.19,236.00,233.19,233.15,232.15,231.24,
                230.34,230.54,229.11,227.48,226.69,225.56,224.90,223.57,221.10,220.19,
                219.59,217.36,216.61,217.28,216.25,213.81,217.53,212.10,210.46,206.69,
                207.71,203.72,205.20,200.64])
    mask_r_kpc = (obs_R > 5.5)
    obs_Vc     = obs_Vc[mask_r_kpc]
    obs_sVc    = obs_sVc[mask_r_kpc]
    obs_R_np   = np.concatenate((np.linspace(0.1, 5.2, 30), obs_R[mask_r_kpc]))
    N_obs      = len(obs_R_np)

    # ── posterior samples ──────────────────────────────────────────────────────
    ps     = dict(np.load(posterior_path))
    params = [
        'rho_TwoPowerTriaxial_halo',
        'gamma_TwoPowerTriaxial_halo',
        'a_TwoPowerTriaxial_halo',
        'q_TwoPowerTriaxial_halo',
        'r_Disk', 'z_Disk', 'Sigma_Disk',
    ]
    ps_flat     = {k: ps[k].reshape(-1) for k in params}
    n_available = ps_flat[params[0]].shape[0]
    N_sample    = min(1000, n_available)
    rng         = np.random.default_rng(42)
    indices     = rng.choice(n_available, size=N_sample, replace=False)

    # ── main loop: vcirc + M200/R200 in one subprocess call each ──────────────
    all_vcirc = np.full((N_sample, N_obs), np.nan)
    all_R200  = np.full(N_sample, np.nan)   # NEW
    all_M200  = np.full(N_sample, np.nan)   # NEW

    for j, idx in enumerate(tqdm(indices, desc="Computing v_circ + M200")):
        p = {k: float(ps_flat[k][idx]) for k in params}

        # CHANGED: ask worker to compute both at once
        result = call_worker(p, obs_R_np=obs_R_np, compute='both')
        if result is None:
            continue

        if 'vcirc' in result:
            v2_ok = True
            vcirc = np.array(result['vcirc'])
            all_vcirc[j] = vcirc

        # NEW: store M200 / R200
        R200 = result.get('R200', np.nan)
        M200 = result.get('M200', np.nan)
        if R200 is not None and not np.isnan(R200):
            all_R200[j] = R200
            all_M200[j] = M200

    # ── rotation-curve plot (unchanged logic) ──────────────────────────────────
    valid     = ~np.isnan(all_vcirc[:, 0])
    vc_valid  = all_vcirc[valid]
    print(f"\n{valid.sum()}/{N_sample} vcirc samples succeeded")

    fig, ax = plt.subplots(figsize=(10, 6))
    for row in vc_valid:
        ax.plot(obs_R_np, row, color='grey', alpha=0.4, lw=0.4)
    mean_v = np.mean(vc_valid, axis=0)
    std_v  = np.std(vc_valid,  axis=0)
    ax.plot(obs_R_np, mean_v, color='steelblue', lw=2, label='Posterior mean')
    ax.fill_between(obs_R_np, mean_v - 3*std_v, mean_v + 3*std_v,
                    color='steelblue', alpha=0.3, label='Posterior ±3σ')
    ax.errorbar(obs_R[mask_r_kpc], obs_Vc, yerr=3*obs_sVc,
                fmt='o', color='crimson', ms=4, lw=1, capsize=3,
                label='Zhou et al. (2023) ±3σ', zorder=5)
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Circular velocity [km/s]')
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_vcirc), exist_ok=True)
    fig.savefig(output_vcirc)
    print(f"Saved rotation curve → {output_vcirc}")
    plt.close(fig)

    # ── NEW: M200 / R200 posterior plot ───────────────────────────────────────
    valid_m   = ~np.isnan(all_R200)
    R200_ok   = all_R200[valid_m]
    M200_ok   = all_M200[valid_m]
    logM200   = np.log10(M200_ok)
    print(f"{valid_m.sum()}/{N_sample} M200 samples succeeded")
    print(f"  R200: {np.median(R200_ok):.1f}  ± {np.std(R200_ok):.1f} kpc  "
          f"(16–84th: {np.percentile(R200_ok,16):.1f}–{np.percentile(R200_ok,84):.1f})")
    print(f"  log10 M200: {np.median(logM200):.3f} ± {np.std(logM200):.3f}  "
          f"(16–84th: {np.percentile(logM200,16):.3f}–{np.percentile(logM200,84):.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(R200_ok, bins=40, color='turquoise', lw=0.4, density=True)
    axes[0].axvline(np.median(R200_ok), color='k',      lw=1.5, label='Median')
    axes[0].axvline(np.percentile(R200_ok, 16), color='k', lw=1, ls='--')
    axes[0].axvline(np.percentile(R200_ok, 84), color='k', lw=1, ls='--', label='16/84th pct')
    axes[0].set_xlabel('R$_{200}$ [kpc]')
    axes[0].set_ylabel('Normalized Counts')
    axes[0].legend()

    axes[1].hist(logM200, bins=40, color='aquamarine',  lw=0.4, density=True)
    axes[1].axvline(np.median(logM200), color='k',      lw=1.5, label='Median')
    axes[1].axvline(np.percentile(logM200, 16), color='k', lw=1, ls='--')
    axes[1].axvline(np.percentile(logM200, 84), color='k', lw=1, ls='--', label='16/84th pct')
    axes[1].set_xlabel('$\log_{10}(M_{200})$ [$M_\odot$]')
    axes[1].set_ylabel('Normalized Counts')
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_m200)
    np.savez(output_m200.replace('.png', '.npz'), R200=all_R200, M200=all_M200)
    print(f"Saved M200/R200 posterior → {output_m200}")
    plt.close(fig)


if __name__ == '__main__':
    main()