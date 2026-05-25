# """JAX-free agama worker. Called via subprocess, communicates via stdin/stdout JSON."""
# import sys
# import json
# import numpy as np
# import agama


# def main():
#     agama.setUnits(length=1, velocity=1, mass=1)
#     timeUnitGyr = agama.getUnits()['time'] / 1e3  # time unit is 1 kpc / (1 km/s)
#     data     = json.loads(sys.stdin.read())
#     p        = data['p']
#     obs_R    = np.array(data['obs_R'])
#     points   = np.column_stack((obs_R, np.zeros_like(obs_R), np.zeros_like(obs_R)))

#     pot_params = [
#         dict(type='Spheroid',
#              scaleRadius=75/1e3, densityNorm=9.6e10,
#              gamma=0, alpha=1, beta=1.8, cutoffStrength=2,
#              outerCutoffRadius=2.1, axisRatioY=1.0, axisRatioZ=0.5),
#         dict(type='Spheroid',
#              scaleRadius=p['a_TwoPowerTriaxial_halo'],
#              densityNorm=p['rho_TwoPowerTriaxial_halo'],
#              gamma=p['gamma_TwoPowerTriaxial_halo'],
#              alpha=1, beta=3, cutoffStrength=2, outerCutoffRadius=np.inf,
#              axisRatioY=1.0, axisRatioZ=p['q_TwoPowerTriaxial_halo']),
#         dict(type='Disk',
#              scaleRadius=p['r_Disk'], scaleHeight=p['z_Disk'],
#              surfaceDensity=p['Sigma_Disk'],
#              sersicIndex=1, innerCutoffRadius=0),
#     ]
#     pot    = agama.Potential(*pot_params)
#     forces = pot.force(points)
#     v2     = -obs_R * forces[:, 0]
#     if np.any(v2 < 0):
#         sys.exit(1)
#     print(json.dumps(np.sqrt(v2).tolist()))

# if __name__ == '__main__':
#     main()



"""JAX-free agama worker. Called via subprocess, communicates via stdin/stdout JSON.

Payload keys
------------
p       : dict of posterior parameters (required)
obs_R   : list of radii in kpc (required for vcirc)
compute : one of "vcirc" | "m200" | "both"  (default: "vcirc")
H0      : Hubble constant in km/s/kpc        (default: 67.4e-3)

Output (always a JSON dict)
---------------------------
{
  "vcirc"  : [...]      # km/s, present when compute in {"vcirc","both"}
  "R200"   : float,     # kpc,  present when compute in {"m200","both"}
  "M200"   : float,     # Msun, present when compute in {"m200","both"}
}
"""
import sys
import json
import numpy as np
import agama
from scipy.optimize import brentq

# ── physical constants (agama units: kpc, km/s, Msun) ────────────────────────
G_KPC_KMS_MSUN = 4.3009e-6   # kpc (km/s)^2 Msun^{-1}
H0_DEFAULT      = 70e-3    # km/s/kpc  (Planck 2018: 67.4 km/s/Mpc)

from astropy.cosmology import default_cosmology
from astropy import units as u
from astropy.constants import G
# ── helpers ───────────────────────────────────────────────────────────────────

def _rho_crit(H0: float) -> float:
    """Critical density in Msun/kpc^3:  rho_c = 3 H0^2 / (8 pi G)."""
    rho_c = 3.0 * H0**2 / (8.0 * np.pi * G_KPC_KMS_MSUN)
    # cosmo = default_cosmology.get()
    # rho_c = (3 * cosmo.H(0.0) ** 2 / (8 * np.pi * G)).to(u.Msun / u.kpc**3)
    return rho_c


def _build_halo_potential(p: dict) -> agama.Potential:
    """Halo-only Spheroid potential (identical params to the full pot below)."""
    return agama.Potential(
        dict(
            type              = 'Spheroid',
            densityNorm       = p['rho_TwoPowerTriaxial_halo'],
            scaleRadius       = p['a_TwoPowerTriaxial_halo'],
            gamma             = p['gamma_TwoPowerTriaxial_halo'],
            alpha             = 1,
            beta              = 3,          # fixed
            cutoffStrength    = 2,
            outerCutoffRadius = np.inf,     # no truncation
            axisRatioY        = 1.0,
            axisRatioZ        = p['q_TwoPowerTriaxial_halo'],
        )
    )


def _compute_m200(p: dict, H0: float, r_search: tuple = (10.0, 2000.0)):
    """
    Solve for R_200 and M_200 of the halo-only potential.

    Criterion:  M(<r) / (4pi/3 r^3)  =  200 * rho_crit
    agama.Potential.enclosedMass(r) integrates the density over a sphere of
    radius r, which is the correct quantity for the spherical overdensity
    definition even when axisRatioZ != 1.
    """
    rho_c   = _rho_crit(H0)
    halo    = _build_halo_potential(p)

    def overdensity(r):
        M        = halo.enclosedMass(r)
        rho_mean = M / (4.0 / 3.0 * np.pi * r**3)
        return rho_mean / rho_c - 200.0

    r_min, r_max = r_search
    d_min, d_max = overdensity(r_min), overdensity(r_max)

    if d_min * d_max > 0:
        # bracket failed — profile may be unusual; signal with NaNs
        return np.nan, np.nan

    R200 = brentq(overdensity, r_min, r_max, xtol=0.1, rtol=1e-4)
    M200 = halo.enclosedMass(R200)
    return float(R200), float(M200)


# ── main entry point ──────────────────────────────────────────────────────────

def main():
    agama.setUnits(length=1, velocity=1, mass=1)

    data    = json.loads(sys.stdin.read())
    p       = data['p']
    mode    = data.get('compute', 'vcirc')   # NEW: default keeps old behaviour
    H0      = data.get('H0', H0_DEFAULT)     # NEW: optional cosmology override
    out     = {}

    # ── circular velocity ─────────────────────────────────────────────────────
    if mode in ('vcirc', 'both'):
        obs_R  = np.array(data['obs_R'])
        points = np.column_stack((obs_R, np.zeros_like(obs_R), np.zeros_like(obs_R)))

        pot_params = [
            dict(type='Spheroid',
                 scaleRadius=75/1e3, densityNorm=9.6e10,
                 gamma=0, alpha=1, beta=1.8, cutoffStrength=2,
                 outerCutoffRadius=2.1, axisRatioY=1.0, axisRatioZ=0.5),
            dict(type='Spheroid',
                 scaleRadius=p['a_TwoPowerTriaxial_halo'],
                 densityNorm=p['rho_TwoPowerTriaxial_halo'],
                 gamma=p['gamma_TwoPowerTriaxial_halo'],
                 alpha=1, beta=3, cutoffStrength=2, outerCutoffRadius=np.inf,
                 axisRatioY=1.0, axisRatioZ=p['q_TwoPowerTriaxial_halo']),
            dict(type='Disk',
                 scaleRadius=p['r_Disk'], scaleHeight=p['z_Disk'],
                 surfaceDensity=p['Sigma_Disk'],
                 sersicIndex=1, innerCutoffRadius=0),
        ]
        pot    = agama.Potential(*pot_params)
        forces = pot.force(points)
        v2     = -obs_R * forces[:, 0]
        if np.any(v2 < 0):
            sys.exit(1)
        out['vcirc'] = np.sqrt(v2).tolist()

    # ── M200 / R200 ───────────────────────────────────────────────────────────
    if mode in ('m200', 'both'):
        R200, M200    = _compute_m200(p, H0)
        out['R200']   = R200
        out['M200']   = M200

    print(json.dumps(out))


if __name__ == '__main__':
    main()