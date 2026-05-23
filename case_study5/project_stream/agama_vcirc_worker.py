"""JAX-free agama worker. Called via subprocess, communicates via stdin/stdout JSON."""
import sys
import json
import numpy as np
import agama


def main():
    agama.setUnits(length=1, velocity=1, mass=1)
    timeUnitGyr = agama.getUnits()['time'] / 1e3  # time unit is 1 kpc / (1 km/s)
    data     = json.loads(sys.stdin.read())
    p        = data['p']
    obs_R    = np.array(data['obs_R'])
    points   = np.column_stack((obs_R, np.zeros_like(obs_R), np.zeros_like(obs_R)))

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
    print(json.dumps(np.sqrt(v2).tolist()))

if __name__ == '__main__':
    main()