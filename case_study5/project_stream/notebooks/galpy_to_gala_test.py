import os
from galpy.potential import (mwpotentials, 
                             mwpot_helpers, 
                             DiskSCFPotential,
                             scf_compute_coeffs_axi,
                             SCFPotential,
                             PlummerPotential,  #for the self gravity
                             )
from galpy.util import conversion
from galpy import potential
from gala.potential.potential.interop import galpy_to_gala_potential

import numpy as np

import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic
from gala.dynamics import mockstream as ms
from astropy import units as u

path_to_plot = '../data/plots/galpy_gala_test'

pot = gp.CCompositePotential()

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

parameters_dict = sample_halo_prior()
parameters_dict["hz_thin_disk"] = [0.3]
parameters_dict["hr_thin_disk"] = [2.63]
parameters_dict["rho_thin_disk"] = [731.0 * (u.Msun / u.pc**2).to(u.Msun / u.kpc**2)]
parameters_dict["hz_thick_disk"] = [0.9]
parameters_dict["hr_thick_disk"] = [3.80]
parameters_dict["rho_thick_disk"] = [101.0 * (u.Msun / u.pc**2).to(u.Msun / u.kpc**2)]

pot['thin_disk'] = gp.MN3ExponentialDiskPotential(m = 4 * np.pi * parameters_dict['rho_thin_disk'][0]*parameters_dict['hr_thin_disk'][0]**2 * parameters_dict['hz_thin_disk'][0],
                                                    h_R=parameters_dict['hr_thin_disk'][0],
                                                    h_z=parameters_dict['hz_thin_disk'][0],
                                                    units=galactic,
                                                    positive_density=True)
pot['thick_disk'] = gp.MN3ExponentialDiskPotential(m = 4 * np.pi * parameters_dict['rho_thick_disk'][0] *parameters_dict['hr_thick_disk'][0]**2 * parameters_dict['hz_thick_disk'][0],
                                                    h_R=parameters_dict['hr_thick_disk'][0],
                                                    h_z=parameters_dict['hz_thick_disk'][0],
                                                    units=galactic,
                                                    positive_density=True)

#galpy part
ro = 8.122
vo = 229
sigo = conversion.surfdens_in_msolpc2(vo=vo, ro=ro)
rhoo = conversion.dens_in_msolpc3(vo=vo, ro=ro)
rho0 = 0.01 / rhoo #this should 0.02 Msun/pc^3, which is 2*10^7 Msun/kpc^3 
# rho0 = 0.01
a = 20.0 / ro
# a = 20.0    
alpha = 0.06
beta = 3.0
q = 1.2

pot_halo = potential.TwoPowerTriaxialPotential(
    amp=rho0,
    a=a,
    alpha=alpha,
    beta=beta,
    b=1.0,
    c=q,
    ro=ro,
    vo=vo,
)

pot['halo'] = galpy_to_gala_potential(pot_halo)


pot_bulge = mwpotentials.Cautun20[2]
pot['bulge'] = galpy_to_gala_potential(pot_bulge)


grid = np.linspace(-10., 10., 100)
from matplotlib import colors
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(5,5))
fig = pot.plot_contours(grid=(grid, grid),
                        levels=np.logspace(-3, 1, 10),
                        norm=colors.LogNorm(),
                        cmap='Blues', ax=ax)
fig.savefig(os.path.join(path_to_plot, 'galpy_gala_test.png'))
print('The potential has been plotted and saved to', os.path.join(path_to_plot, 'galpy_gala_test.png'))

