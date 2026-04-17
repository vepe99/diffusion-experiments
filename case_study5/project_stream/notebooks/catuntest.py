import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from galpy.potential import (mwpotentials, 
                             mwpot_helpers, 
                             DiskSCFPotential,
                             scf_compute_coeffs_axi,
                             SCFPotential,
                             PlummerPotential,  #for the self gravity
                             )
from galpy.df import chen24spraydf
from galpy import potential
from galpy.util.conversion import get_physical
from galpy.orbit import Orbit
from galpy.util import conversion

import warnings
from astropy.coordinates import SkyCoord
from astropy import units as u  
import matplotlib.pyplot as plt
import numpy as np
## print old test
# cautun20 = mwpotentials.Cautun20
# for p in cautun20:
#     print(f"  {type(p).__name__}")
# print(f"\nPhysical units: {get_physical(cautun20)}")

path_to_plot = '../data/plots/galpy_test'
os.makedirs(path_to_plot, exist_ok=True)



ro = 8.122
vo = 229
#progentior initial position from astropy coordinates
#pal5
sc = SkyCoord(
    ra= 229.022 * u.deg,
    dec= -0.112 * u.deg,
    distance=20.6 * u.kpc,
    pm_ra_cosdec= -2.736 * u.mas / u.yr,
    pm_dec= -2.646 * u.mas / u.yr,
    radial_velocity=-58.6 * u.km / u.s,
     )
prog = Orbit(sc, 
             ro=ro, 
             vo=vo
             ) 

#potentials  
#bulge
cautun20 = mwpotentials.Cautun20
pot_bulge = cautun20[2]
print(f"Bulge potential: {type(pot_bulge).__name__}")

#disk thin and thick #we are neglecting the gas disk, but we could import it

sigo = conversion.surfdens_in_msolpc2(vo=vo, ro=ro)
rhoo = conversion.dens_in_msolpc3(vo=vo, ro=ro)
zd_thin = 0.3 / ro
Rd_thin = 2.63 / ro
Sigma0_thin = 731.0 / sigo
zd_thick = 0.9 / ro
Rd_thick = 3.80 / ro
Sigma0_thick = 101.0 / sigo

def stellar_dens(R, z):
    return mwpot_helpers.expexp_dens(
        R, z, Rd_thin, zd_thin, Sigma0_thin
    ) + mwpot_helpers.expexp_dens(R, z, Rd_thick, zd_thick, Sigma0_thick)
# dicts used in DiskSCFPotential
sigmadict = [
    {"type": "exp", "h": Rd_thin, "amp": Sigma0_thin, "Rhole": 0.0},
    {"type": "exp", "h": Rd_thick, "amp": Sigma0_thick, "Rhole": 0.0},
]

hzdict = [
    {"type": "exp", "h": zd_thin},
    {"type": "exp", "h": zd_thick},
]

# pot_disk = DiskSCFPotential(
#     dens=lambda R, z:  stellar_dens(R, z),
#     Sigma=sigmadict,
#     hz=hzdict,
#     a=2.5,
#     N=30,
#     L=30,
#     ro=ro,
#     vo=vo,
# )

#we try to use directly the TwoPotwerTriaxialPotential, maybe better to use SCF on it
rho0 = 0.01 / rhoo #this should 0.02 Msun/pc^3, which is 2*10^7 Msun/kpc^3 
rho0 = 0.01
a = 20.0 / ro
a = 20.0    
alpha = 0.06
beta = 3.0
q = 1.2


# def halo_density_base(s, rho0, a, alpha, beta,):
#     return rho0 * (s / a) ** (-alpha) * (1 + (s / a)) ** (alpha - beta)

# def halo_density_flattened(R, z, rho0, a, alpha, beta, q):
#     s = (R**2 + (z/q)**2)**0.5
#     return halo_density_base(s, rho0=rho0, a=a, alpha=alpha, beta=beta, )

# def halo_potential(R, z):
#     return halo_density_base(R, z, rho0, a, alpha, beta, q)

# pot_halo = SCFPotential.from_density(
#     dens=lambda R, z, phi=0: halo_density_flattened(R, z, rho0, a, alpha, beta, q),
#     N=30,
#     L=30,
#     a=25.0, #/ ro,          # <-- also pass in natural units (divide by ro)
#     symmetry="axisymmetry",
#     ro=ro,
#     vo=vo,
# )


pot_disk_thin = potential.MN3ExponentialDiskPotential(
    amp=Sigma0_thin,
    hr=Rd_thin,
    hz=zd_thin,
    ro=ro,
    vo=vo,
)

pot_disk_thick = potential.MN3ExponentialDiskPotential(
    amp=Sigma0_thick,
    hr=Rd_thick,
    hz=zd_thick,
    ro=ro,  
    vo=vo,
)

pot_disk = pot_disk_thin + pot_disk_thick


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


#we are ready to integrate (hopefully)
pot = pot_bulge + pot_disk + pot_halo

progpot = PlummerPotential(4.3*1e3 * u.Msun, 8.43 * u.pc,  ) #progenitor potential, we are using a plummer sphere with the same mass and size as the progenitor in the chen24spraydf paper, but we could use something else
spdf_with_prog = chen24spraydf(
    progenitor_mass= 4.3*1e3 * u.Msun,
    progenitor=prog,
    pot=pot,
    tdisrupt=4.0 * u.Gyr,
    tail="both",
    # progpot=progpot,
    ro=ro,      # <-- add these
    vo=vo,      # <-- add these
)

stream_with_prog = spdf_with_prog.sample(n=500)
plt.plot(stream_with_prog.x(), stream_with_prog.z(), "k.", ms=1)
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
plt.title("chen24spraydf with progenitor gravity")
plt.savefig(os.path.join(path_to_plot, "chen24spraydf_with_prog.png"), dpi=300)
print(f"\nSaved plot to '{os.path.join(path_to_plot, 'chen24spraydf_with_prog.png')}'.")