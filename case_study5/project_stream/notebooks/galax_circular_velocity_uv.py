# from autocvd import autocvd
# autocvd(num_gpus=1)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''


from functools import partial
import astropy.units as u
import astropy.coordinates as coord
import numpy as np
from jax import jit
import jax.random as jr
import jax.numpy as jnp
import jax

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
from unxt import Quantity

parameters_dict = {
    'm_Triaxial_halo': Quantity(1e12, u.Msun),
    'r_Triaxial_halo': Quantity(20, u.kpc),
    'q1_Triaxial_halo': 1,
    'q2_Triaxial_halo': 0.9,
    'rho_thin_disk': Quantity(0.1, u.Msun / u.pc**3),
    'hr_thin_disk': Quantity(3, u.kpc),
    'hz_thin_disk': Quantity(0.3, u.kpc),
    'rho_thick_disk': Quantity(0.01, u.Msun / u.pc**3),
    'hr_thick_disk': Quantity(3, u.kpc),
    'hz_thick_disk': Quantity(0.9, u.kpc),
    'm_bulge': Quantity(1e10, u.Msun),
    'r_bulge': Quantity(0.5, u.kpc),
    'alpha_bulge': 1.8
}


pot = gp.CompositePotential(

    halo = gp.TriaxialNFWPotential( m     = parameters_dict['m_Triaxial_halo'],
                                    r_s   = parameters_dict['r_Triaxial_halo'],
                                    q1     = parameters_dict['q1_Triaxial_halo'],
                                    q2     = parameters_dict['q2_Triaxial_halo'],
                                    units ="galactic"),

    thin_disk = gp.MN3ExponentialPotential(m_tot = 4 * np.pi * parameters_dict['rho_thin_disk']*parameters_dict['hr_thin_disk']**2 * parameters_dict['hz_thin_disk'],
                                                    h_R=parameters_dict['hr_thin_disk'],
                                                    h_z=parameters_dict['hz_thin_disk'],
                                                    units="galactic",
                                                    positive_density=True),
    thick_disk = gp.MN3ExponentialPotential(m_tot = 4 * np.pi * parameters_dict['rho_thick_disk'] *parameters_dict['hr_thick_disk']**2 * parameters_dict['hz_thick_disk'],
                                                    h_R=parameters_dict['hr_thick_disk'],
                                                    h_z=parameters_dict['hz_thick_disk'],
                                                    units="galactic",
                                                    positive_density=True),
    bulge = gp.PowerLawCutoffPotential(m_tot=parameters_dict['m_bulge'],
                                                r_c=parameters_dict['r_bulge'],
                                                alpha=parameters_dict['alpha_bulge'],
                                                units="galactic")
    )

def func(xyz):
    w = gc.PhaseSpaceCoordinate(q = Quantity([xyz[0], xyz[1], xyz[2]], "kpc"),
                                p=Quantity([0.0, 0.0, 0.0], "km/s"),
                                t=Quantity(0.0, "Gyr"))
    return pot.local_circular_velocity(w).value
w = jnp.array([8.0, 12.0, 4.0])
grad_of_circ = jax.grad(func)(w)

print(grad_of_circ)