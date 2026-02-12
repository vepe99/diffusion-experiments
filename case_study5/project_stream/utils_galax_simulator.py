from functools import partial
import astropy.units as u
import astropy.coordinates as coord
import numpy as np
from jax import jit
import jax.random as jr
import jax.numpy as jnp

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
from unxt import Quantity


@partial(jit, static_argnames=('config', 'code_units'))
def simulate_stream_galax(parameters_dict, config, code_units, random_seed:int):
    '''
    Docstring for simulate_stream_galax
    code_units it is not used 
    
    :param parameters_dict: Description
    :param config: Description
    :type config: SimulationConfig
    :param code_units: Description
    :type code_units: CodeUnits
    :param random_seed: Description
    :type random_seed: int
    '''
    pot = gp.CompositePotential(

    halo = gp.TriaxialNFWPotential( m     = parameters_dict['m_Triaxial_halo'][0],
                                    r_s   = parameters_dict['r_Triaxial_halo'][0],
                                    q1     = parameters_dict['q1_Triaxial_halo'][0],
                                    q2     = parameters_dict['q2_Triaxial_halo'][0],
                                    units ="galactic"),

    thin_disk = gp.MN3ExponentialPotential(m_tot = 4 * np.pi * parameters_dict['rho_thin_disk'][0]*parameters_dict['hr_thin_disk'][0]**2 * parameters_dict['hz_thin_disk'][0],
                                                    h_R=parameters_dict['hr_thin_disk'][0],
                                                    h_z=parameters_dict['hz_thin_disk'][0],
                                                    units="galactic",
                                                    positive_density=True),
    thick_disk = gp.MN3ExponentialPotential(m_tot = 4 * np.pi * parameters_dict['rho_thick_disk'][0] *parameters_dict['hr_thick_disk'][0]**2 * parameters_dict['hz_thick_disk'][0],
                                                    h_R=parameters_dict['hr_thick_disk'][0],
                                                    h_z=parameters_dict['hz_thick_disk'][0],
                                                    units="galactic",
                                                    positive_density=True),
    bulge = gp.PowerLawCutoffPotential(m_tot=parameters_dict['m_bulge'][0],
                                                r_c=parameters_dict['r_bulge'][0],
                                                alpha=parameters_dict['alpha_bulge'][0],
                                                units="galactic")
    )

    # w0 = coord.Galactocentric(x=parameters_dict['x'][0], y=parameters_dict['y'][0], z=parameters_dict['z'][0],
    #                         v_x=parameters_dict['vx'][0]*u.km/u.s, v_y=parameters_dict['vy'][0]*u.km/u.s, v_z=parameters_dict['vz'][0]*u.km/u.s)
    w0 = gc.PhaseSpacePosition(q = Quantity([parameters_dict['x'][0], parameters_dict['y'][0], parameters_dict['z'][0]], "kpc"),
                                p = Quantity([parameters_dict['vx'][0], parameters_dict['vy'][0],parameters_dict['vz'][0]], "km/s"),
                                )

    t_end = parameters_dict['t_end'][0] * u.Gyr.to(u.Myr)
    t_array = Quantity(-jnp.linspace(0, t_end, int(config.n_timesteps/2)), "Myr")


    prog_mass = Quantity(parameters_dict['m_progenitor'][0], "Msun")

    if config.df_type == "ChenStreamDF":
        df = gd.ChenStreamDF()
    elif config.df_type == "FardalStreamDF": 
        df = gd.FardalStreamDF()
    gen = gd.MockStreamGenerator(df, pot)
    func = lambda k, t, w0, prog_mass: gen.run(k, t, w0, prog_mass)
    stream, _ = func(jr.key(random_seed), t_array, w0, prog_mass)

    return jnp.array([stream.q.x.value, stream.q.y.value,stream.q.z.value, stream.p.x.value, stream.p.y.value, stream.p.z.value]).T




