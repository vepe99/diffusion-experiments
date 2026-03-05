from functools import partial
import astropy.units as u
import astropy.coordinates as coord
import numpy as np
from jax import jit
import jax.random as jr
import jax.numpy as jnp
from jax.scipy import special
from math import log10

import StreaMAX


@partial(jit, static_argnames=('config', 'code_units'))
def simulate_stream_StreaMAX(parameters_dict, config, code_units, random_seed:int):
    '''
    Docstring for simulate_stream_StreaMAX
    code_units it is not used 
    
    :param parameters_dict: Description
    :param config: Description
    :type config: SimulationConfig
    :param code_units: Description
    :type code_units: CodeUnits
    :param random_seed: Description
    :type random_seed: int
    '''
    
    q_min = 0.5
    q_max = 1.5
    flip = jnp.where(parameters_dict['dirz_Triaxial_rotated_halo'][0] < 0, -1.0, 1.0)
    dirx = parameters_dict['dirx_Triaxial_rotated_halo'][0] * flip
    diry = parameters_dict['diry_Triaxial_rotated_halo'][0] * flip
    dirz = parameters_dict['dirz_Triaxial_rotated_halo'][0] * flip
    r = jnp.sqrt(dirx**2 + diry**2 + dirz**2)
    u_from_r = special.erf(r/jnp.sqrt(2)) - jnp.sqrt(2/jnp.pi)*r*jnp.exp(-(r**2)/2)
    q = q_min + (q_max - q_min) * u_from_r

    # Flattened NFW halo
    type_host   = config.type_host
    if type_host == "MW2022":
        params_halo = {'logM': jnp.log10(parameters_dict['m_Triaxial_rotated_halo'][0]), 
                    'Rs': parameters_dict['r_Triaxial_rotated_halo'][0], 
                        'a':1.0, 'b': 1.0, 'c': q,
                        'dirx': dirx, 'diry': diry, 'dirz': dirz,
                        'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0}
        params_disk = {'logM': jnp.log10(parameters_dict['M_MN3ExpDisk'][0]), 
                        'Rs': parameters_dict['r_MN3ExpDisk'][0], 'Hs': parameters_dict['z_MN3ExpDisk'][0], 
                        'positive_density': True, 'sech2_z':True,
                        'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
                        'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0}
        params_bulge = {'logM': jnp.log10(parameters_dict['m_bulge_2022'][0]), 'Rs': parameters_dict['R_bulge_2022'][0],
                    'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
                    'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0}
        params_nucleus = {'logM': jnp.log10(parameters_dict['m_nucleus'][0]), 'Rs': parameters_dict['R_nucleus'][0],
                    'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
                    'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0}
        params_host = {'halo_params': params_halo, 
                    'disk_params': params_disk, 
                    'bulge_params': params_bulge, 
                    'nucleus_params': params_nucleus}
    elif type_host == "MW2014":
        params_halo = {'logM': jnp.log10(parameters_dict['m_Triaxial_rotated_halo'][0]), 
                    'Rs': parameters_dict['r_Triaxial_rotated_halo'][0], 
                        'a':1.0, 'b': 1.0, 'c': q,
                        'dirx': dirx, 'diry': diry, 'dirz': dirz,
                        'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0}
        params_disk = {'logM': jnp.log10(parameters_dict['m_disk_MW2014'][0]), 
                        'Rs': parameters_dict['r_disk_MW2014'][0], 'Hs': parameters_dict['z_disk_MW2014'][0], 
                        'positive_density': True, 'sech2_z':True,
                        'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
                        'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0}
        params_bulge = {'logM': jnp.log10(parameters_dict['m_bulge'][0]), 
                        'Rs': parameters_dict['r_bulge'][0],
                    'dirx': 0.0, 'diry': 0.0, 'dirz': 1.0,
                    'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0}
        params_host = {'halo_params': params_halo, 
                    'disk_params': params_disk, 
                    'bulge_params': params_bulge}

    # Plummer Sattelite
    type_sat   = 'Plummer'
    params_sat = {'logM': jnp.log10(parameters_dict['m_progenitor'][0]), 'Rs': parameters_dict['a_progenitor'][0],
                    'x_origin': 0.0, 'y_origin': 0.0, 'z_origin': 0.0}

    # Initial conditions
    xv_f = jnp.array([parameters_dict['x'][0], parameters_dict['y'][0], parameters_dict['z'][0],  # Position in kpc
                    parameters_dict['vx'][0] * (u.km/u.s).to(u.kpc/u.Gyr), parameters_dict['vy'][0] * (u.km/u.s).to(u.kpc/u.Gyr), parameters_dict['vz'][0] * (u.km/u.s).to(u.kpc/u.Gyr)])   # Velocity in kpc/Gyr

    # Integration time
    time  = parameters_dict['t_end'][0] # Gyr
    alpha = 1.

    n_particles = config.N_particles
    n_steps     = config.n_timesteps # n_steps+1 must be a factor of n_particles

    # m_f_sat = jnp.log10(parameters_dict['m_progenitor'][0])
    m_f_sat = 0.0

    t_sat, xv_sat, xv_stream, xhi_stream = StreaMAX.generate_stream(xv_f, 
                                                                    type_host, params_host, 
                                                                    type_sat, params_sat, 
                                                                    time, alpha, n_steps,
                                                                    n_particles, 
                                                                    config.unroll, type_method=config.type_method,
                                                                    m_f_sat=m_f_sat)
    xv_stream =xv_stream.at[:, 3:].set(xv_stream[:,3:]* (u.kpc/u.Gyr).to(u.km/u.s)) # Convert velocities back to km/s
    
    #xv_stream has shape (n_particles, 6) and contains the phase-space coordinates of the stream particles at the final time step.
    return xv_stream




