import numpy as np
from astropy.coordinates import SkyCoord, Galactocentric, ICRS
import astropy.units as u
from functools import partial 
from jax import jit, random
import jax.numpy as jnp

from odisseo import construct_initial_state
from odisseo.units import CodeUnits
from odisseo.time_integration import time_integration
from odisseo.initial_condition import Plummer_sphere
from odisseo.option_classes import SimulationConfig, SimulationParams
from odisseo.option_classes import (PlummerParams, 
                                    MNParams, 
                                    NFWParams, 
                                    PSPParams, 
                                    TriaxialNFWParams, 
                                    TwoPowerTriaxialParams, 
                                    ThinMN3DiskParams, 
                                    ThickMN3DiskParams)
from odisseo.option_classes import (TRIAXIAL_NFW_POTENTIAL,
                                    TWO_POWER_TRIAXIAL, 
                                    THIN_MN3_DISK,
                                    THICK_MN3_DISK, 
                                    PSP_POTENTIAL,
                                    DIRECT_ACC_MATRIX, 
                                    DIFFRAX_BACKEND,
                                    TSIT5,
)


def get_prior_sample(paramter_prior_dict: dict, size=None):
    if paramter_prior_dict['type'] == 'uniform':
        low, high = paramter_prior_dict['prior_parameters']
        return np.random.uniform(low, high, size=(size, 1))
    elif paramter_prior_dict['type'] == 'normal':
        mean, std = paramter_prior_dict['prior_parameters']
        return np.random.normal(mean, std, size=(size, 1))
    elif paramter_prior_dict['type'] == 'identity':
        return np.full(shape=(size, 1), fill_value=paramter_prior_dict['prior_parameters'][0])
    
def sample_parameters(prior_global_dict: dict, prior_local_dict: dict, n_samples: int, target_streams: dict, key_seed=0):
    np.random.seed(key_seed)
    global_keys = list(prior_global_dict.keys())
    possible_j = list(target_streams.values())

    first_stream_key = next(iter(prior_local_dict))
    local_param_keys = list(prior_local_dict[first_stream_key].keys())
    all_keys = global_keys + local_param_keys + ['j']

    output = {k: np.zeros(n_samples) for k in all_keys}

    # Vectorized sampling for global parameters
    for key in global_keys:
        output[key] = get_prior_sample(prior_global_dict[key], size=n_samples)

    # Vectorized sampling for j
    js = np.random.choice(possible_j, size=(n_samples, 1))
    output['j'] = js

    # Vectorized sampling for local parameters
    stream_keys = {v: k for k, v in target_streams.items()}
    for key in local_param_keys:
        # For each sample, get the correct stream_key and sample
        output[key] = np.array([
            get_prior_sample(prior_local_dict[stream_keys[j[0]]][key], size=1)[0]
            for j in js
        ])

    # we are going to go to carthesian coordinates to simulate
    c = SkyCoord(
        ra = output['ra'] * u.degree,
        dec = output['dec'] * u.degree,
        distance = output['r'] * u.kpc,
        pm_ra_cosdec = output['mu_ra_cosdec'] * u.mas/u.yr,
        pm_dec = output['mu_dec'] * u.mas/u.yr,
        radial_velocity = output['vr'] * u.km/u.s,  
        frame='icrs'
    )   
    gc = c.transform_to(Galactocentric)
    output['x'] = gc.x.to(u.kpc).value.reshape(-1, 1)
    output['y'] = gc.y.to(u.kpc).value.reshape(-1, 1)
    output['z'] = gc.z.to(u.kpc).value.reshape(-1, 1)
    output['vx'] = gc.v_x.to(u.km/u.s).value.reshape(-1, 1)
    output['vy'] = gc.v_y.to(u.km/u.s).value.reshape(-1, 1)
    output['vz'] = gc.v_z.to(u.km/u.s).value.reshape(-1, 1)

    return output

def sample_parameters_parallel(prior_global_dict: dict, prior_local_dict: dict, n_samples: int, target_streams: dict, key_seed=0,):
    '''
    The number of paralel stream to be sample in parallel with global parameter is going to be inferred from target_streams.
    
    :param prior_global_dict: Description
    :type prior_global_dict: dict
    :param prior_local_dict: Description
    :type prior_local_dict: dict
    :param n_samples: Description
    :type n_samples: int
    :param target_streams: Description
    :type target_streams: dict
    :param key_seed: Description
    '''


    np.random.seed(key_seed)
    global_keys = list(prior_global_dict.keys())
    possible_j = list(target_streams.values()) #this is a list of j, which are int

    first_stream_key = next(iter(prior_local_dict))
    local_param_keys = list(prior_local_dict[first_stream_key].keys())
    all_keys = global_keys + local_param_keys + ['j']

    output = {k: np.zeros(n_samples) for k in all_keys}

    # Vectorized sampling for global parameters
    for key in global_keys:
        output[key] = get_prior_sample(prior_global_dict[key], size=n_samples)

    # Vectorized sampling for j
    js = np.random.choice(possible_j, size=(n_samples, 1))
    output['j'] = np.repeat(possible_j, n_samples).reshape(-1, len(possible_j), 1) #j has shape (n_samples, n_streams, 1) and each column is a different stream, we will use this to sample the local parameters in a vectorized way


    # Vectorized sampling for local parameters
    stream_keys = {v: k for k, v in target_streams.items()} #this gets the string of the stream given the j value, we will use this to get the correct local parameters for each stream
    for key in local_param_keys:
        output[key] = np.zeros((n_samples, len(possible_j), 1))
        for n in range(n_samples):
            for j in range(len(possible_j)):
                output[key][n, j] = get_prior_sample(prior_local_dict[stream_keys[j]][key], size=1)[0]
        
    # we are going to go to carthesian coordinates to simulate
    c = SkyCoord(
        ra = output['ra'] * u.degree,
        dec = output['dec'] * u.degree,
        distance = output['r'] * u.kpc,
        pm_ra_cosdec = output['mu_ra_cosdec'] * u.mas/u.yr,
        pm_dec = output['mu_dec'] * u.mas/u.yr,
        radial_velocity = output['vr'] * u.km/u.s,  
        frame='icrs'
    )   
    gc = c.transform_to(Galactocentric)
    output['x'] = gc.x.to(u.kpc).value.reshape(-1, len(possible_j), 1)
    output['y'] = gc.y.to(u.kpc).value.reshape(-1, len(possible_j), 1)
    output['z'] = gc.z.to(u.kpc).value.reshape(-1, len(possible_j), 1)
    output['vx'] = gc.v_x.to(u.km/u.s).value.reshape(-1, len(possible_j), 1)
    output['vy'] = gc.v_y.to(u.km/u.s).value.reshape(-1, len(possible_j), 1)
    output['vz'] = gc.v_z.to(u.km/u.s).value.reshape(-1, len(possible_j), 1)

    return output

def convert_to_integer_externalacc(ext_acc):
    external_acc_int = []
    for acc in ext_acc:
        if acc == 'PSP_POTENTIAL':
            external_acc_int.append(PSP_POTENTIAL)
        elif acc == 'TWO_POWER_TRIAXIAL':
            external_acc_int.append(TWO_POWER_TRIAXIAL)
        elif acc == 'THIN_MN3_DISK':
            external_acc_int.append(THIN_MN3_DISK)
        elif acc == 'THICK_MN3_DISK':
            external_acc_int.append(THICK_MN3_DISK)
        elif acc == 'TRIAXIAL_NFW_POTENTIAL':
            external_acc_int.append(TRIAXIAL_NFW_POTENTIAL)
    return tuple(external_acc_int)

def convert_to_integer_config(config):
    if config == 'DIRECT_ACC_MATRIX':
        return DIRECT_ACC_MATRIX
    elif config == 'DIFFRAX_BACKEND':
        return DIFFRAX_BACKEND
    elif config == 'TSIT5':
        return TSIT5

def sky_projection_astropy(batch_sim_data):
    batch_shape = batch_sim_data.shape[0]
    n_particles = batch_sim_data.shape[1]
    x = batch_sim_data[:, :, 0].reshape(-1) * u.kpc
    y = batch_sim_data[:, :, 1].reshape(-1) * u.kpc
    z = batch_sim_data[:, :, 2].reshape(-1) * u.kpc
    vx = batch_sim_data[:, :, 3].reshape(-1) * (u.km/u.s)
    vy = batch_sim_data[:, :, 4].reshape(-1) * (u.km/u.s)
    vz = batch_sim_data[:, :, 5].reshape(-1) * (u.km/u.s)

    gc = Galactocentric(x=x, y=y, z=z, v_x=vx, v_y=vy, v_z=vz)
    c = gc.transform_to(ICRS())

    ra = c.ra.to(u.degree).value
    dec = c.dec.to(u.degree).value
    distance = c.distance.to(u.kpc).value
    pm_ra_cosdec = c.pm_ra_cosdec.to(u.mas/u.yr).value
    pm_dec = c.pm_dec.to(u.mas/u.yr).value
    radial_velocity = c.radial_velocity.to(u.km/u.s).value

    projected_sim_data = np.vstack((ra, dec, distance, pm_ra_cosdec, pm_dec, radial_velocity)).T.reshape(batch_shape, n_particles, 6)
    return projected_sim_data

@partial(jit, static_argnames=('config', 'code_units'))
def simulate_stream(parameters_dict, config: SimulationConfig, code_units: CodeUnits, random_seed:int):
    params = SimulationParams(
        t_end = parameters_dict['t_end'][0] * u.Gyr.to(code_units.code_time),  
        Plummer_params = PlummerParams(Mtot=parameters_dict['m_progenitor'] * u.Msun.to(code_units.code_mass), 
                                      a=parameters_dict['a_progenitor'] * u.pc.to(code_units.code_length)),  
        PSP_params = PSPParams(M = parameters_dict['m_bulge'] * u.Msun.to(code_units.code_mass),
                              alpha = parameters_dict['alpha_bulge'], 
                              r_c = parameters_dict['r_bulge'] * u.kpc.to(code_units.code_length)),  
        TriaxialNFW_params = TriaxialNFWParams(Mvir = parameters_dict['m_Triaxial_halo'] * u.Msun.to(code_units.code_mass),
                                                r_s = parameters_dict['r_Triaxial_halo'] * u.kpc.to(code_units.code_length),
                                                q1 = parameters_dict['q1_Triaxial_halo'],
                                                q2 = parameters_dict['q2_Triaxial_halo'],),
        TwoPowerTriaxial_params = TwoPowerTriaxialParams(rho = parameters_dict['rho_TwoPowerTriaxial_halo']* (u.Msun / u.kpc**3).to(code_units.code_mass / code_units.code_length**3),
                                                        a = parameters_dict['a_TwoPowerTriaxial_halo'] * u.kpc.to(code_units.code_length),
                                                        b = parameters_dict['b_TwoPowerTriaxial_halo'] , 
                                                        c = parameters_dict['c_TwoPowerTriaxial_halo'],
                                                        alpha = parameters_dict['alpha_TwoPowerTriaxial_halo'],
                                                        beta = parameters_dict['beta_TwoPowerTriaxial_halo'],),
        ThinMN3Disk_params= ThinMN3DiskParams(M = (4 * jnp.pi * parameters_dict['rho_thin_disk'] * parameters_dict['hr_thin_disk']**2 * parameters_dict['hz_thin_disk']) * u.Msun.to(code_units.code_mass),
                                              hr = parameters_dict['hr_thin_disk'] * u.kpc.to(code_units.code_length),
                                              hz = parameters_dict['hz_thin_disk'] * u.kpc.to(code_units.code_length),),
        ThickMN3Disk_params= ThickMN3DiskParams(M = (4 * jnp.pi * parameters_dict['rho_thick_disk'] * parameters_dict['hr_thick_disk']**2 * parameters_dict['hz_thick_disk']) * u.Msun.to(code_units.code_mass),
                                                hr = parameters_dict['hr_thick_disk'] * u.kpc.to(code_units.code_length),
                                                hz = parameters_dict['hz_thick_disk'] * u.kpc.to(code_units.code_length),),
        G=code_units.G, ) 
    

    #the center of mass needs to be integrated backwards in time first 
    config_com = config._replace(N_particles=1,)
    params_com = params._replace(t_end=-params.t_end,)

    #this is the final position of the cluster, we need to integrate backwards in time 
    pos_com_final = jnp.array([[parameters_dict['x'].squeeze(), parameters_dict['y'].squeeze(), parameters_dict['z'].squeeze()]]) * u.kpc.to(code_units.code_length)
    vel_com_final = jnp.array([[parameters_dict['vx'].squeeze(), parameters_dict['vy'].squeeze(), parameters_dict['vz'].squeeze()]]) * (u.km/u.s).to(code_units.code_velocity)
    mass_com = jnp.array([params_com.Plummer_params.Mtot])
    #we construmt the initial state of the com 
    initial_state_com = construct_initial_state(pos_com_final, vel_com_final,)
    #we run the simulation backwards in time for the center of mass
    final_state_com = time_integration(initial_state_com, mass_com, config=config_com, params=params_com)
    #we calculate the final position and velocity of the center of mass
    pos_com = final_state_com[:, 0]
    vel_com = final_state_com[:, 1]
    
    key = random.PRNGKey(random_seed)
    #set up the particles in the initial state
    positions, velocities, mass = Plummer_sphere(key=key, params=params, config=config)
    #we add the center of mass position and velocity to the Plummer sphere particles
    positions = positions + pos_com
    velocities = velocities + vel_com
    #initialize the initial state
    initial_state_stream = construct_initial_state(positions, velocities, )
    #run the simulation
    stream = time_integration(initial_state_stream, mass, config=config, params=params) #this is in galactic coordinates and code units

    stream = stream.at[:, 0].set(stream[:, 0] * code_units.code_length.to(u.kpc))  # all the positions in kpc
    stream = stream.at[:, 1].set(stream[:, 1] * code_units.code_velocity.to(u.km/u.s))  # all the velocities in km/s
    
    return stream.reshape(-1, 6)


