import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactocentric, ICRS
from functools import partial 



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
    output['j'] = np.tile(np.array(possible_j)[np.newaxis, :, np.newaxis], (n_samples, 1, 1)) #j has shape (n_samples, n_streams, 1) and each column is a different stream, we will use this to sample the local parameters in a vectorized way

    # Vectorized sampling for local parameters
    stream_keys = {v: k for k, v in target_streams.items()} #this gets the string of the stream given the j value, we will use this to get the correct local parameters for each stream
    for key in local_param_keys:
        output[key] = np.zeros((n_samples, len(possible_j), 1))
        for n in range(n_samples):
            for j in range(len(possible_j)):
                output[key][n, j] = get_prior_sample(prior_local_dict[stream_keys[j]][key], size=1)[0]
        if key in ['ra', 'dec', 'r', 'mu_ra_cosdec', 'mu_dec', 'vr']:
            output[key] = output[key].reshape(-1, 1) #flatten to use astropy
        
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
    for key in ['ra', 'dec', 'r', 'mu_ra_cosdec', 'mu_dec', 'vr']:
        output[key] = output[key].reshape(n_samples, len(possible_j), 1) #reshape back to (n_samples, n_streams, 1) for the rest of the code
    gc = c.transform_to(Galactocentric)
    output['x'] = gc.x.to(u.kpc).value.reshape(-1, len(possible_j), 1)
    output['y'] = gc.y.to(u.kpc).value.reshape(-1, len(possible_j), 1)
    output['z'] = gc.z.to(u.kpc).value.reshape(-1, len(possible_j), 1)
    output['vx'] = gc.v_x.to(u.km/u.s).value.reshape(-1, len(possible_j), 1)
    output['vy'] = gc.v_y.to(u.km/u.s).value.reshape(-1, len(possible_j), 1)
    output['vz'] = gc.v_z.to(u.km/u.s).value.reshape(-1, len(possible_j), 1)

    return output


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

