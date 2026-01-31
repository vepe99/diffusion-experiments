

import numpy as np

def get_prior_sample(paramter_prior_dict: dict):
    if paramter_prior_dict['type'] == 'uniform':
        low, high = paramter_prior_dict['prior_parameters']
        return np.random.uniform(low, high)
    elif paramter_prior_dict['type'] == 'normal':
        mean, std = paramter_prior_dict['prior_parameters']
        return np.random.normal(mean, std)
    elif paramter_prior_dict['type'] == 'identity':
        return paramter_prior_dict['prior_parameters'][0]

def get_prior_local_sample(prior_local_dict, stream_key, samples):
    paramter_prog_dict = prior_local_dict[stream_key]
    for key in paramter_prog_dict.keys():
        samples[key] = get_prior_sample(paramter_prog_dict[key])
    return samples

def sample_parameters(prior_global_dict: dict, prior_local_dict: dict, n_samples: int, target_streams, key_seed=0):
    np.random.seed(key_seed)
    samples_list = []
    global_keys = list(prior_global_dict.keys())
    possible_j = list(target_streams.values())
    stream_keys = list(target_streams.keys())

    for _ in range(n_samples):
        sample = {}
        # Sample global parameters
        for key in global_keys:
            sample[key] = get_prior_sample(prior_global_dict[key])
        # Sample local parameters
        j = np.random.choice(possible_j)  # j is the integer index
        # Find the stream key whose value matches j
        stream_key = next(k for k, v in target_streams.items() if v == j)
        sample['j'] = j  # Use integer index instead of stream name
        sample = get_prior_local_sample(prior_local_dict, stream_key, sample)
        samples_list.append(sample)
    return samples_list