import numpy as np
import json

def get_priorscore_from_simconfig(params_name, path_to_config):
    pass

def load_npz_as_dict(file):
    with np.load(file) as data:
        return dict(data)