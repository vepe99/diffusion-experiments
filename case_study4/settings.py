from pathlib import Path

import numpy as np
from tqdm import tqdm


N_TRAINING_BATCHES = 256
BATCH_SIZE = 128
EPOCHS = 1000
N_TRIALS = 30
N_SUBJECTS = 100
N_SAMPLES = 100
N_TEST = 100
METHOD = 'two_step_adaptive'
STEPS = "adaptive"
MAX_STEP = 1_000
BASE = Path(__file__).resolve().parent


def sample_in_batches(data, workflow, num_samples, batch_size, sampler_settings=None) -> dict:
    posterior_samples = None
    for i in tqdm(range(0, len(data['sim_data']), batch_size)):
        batch_data = {k: v[i:i + batch_size] for k, v in data.items()}
        if sampler_settings is None:
            batch_samples = workflow.sample(conditions=batch_data, num_samples=num_samples)
        else:
            batch_samples = workflow.sample(conditions=batch_data, num_samples=num_samples,
                                            **sampler_settings)
        if i == 0:
            posterior_samples = batch_samples
        else:
            for key in posterior_samples.keys():
                posterior_samples[key] = np.vstack([posterior_samples[key], batch_samples[key]])
    return posterior_samples
