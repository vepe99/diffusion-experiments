from pathlib import Path

import numpy as np
from tqdm import tqdm


N_TRAINING_BATCHES = 256
BATCH_SIZE = 128
EPOCHS = 2
N_TRIALS = 30
N_SUBJECTS = 2
N_SAMPLES = 100
N_TEST = 50
METHOD = 'two_step_adaptive'
STEPS = "adaptive"
MAX_STEP = 1_000
BASE = Path(__file__).resolve().parent