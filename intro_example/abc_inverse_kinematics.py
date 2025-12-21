import os
import tempfile
import logging
from pathlib import Path

import numpy as np

import pyabc
from bayesflow.simulators import InverseKinematics

logging.getLogger("bayesflow").setLevel(logging.ERROR)

BASE = Path(__file__).resolve().parent
MIN_EPSILON = 0.002
NUM_SAMPLES_INFERENCE = 1000

if __name__ == "__main__":

    simulator = InverseKinematics()

    obs = {"observables": np.array([0, 1.5])}

    prior = pyabc.Distribution(
        p1=pyabc.RV("norm", 0, 0.25), 
        p2=pyabc.RV("norm", 0, 0.5), 
        p3=pyabc.RV("norm", 0, 0.5), 
        p4=pyabc.RV("norm", 0, 0.5)
    )

    def model(p):
        p = np.array([p["p1"], p["p2"], p["p3"], p["p4"]])
        return {"observables": simulator.observation_model(p)}

    abc = pyabc.ABCSMC(model, prior, population_size=NUM_SAMPLES_INFERENCE)

    db_path = os.path.join(tempfile.gettempdir(), "test.db")
    abc.new("sqlite:///" + db_path, obs)
    history = abc.run(minimum_epsilon=MIN_EPSILON)

    abc_samples_weighted, weights = history.get_distribution()
    abc_samples = pyabc.resample(abc_samples_weighted.values, weights, n=NUM_SAMPLES_INFERENCE)
    np.save(BASE / "models" / "abc_inverse_kinematics.npy", abc_samples)
    logging.info(f"ABC-SMC inference completed. Samples saved.")
