import os
import tempfile

import numpy as np

import pyabc

from bayesflow.simulators import InverseKinematics


MIN_EPSILON = 0.001

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

    def distance(x, x0):
        return np.linalg.norm(x["observables"] - x0["observables"])

    abc = pyabc.ABCSMC(model, prior, distance, population_size=1000)

    db_path = os.path.join(tempfile.gettempdir(), "test.db")
    abc.new("sqlite:///" + db_path, obs)

    history = abc.run(minimum_epsilon=MIN_EPSILON, max_nr_populations=50)

    abc_samples = history.get_distribution()[0].values

    np.save("samples/abc_inverse_kinematics.npy", abc_samples)
