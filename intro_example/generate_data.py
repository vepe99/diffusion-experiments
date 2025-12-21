import pickle
from pathlib import Path

from bayesflow.simulators import InverseKinematics


NUM_TRAIN = 10_000
NUM_VALIDATION = 512
BASE = Path(__file__).resolve().parent

if __name__ == '__main__':

    # Inverse kinematics low-dimensional benchmark
    inverse_kinematics_simulator = InverseKinematics()

    inverse_kinematics_train_data = inverse_kinematics_simulator.sample((NUM_TRAIN,))
    inverse_kinematics_validation_data = inverse_kinematics_simulator.sample((NUM_VALIDATION,))

    with open(BASE / 'models' / 'inverse_kinematics_train_data.pkl', 'wb') as f:
        pickle.dump(inverse_kinematics_train_data, f)
    with open(BASE / 'models' / 'inverse_kinematics_validation_data.pkl', 'wb') as f:
        pickle.dump(inverse_kinematics_validation_data, f)
