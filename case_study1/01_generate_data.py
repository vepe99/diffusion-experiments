import pickle

from bayesflow.simulators import TwoMoons, InverseKinematics


NUM_TRAIN = 2048
NUM_VALIDATION = 512
NUM_TEST = 512


if __name__ == '__main__':

    # Two moons low-dimensional benchmark
    two_moons_simulator = TwoMoons()

    two_moons_train_data = two_moons_simulator.sample(NUM_TRAIN)
    two_moons_validation_data = two_moons_simulator.sample(NUM_VALIDATION)

    with open('./data/two_moons_train_data.pkl', 'wb') as f:
        pickle.dump(two_moons_train_data, f)
    with open('./data/two_moons_validation_data.pkl', 'wb') as f:
        pickle.dump(two_moons_validation_data, f)

    # Inverse kinematics low-dimensional benchmark
    inverse_kinematics_simulator = InverseKinematics()

    inverse_kinematics_train_data = inverse_kinematics_simulator.sample(NUM_TRAIN)
    inverse_kinematics_validation_data = inverse_kinematics_simulator.sample(NUM_VALIDATION)

    with open('./data/inverse_kinematics_train_data.pkl', 'wb') as f:
        pickle.dump(inverse_kinematics_train_data, f)
    with open('./data/inverse_kinematics_validation_data.pkl', 'wb') as f:
        pickle.dump(inverse_kinematics_validation_data, f)
