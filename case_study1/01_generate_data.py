import pickle

import bayesflow as bf

NUM_TRAIN = 4096
NUM_VALIDATION = 512
NUM_TEST = 512


if __name__ == '__main__':

    # Two moons benchmark
    simulator = bf.simulators.TwoMoons()

    two_moons_train_data = simulator.sample(NUM_TRAIN)
    two_moons_validation_data = simulator.sample(NUM_VALIDATION)
    two_moons_test_data = simulator.sample(NUM_TEST)

    with open('./data/two_moons_train_data.pkl', 'wb') as f:
        pickle.dump(two_moons_train_data, f)
    with open('./data/two_moons_validation_data.pkl', 'wb') as f:
        pickle.dump(two_moons_validation_data, f)
    with open('data/two_moons_test_data.pkl', 'wb') as f:
        pickle.dump(two_moons_test_data, f)
