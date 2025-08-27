import pickle

from bayesflow.simulators import TwoMoons

NUM_TRAIN = 4096
NUM_VALIDATION = 512
NUM_TEST = 512


if __name__ == '__main__':

    # Two moons benchmark
    simulator = TwoMoons()

    two_moons_train_data = simulator.sample(NUM_TRAIN)
    two_moons_validation_data = simulator.sample(NUM_VALIDATION)

    with open('./data/two_moons_train_data.pkl', 'wb') as f:
        pickle.dump(two_moons_train_data, f)
    with open('./data/two_moons_validation_data.pkl', 'wb') as f:
        pickle.dump(two_moons_validation_data, f)
