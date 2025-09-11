import sys
import numpy as np
import pickle

problem_name = sys.argv[1] if len(sys.argv) > 1 else "Boehm_JProteomeRes2014"
print(problem_name)

storage = f'plots/{problem_name}/'
all_samples = []
for i in range(100):
    mcmc_path = f'{storage}mcmc_samples_{problem_name}_{i}.pkl'
    try:
        with open(mcmc_path, 'rb') as f:
            mcmc_samples = pickle.load(f)
        all_samples.append(mcmc_samples)
    except FileNotFoundError:
        print(f"File not found: {mcmc_path}")

all_samples = np.array(all_samples)
print(all_samples.shape)

# pickle the combined samples
with open(f'{storage}mcmc_samples_{problem_name}.pkl', 'wb') as f:
    pickle.dump(all_samples, f)
