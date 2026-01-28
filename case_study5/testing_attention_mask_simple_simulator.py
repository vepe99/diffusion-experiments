from autocvd import autocvd
autocvd(num_gpus = 1)

backend = "torch"
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = backend

import bayesflow as bf
import numpy as np

n_training = 10             #size of training set
n_trials = 20               #size of the order invariant input of the SetTransformer is (n_trials, 1)
n_test = 13                 #size of test set
n_subject_test = 5          #N_compositional_conditions
n_samples_posterior = 24
check_compositional_sampling = False
check_modified_compositional_sampling = True
check_normal_sampling = False    

def score_log_norm(x, m, s):
    return -(x-m) / s**2

def simulate_(mu, sigma, n_subjects=1, n_trials=1):
    if isinstance(mu, (float, int)):
        mu = np.ones((n_subjects,)) * mu
        sigma = np.ones((n_subjects,)) * sigma
    data = np.zeros((n_subjects, n_trials, 1))
    for j_subject in range(n_subjects):
        for i_trial in range(n_trials):
            data[j_subject, i_trial] = np.random.normal(loc=mu[j_subject], scale=sigma[j_subject])
    if n_subjects == 1 and n_trials == 1:
        data = data[0, 0]
    elif n_subjects == 1:
        data = data[0]
    elif n_trials == 1:
        data = data[:, 0]
    return dict(sim_data=data)

def sample_hierarchical_priors(n_subjects=1):

    #Group level
    mean_mu = np.random.normal(-10, 10)
    mean_sigma = np.random.normal(5, 0,1)

    #Subject level
    mu = np.random.normal(loc=mean_mu, scale=1.0, size=n_subjects)
    sigma = np.random.normal(loc=mean_sigma, scale=0.1, size=n_subjects)
    return dict(mean_mu=mean_mu, 
                mean_sigma=mean_sigma, 
                mu=mu, 
                sigma=sigma)


def prior_global_score(x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mean_mu = x["mean_mu"]
    mean_sigma = x["mean_sigma"]
    parts = {
        "mean_mu": score_log_norm(mean_mu, m=0.0, s=10.0),
        "mean_sigma": score_log_norm(mean_sigma, m=5.0, s=1.0),
    }
    return parts


param_names_global = ["mean_mu", "mean_sigma"]
adapter = (
    bf.adapters.Adapter()
    .to_array()
    .convert_dtype("float64", "float32")
    .concatenate(param_names_global, into="inference_variables")
    .rename("sim_data", "summary_variables")
)
workflow_global = bf.BasicWorkflow(
    adapter=adapter,
    summary_network=bf.networks.SetTransformer(summary_dim=16, dropout=0.1),
    inference_network=bf.networks.CompositionalDiffusionModel(),
)


simulator_hierarchical = bf.simulators.make_simulator([sample_hierarchical_priors, simulate_])
training_data = simulator_hierarchical.sample((n_training), n_trials=n_trials)

#let's create a random mask, the SetTrasnformer will be trained with a variable number of permutationally invariant observation
min_valid_fraction = 0.5    #at least 50% of the observation are valid
max_valid_fraction = 1.0    #at most 100% of the observation are valid
attention_mask_training = np.zeros((n_training, n_trials, 1), dtype=np.float32)
for i in range(n_training):
    # Random fraction of valid particles for this sample
    valid_fraction = np.random.uniform(min_valid_fraction, max_valid_fraction)
    n_valid = int(n_trials * valid_fraction)
    # Randomly select which particles are valid
    valid_indices = np.random.choice(n_trials, size=n_valid, replace=False)
    attention_mask_training[i, valid_indices, 0] = 1.0
print('attention mask training shape', attention_mask_training.shape)

history = workflow_global.fit_offline(
        training_data,
        epochs=2,
        batch_size=3,
        verbose=2,
        kwargs={'attention_mask': attention_mask_training},
    )

#TESTING USING MORE SUBJECTS=N_compositional_conditions
test_data = simulator_hierarchical.sample(n_test, n_subjects=n_subject_test, n_trials=n_trials, )
print('test data shape', test_data['sim_data'].shape)

#same as for the training set, we are going to use a mask for the SetTransformer
# Create mask with shape (N_TEST, N_SUBJECTS, N_PARTICLES) first
# attention_mask_3d = np.zeros((n_test, n_subject_test, n_trials), dtype=np.float32)

# for i in range(n_test):
#     for j in range(n_subject_test):
#         valid_fraction = np.random.uniform(min_valid_fraction, max_valid_fraction)
#         n_valid = int(n_trials * valid_fraction)
#         valid_indices = np.random.choice(n_trials, size=n_valid, replace=False)
#         attention_mask_3d[i, j, valid_indices] = 1.0
# # Flatten to (N_TEST * N_SUBJECTS, N_PARTICLES). The sim_data is flattened internally so we need to do the same for the attention mask
# attention_mask_test = attention_mask_3d.reshape(n_test * n_subject_test, n_trials)


B = n_test * n_subject_test
T = 1
S = n_trials 
attention_mask_test = np.zeros((B, T, S), dtype=np.bool_)

for b in range(B):
    for t in range(T):
        valid_fraction = np.random.uniform(min_valid_fraction, max_valid_fraction)
        n_valid = int(S * valid_fraction)
        valid_idx = np.random.choice(S, n_valid, replace=False)
        attention_mask_test[b, t, valid_idx] = True

if check_compositional_sampling:

    global_posterior = workflow_global.compositional_sample(
        num_samples=n_samples_posterior,
        conditions={'sim_data': test_data['sim_data']},
        compute_prior_score=prior_global_score,
        compositional_bridge_d1=1/n_subject_test,
        mini_batch_size=n_subject_test,
        method='two_step_adaptive',
        steps='adaptive',
        max_steps=1000,
        attention_mask=attention_mask_test,
    )

elif check_modified_compositional_sampling:
    # Possible step in the right direction to avoid mixing tensor and non tensor kwargs as suggest by @Jonas aruda
    workflow_global.approximator.inference_network.integrate_kwargs.update({
    'method': 'two_step_adaptive',
    'steps': 'adaptive',
    'compositional_bridge_d1': 1/n_subject_test,
    'mini_batch_size': n_subject_test,
    })

    global_posterior = workflow_global.compositional_sample(
        num_samples=n_samples_posterior,
        conditions={'sim_data': test_data['sim_data']},
        compute_prior_score=prior_global_score,
        attention_mask =attention_mask_test,
    )

elif check_normal_sampling:
    print('Normal sampling')
    global_posterior = workflow_global.sample(
        num_samples=n_samples_posterior,
        conditions={'sim_data': test_data['sim_data'].reshape(n_test * n_subject_test, n_trials, 1)},
        attention_mask=attention_mask_test,
        mini_batch_size=n_subject_test,
    )


print('Global posterior samples:')
print('mean_mu:', global_posterior['mean_mu'].shape)
print('mean_sigma:', global_posterior['mean_sigma'].shape)