import bayesflow as bf

EPOCHS = 1000
BATCH_SIZE = 128
NUM_SAMPLES_INFERENCE = 3000
MODELS = {
        "flow_matching": (bf.networks.FlowMatching, {}),
        "ot_flow_matching": (bf.networks.FlowMatching, {"use_optimal_transport": True}),
        "consistency_model": (bf.networks.ConsistencyModel, {"total_steps": EPOCHS*BATCH_SIZE}),
        "stable_consistency_model": (bf.experimental.StableConsistencyModel, {"embedding_kwargs": {"embed_dim": 2}}),
        "diffusion_edm_vp": (bf.networks.DiffusionModel, {
            "noise_schedule": "edm",
            "prediction_type": "F",
            "schedule_kwargs": {"variance_type": "preserving"}}),
        "diffusion_edm_ve": (bf.networks.DiffusionModel, {
            "noise_schedule": "edm",
            "prediction_type": "F",
            "schedule_kwargs": {"variance_type": "exploding"}}),
        "diffusion_cosine_F": (bf.networks.DiffusionModel, {
            "noise_schedule": "cosine",
            "prediction_type": "F", }),
        "diffusion_cosine_v": (bf.networks.DiffusionModel, {
            "noise_schedule": "cosine",
            "prediction_type": "velocity"}),
        "diffusion_cosine_noise": (bf.networks.DiffusionModel, {
            "noise_schedule": "cosine",
            "prediction_type": "noise"}),
    }


SAMPLER_SETTINGS = {
    'ode': {
        'method': 'rk45',
        'steps': 100
    },
    'sde': {
        'method': 'euler_maruyama',
        'steps': 100
    },
    'sde-pc': {
        'method': 'euler_maruyama',
        'steps': 100,
        'corrector_steps': 1,
    }
}