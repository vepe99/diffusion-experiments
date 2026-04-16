from autocvd import autocvd
# autocvd(num_gpus = 1)
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import yaml
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.core.config_store import ConfigStore
import numpy as np

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"
import keras
import bayesflow as bf
from scipy import  special 
import jax.numpy as jnp 

import logging
logging.getLogger('bayesflow').setLevel(logging.DEBUG)

from config.EvalConfig import EvalConfig
from utils.utils_train_jax_new import AugmentationsClass #we will need to use the augmentations on the test_set



import astropy.units as u
import jax.numpy as jnp
import jax

from odisseo.dynamics import DIRECT_ACC_MATRIX
from odisseo.potentials import combined_external_acceleration_vmpa_switch
from odisseo.option_classes import SimulationConfig, SimulationParams, PlummerParams, PSPParams, TriaxialNFWParams,ThickMN3DiskParams, ThinMN3DiskParams 
from odisseo.option_classes import PSP_POTENTIAL, TRIAXIAL_NFW_POTENTIAL, THICK_MN3_DISK, THIN_MN3_DISK
from odisseo.units import CodeUnits




cs = ConfigStore.instance()
cs.store(name="eval_config", node=EvalConfig)



def _patch_build_config_batch_size(obj, new_batch_size=2):
    """Recursively walk a deserialized config dict and replace the first
    element (batch size) of any 'input_shape' list inside a 'build_config'
    with `new_batch_size`, so that the model can be loaded on a smaller GPU."""
    if isinstance(obj, dict):
        if "build_config" in obj and isinstance(obj["build_config"], dict):
            bc = obj["build_config"]
            if "input_shape" in bc and isinstance(bc["input_shape"], list):
                shape = bc["input_shape"]
                if len(shape) >= 1 and isinstance(shape[0], int) and shape[0] > new_batch_size:
                    shape[0] = new_batch_size
        for v in obj.values():
            _patch_build_config_batch_size(v, new_batch_size)
    elif isinstance(obj, list):
        for item in obj:
            _patch_build_config_batch_size(item, new_batch_size)


def fix_keras_model(model_path, ):
    import zipfile
    import json
    fixed_model_path = model_path.replace('.keras', '_fixed.keras')
    if not os.path.exists(fixed_model_path):
        with zipfile.ZipFile(model_path, 'r') as zin:
            with zipfile.ZipFile(fixed_model_path, 'w') as zout:
                for item in zin.infolist():
                    data = zin.read(item.filename)
                    if item.filename == 'config.json':
                        config_str = data.decode('utf-8')
                        config_str = config_str.replace(
                            '__bayesflow_type__ArrayImpl',
                            '__bayesflow_type__ndarray'
                        )
                        data = config_str.encode('utf-8')
                    zout.writestr(item, data)
        print(f"Created fixed model at {fixed_model_path}")
    return fixed_model_path


def disk_masses_from_params(p):
    """
    Compute total masses of thin and thick disks from parameter dict.

    Args:
        p: dict of JAX arrays (same structure as parameters_dict)

    Returns:
        dict with:
            - M_thin_disk
            - M_thick_disk
    """

    def s(x):
        return jnp.squeeze(x)  # ensure scalar for clean autodiff

    # Extract parameters
    rho_thin = s(p['rho_thin_disk']) * (u.Msun / u.pc**3).to(u.Msun / u.kpc**3)
    hr_thin  = s(p['hr_thin_disk'])
    hz_thin  = s(p['hz_thin_disk'])

    rho_thick = s(p['rho_thick_disk']) * (u.Msun / u.pc**3).to(u.Msun / u.kpc**3)
    hr_thick  = s(p['hr_thick_disk'])
    hz_thick  = s(p['hz_thick_disk'])

    # Mass formula
    M_thin  = 4 * jnp.pi * rho_thin  * hr_thin**2  * hz_thin
    M_thick = 4 * jnp.pi * rho_thick * hr_thick**2 * hz_thick

    return {
        "M_thin_disk": M_thin,
        "M_thick_disk": M_thick,
    }

code_length = 1 * u.kpc
code_mass = 1 * u.Msun
G = 1
code_time = 1 * u.Myr
code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  

config = SimulationConfig(N_particles = 1000, 
                          return_snapshots = True, 
                          num_snapshots = 1000, 
                          num_timesteps = 1000, 
                          external_accelerations=(TRIAXIAL_NFW_POTENTIAL, THICK_MN3_DISK, THIN_MN3_DISK, PSP_POTENTIAL), 
                          acceleration_scheme = DIRECT_ACC_MATRIX,
                          softening = (0.1 * u.pc).to(code_units.code_length).value,) #default values


# Numeric conversion factors (Python floats, JAX-safe in traced code)
MSUN_TO_CODE_MASS = float(u.Msun.to(code_units.code_mass))
KPC_TO_CODE_LENGTH = float(u.kpc.to(code_units.code_length))
CODE_VEL_TO_KMS = float(code_units.code_velocity.to(u.km / u.s))


def s(x):
    """Squeeze array params to scalars for JAX grad compatibility."""
    return jnp.squeeze(x)
# Fix 1: never mutate the input dict — use fixed bulge constants separately
BULGE_M     = 4501365375.06545 * MSUN_TO_CODE_MASS
BULGE_ALPHA = 1.8
BULGE_RC    = 1.9 * KPC_TO_CODE_LENGTH

def construct_params_from_dict(p):
    """Build SimulationParams from a flat dict of JAX scalars. Does NOT mutate p."""
    disk_masses = disk_masses_from_params(p)
    return SimulationParams(
        t_end = (4 * u.Gyr).to(code_units.code_time).value,
        Plummer_params= PlummerParams(
            Mtot=(2.5e4 * u.Msun).to(code_units.code_mass).value,
            a=(8 * u.pc).to(code_units.code_length).value
        ),
        PSP_params= PSPParams(
            M     = BULGE_M,        # fixed constant, not from p
            alpha = BULGE_ALPHA,
            r_c   = BULGE_RC,
        ),
        TriaxialNFW_params= TriaxialNFWParams(
            Mvir = s(p['m_Triaxial_halo']) * MSUN_TO_CODE_MASS,
            r_s  = s(p['r_Triaxial_halo']) * KPC_TO_CODE_LENGTH,
            q1   = 1.0,
            q2   = s(p['q2_Triaxial_halo'])
        ),
        ThinMN3Disk_params= ThinMN3DiskParams(
            M  = s(disk_masses['M_thin_disk']) * MSUN_TO_CODE_MASS,
            hr = s(p['hr_thin_disk']) * KPC_TO_CODE_LENGTH,
            hz = s(p['hz_thin_disk']) * KPC_TO_CODE_LENGTH
        ),
        ThickMN3Disk_params= ThickMN3DiskParams(
            M  = s(disk_masses['M_thick_disk']) * MSUN_TO_CODE_MASS,
            hr = s(p['hr_thick_disk']) * KPC_TO_CODE_LENGTH,
            hz = s(p['hz_thick_disk']) * KPC_TO_CODE_LENGTH
        ),
        G=code_units.G,
    )


@partial(jax.jit, static_argnames=['config'])
def circular_velocity_at_xyz(xyz: jnp.ndarray,
                              config: SimulationConfig,
                              p: dict) -> jnp.ndarray:
    """
    Compute the local circular velocity at arbitrary (x, y, z) positions.

    Uses the general formula:
        v_circ = sqrt(r * |dPhi/dr|) = sqrt(r * |∇Φ · r̂|)

    where r = ||xyz|| and r̂ = xyz / r.

    Args:
        xyz: array of shape (N, 3) — positions in code units
        config: SimulationConfig (static)
        params: SimulationParams (differentiable)

    Returns:
        v_circ: array of shape (N,)
    """
    params = construct_params_from_dict(p)
    xyz = jnp.atleast_2d(xyz)           # (N, 3)
    n = xyz.shape[0]

    # Build state (N, 2, 3) with zero velocities
    state = jnp.stack([xyz, jnp.zeros_like(xyz)], axis=1)

    # acc = -∇Φ, shape (N, 3)
    acc = combined_external_acceleration_vmpa_switch(state, config, params, return_potential=False)

    # r and r̂
    r = jnp.linalg.norm(xyz, axis=-1)          # (N,)
    r_hat = xyz / r[:, None]                    # (N, 3)

    # dPhi/dr = ∇Φ · r̂ = -acc · r̂
    dPhi_dr = -jnp.sum(acc * r_hat, axis=-1)   # (N,)

    # return jnp.sqrt(r * jnp.abs(dPhi_dr)) * CODE_VEL_TO_KMS
    return jnp.sqrt(r * jnp.abs(dPhi_dr)) 


@partial(jax.jit, static_argnames=['config', 'func'])
def vcirc_func(xyz: jnp.ndarray,
               config: SimulationConfig,
               p: dict,
               func=lambda vc: vc) -> jnp.ndarray:
    """
    Evaluate an arbitrary scalar function of the circular velocity at positions xyz.

    Args:
        xyz: array of shape (N, 3) — positions in code units
        config: static config
        params: differentiable params
        func: callable applied to v_circ array — should return a scalar for grad

    Returns:
        func(v_circ(xyz))
    """
    vc = circular_velocity_at_xyz(xyz, config, p) 
    return func(vc)

@hydra.main(version_base=None, config_path="config", config_name="eval_config_new_constrain",)
def main(cfg: EvalConfig):

    print(cfg)
    print("##############")
    model_path = os.path.join(cfg.base_dir, cfg.model_dir, 'global_model.keras' )
    # Fix ArrayImpl serialization issue in the .keras fil
    model_path = fix_keras_model(model_path, )
    import zipfile
    import json
    fixed_model_path = model_path.replace('.keras', '_fixed.keras')
    if not os.path.exists(fixed_model_path):
        with zipfile.ZipFile(model_path, 'r') as zin:
            with zipfile.ZipFile(fixed_model_path, 'w') as zout:
                for item in zin.infolist():
                    data = zin.read(item.filename)
                    if item.filename == 'config.json':
                        config_str = data.decode('utf-8')
                        config_str = config_str.replace(
                            '__bayesflow_type__ArrayImpl',
                            '__bayesflow_type__ndarray'
                        )
                        data = config_str.encode('utf-8')
                    zout.writestr(item, data)
        print(f"Created fixed model at {fixed_model_path}")
    model_path = fixed_model_path
    print('Loading model from ', model_path)
    print("##############")
    param_names_global = list(cfg.parameters_global)
    sim_data = str(cfg.sim_data)
    inference_conditions = str(cfg.inference_conditions[0])
    test_data_path = os.path.join(cfg.base_dir, cfg.data_dir, f'simulation_multistream_{cfg.multistream_n_simulation}.npz')
    print('Loading test data from ', test_data_path)
    test_data = dict(np.load(test_data_path, allow_pickle=True))
    keys_to_drop = set(test_data.keys()) - set(param_names_global) - {sim_data} - set(inference_conditions)
    keys_to_drop = list(keys_to_drop) 
    with open(os.path.join(cfg.base_dir, cfg.model_dir, '.hydra', 'config.yaml'), "r") as f:
        model_config = yaml.safe_load(f)
    print(model_config)


    adapter = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .drop(keys_to_drop)
        .concatenate(param_names_global, into="inference_variables")
        .rename(sim_data, "summary_variables")
        .rename(inference_conditions, "inference_conditions")
    )
    if cfg.noise_schedule is not None:
        inference_network = bf.networks.CompositionalDiffusionModel(
                                                        subnet_kwargs={
                                                        "widths": [model_config['global_model']['inference_mlp_width']] * model_config['global_model']['inference_mlp_depth'],
                                                        "time_embedding_dim": model_config['global_model']['inference_time_embedding_dim'],
                                                        },
                                                        schedule_kwargs = {**cfg.noise_schedule,},
                                                        )
    else:
        #probably needs to fix it to the training noise schedule 
        inference_network = bf.networks.CompositionalDiffusionModel(subnet_kwargs={
                                                        "widths": [model_config['global_model']['inference_mlp_width']] * model_config['global_model']['inference_mlp_depth'],
                                                        "time_embedding_dim": model_config['global_model']['inference_time_embedding_dim'],
                                                        },)
    

    workflow_global = bf.BasicWorkflow(
        adapter=adapter,
        summary_network=bf.networks.SetTransformer(summary_dim=model_config['global_model']['summary_dim'], 
                                                   num_heads=(model_config['global_model']['num_heads'],model_config['global_model']['num_heads'],),
                                                   embed_dims = (model_config['global_model']['embed_dims'], model_config['global_model']['embed_dims'],),
                                                   mlp_depths=(model_config['global_model']['mlp_depths'], model_config['global_model']['mlp_depths']),
                                                   mlp_widths=(model_config['global_model']['mlp_widths'], model_config['global_model']['mlp_widths']),
                                                   dropout=0.1),
        # inference_network = bf.networks.CompositionalDiffusionModel(subnet_kwargs={
        #                                                 "widths": [model_config['global_model']['inference_mlp_width']] * model_config['global_model']['inference_mlp_depth'],
        #                                                 "time_embedding_dim": model_config['global_model']['inference_time_embedding_dim'],
        #                                                 },),
        inference_network = inference_network,
        standardize=["inference_variables", "summary_variables"]
    )
    workflow_global.approximator = keras.models.load_model(model_path)

    test_data = {k: test_data[k] for k in cfg.parameters_global + [cfg.sim_data, "j"] }
    # Augmentation
    augmentations_class = AugmentationsClass(cfg)
    augmentations = []
    # --- Coordinate transforms (must be first, before any masking) ---
    if "remove_los_velocity" in cfg.augmentations:
        augmentations.append(augmentations_class.remove_los_velocity)
    if "convert_distance_to_parallax" in cfg.augmentations:
        augmentations.append(augmentations_class.convert_distance_to_parallax)

    # --- Observational selection (window → subsample → compact) ---
    if "observational_window" in cfg.augmentations:
        augmentations.append(augmentations_class.observational_window)
    if "observed_n_stars" in cfg.augmentations:
        augmentations.append(augmentations_class.subsampling_to_observed_n_stars)
    if "compact_to_attended" in cfg.augmentations:
        augmentations.append(augmentations_class.compact_to_attended)

    # --- Photometric augmentation (magnitudes → errors → apply) ---
    if "sample_magnitudes" in cfg.augmentations:
        augmentations.append(augmentations_class.sample_magnitudes)
    if "sample_obs_error" in cfg.augmentations:
        augmentations.append(augmentations_class.sample_obs_error)
    if "apply_obs_error" in cfg.augmentations:
        augmentations.append(augmentations_class.apply_obs_error)

    # --- v_los masking (must be after apply_obs_error) ---
    if "mask_vlos" in cfg.augmentations:
        augmentations.append(augmentations_class.mask_vlos)

    # --- Symmetry augmentations ---
    if "flip_dirz" in cfg.augmentations:
        augmentations.append(augmentations_class.flip_dirz)

    # --- Feature concatenations (must be last) ---
    if "concatentate_sigma_error_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatentate_sigma_error_to_sim_data)
    if "concatenate_magnitudes_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_magnitudes_to_sim_data)
    if "concatenate_vlos_mask_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_vlos_mask_to_sim_data)
    if "concatenate_j_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_j_to_sim_data)

    test_data[cfg.sim_data] = test_data[cfg.sim_data].reshape(-1, test_data[cfg.sim_data].shape[-2], test_data[cfg.sim_data].shape[-1])
    test_data['j'] = test_data['j'].reshape(-1, 1)

    radial_positions = jnp.linspace(5, 15, 38)  # (38,) kpc

    # xyz shape: (38, 3) — all positions at once, no vmap needed over positions
    xyz_batch = jnp.stack([
        radial_positions,
        jnp.zeros_like(radial_positions),
        jnp.zeros_like(radial_positions)
    ], axis=-1)  # (38, 3)

    def compute_vcirc_single_sample(params_dict):
        """
        Compute v_circ (km/s) for ONE sample across all radial positions.
        
        Args:
            params_dict: flat dict of scalar JAX arrays (one sample's parameters)
        Returns:
            v_circ: shape (38,) in km/s
        """
        # circular_velocity_at_xyz already handles (N, 3) — no vmap needed here
        return circular_velocity_at_xyz(xyz_batch, config, params_dict)

    # vmap ONLY over the sample/batch dimension
    vmap_vcirc = jax.vmap(compute_vcirc_single_sample)

    # test_data[k] shape: (N_samples,) after [:, 0] slicing
    vel_circ_true = vmap_vcirc({k: test_data[k][:, 0] for k in cfg.parameters_global})
    # shape: (N_samples, 38)

    print('Radial positions (kpc):', radial_positions)
    print('True circular velocities (km/s), shape:', vel_circ_true.shape)
    print('First sample v_circ:', vel_circ_true[0])


    print('Test data sim shape before augmentation: ', test_data[cfg.sim_data].shape)
    for aug in augmentations:
        print(f"Applying augmentation: {aug.__name__}")
        test_data = aug(test_data)
    test_data[cfg.sim_data] = test_data[cfg.sim_data].reshape(-1, len(cfg.target_streams.keys()), test_data[cfg.sim_data].shape[-2], test_data[cfg.sim_data].shape[-1])
    test_data['j'] = test_data['j'].reshape(-1,len(cfg.target_streams.keys()), 1)
    print('Test data sim shape after augmentation: ', test_data[cfg.sim_data].shape)
    print('Test data keys: ', test_data.keys())
    print('Test data attention mask shape: ', test_data['attention_mask'].shape)
    with open(os.path.join(cfg.base_dir, cfg.data_dir, '.hydra', 'config.yaml'), "r") as f:
        test_sim_config = yaml.safe_load(f)
    print('Test simulation config prior: ', test_sim_config['priors_global'])

    def prior_global_score(x, time, cfg=cfg, test_sim_config=test_sim_config):
        
        score = {}
        
        for k in cfg.parameters_global:
            # print(f"Computing prior score for {k} with type {test_sim_config['priors_global'][k]['type']}")
            if test_sim_config['priors_global'][k]['type'] == 'uniform':
                score[k] = (1-time)*jnp.zeros_like(x[k])
            elif test_sim_config['priors_global'][k]['type'] == 'normal':
                mean = test_sim_config['priors_global'][k]['prior_parameters'][0]
                std = test_sim_config['priors_global'][k]['prior_parameters'][1]
                score[k] = -(1-time)*(x[k] - mean) / std**2 
        return score

    logging.info("Starting Partial-Pooling (global) inference...")

    # #LOADING FROM WEIGTH
    # dummy_test_set_sample = {k: v[:2] for k, v in test_data.items()}

    # # Flatten streams into batch dim, matching what the model saw during training
    # n_streams = len(cfg.target_streams)
    # dummy_test_set_sample[cfg.sim_data] = dummy_test_set_sample[cfg.sim_data].reshape(
    #     -1, 
    #     dummy_test_set_sample[cfg.sim_data].shape[-2],  # n_stars=300
    #     dummy_test_set_sample[cfg.sim_data].shape[-1],  # features=15
    # )
    # dummy_test_set_sample['j'] = dummy_test_set_sample['j'].reshape(-1, 1)

    # # Now adapt and build
    # dummy_adapted = workflow_global.adapter(dummy_test_set_sample)
    # dummy_tensors = keras.tree.map_structure(keras.ops.convert_to_tensor, dummy_adapted)
    # workflow_global.approximator.build_from_data(dummy_tensors)
    # workflow_global.approximator.load_weights(model_path.replace('.keras', '.weights.h5'))
    # print('Model loaded and weights set successfully.')


    # Update inference kwargs with values from config
    workflow_global.approximator.inference_network.integrate_kwargs.update({
        'method': cfg.method,
        'steps': cfg.steps,
        'compositional_bridge_d1': 1/cfg.inverse_compositional_bridge_d1,
        'mini_batch_size': cfg.mini_batch_size,
        "max_steps": cfg.max_steps,
        })
    
    # Precompute v_circ for ALL test samples: (N_test, 38)
    test_data = {k: test_data[k][:5] for k in test_data.keys()}  # TEMP: use only 10 samples for quick testing
    vel_circ_true = vmap_vcirc({k: test_data[k][:, 0] for k in cfg.parameters_global})
    print('True circular velocities shape:', vel_circ_true.shape)  # (N_test, 38)

    all_posteriors = []
    N_test = vel_circ_true.shape[0]

    # for j in range(N_test):
    #     print(f"\n--- Test sample {j+1}/{N_test} ---")

    #     # Fixed reference for this sample: (38,)
    #     vel_circ_j = vel_circ_true[j]   # (38,)

    #     std_layer = workflow_global.approximator.standardize_layers["inference_variables"]

    #     def c_ineq_raw(z, vel_circ_ref=vel_circ_j, tol=0.3):
    #         # z is in standardized inference-variable space
    #         params = std_layer(z, forward=True)
    #         print("cfg.parameters_global =", cfg.parameters_global)
    #         print("params shape =", params.shape)
    #         parameters_dict = {k: params[:, i] for i, k in enumerate(cfg.parameters_global)}
    #         jax.debug.print('params in c_ineq_raw: {parameters_dict}', parameters_dict={k: parameters_dict[k][:10] for k in cfg.parameters_global})
    #         jax.debug.print('true params for sample {j}: {true_params}', j=j, true_params={k: test_data[k][j, 0] for k in cfg.parameters_global})

    #         vel_circ_pred = jax.vmap(
    #             lambda p: circular_velocity_at_xyz(xyz_batch, config, p)
    #         )(parameters_dict)  # (B, 38)

    #         vel_circ_pred = jnp.nan_to_num(vel_circ_pred, nan=0.0, posinf=0.0, neginf=0.0)
    #         # jax.debug.print("vel_circ_pred={vel_circ_pred}, vel_circ_ref={vel_circ_ref}", vel_circ_pred=vel_circ_pred, vel_circ_ref=vel_circ_ref)

    #         rel_dev = (vel_circ_pred - vel_circ_ref[None, :]) / (jnp.abs(vel_circ_ref[None, :]) + 1e-6)
    #         abs_rel = jnp.sqrt(rel_dev**2 + 1e-8)

    #         # per-sample inequality: should be < 0 when feasible
    #         c_per_sample = jnp.mean(abs_rel - tol, axis=-1)  # (B,)
    #         # jax.debug.print("c_ineq_raw: vel_circ_pred={vel_circ_pred}, rel_dev={rel_dev}, abs_rel={abs_rel}, c_per_sample={c_per_sample}",
    #         #                 vel_circ_pred=vel_circ_pred, rel_dev=rel_dev, abs_rel=abs_rel, c_per_sample=c_per_sample)  
    #         return c_per_sample

    #     def mild_scaling_function(t):
    #         # much milder than default alpha^2/sigma^2
    #         return keras.ops.clip(0.05 * (1.0 - t), 1e-3, 0.05)
  


    #     # Conditions for this single test sample — add batch dim back for the model
    #     conditions_j = {
    #         cfg.sim_data: test_data[cfg.sim_data][j:j+1],   # (1, n_streams, n_stars, features)
    #         "j": test_data["j"][j:j+1],                      # (1, n_streams, 1)
    #     }
    #     kwargs_j = {"attention_mask": test_data["attention_mask"][j:j+1]}

    #     posterior_j = workflow_global.compositional_sample(
    #         num_samples=cfg.n_samples,
    #         conditions=conditions_j,
    #         compute_prior_score=prior_global_score,
    #         batch_size=1,
    #         kwargs=kwargs_j,
    #         guidance_constraints=dict(
    #             constraints=c_ineq_raw,          # raw c(x), no softplus here
    #             guidance_strength=1e-3,          # start small (1e-4..1e-2 sweep)
    #             scaling_function=mild_scaling_function,
    #             reduce="sum",
    #         ),
    #     )
    #     # quick NaN/Inf check
    #     for kk, vv in posterior_j.items():
    #         arr = np.asarray(vv)
    #         if not np.isfinite(arr).all():
    #             print(f"[WARN] Non-finite posterior in key={kk}:",
    #                   "nan=", np.isnan(arr).any(), "inf=", np.isinf(arr).any())
    #     all_posteriors.append(posterior_j)
    # global_posterior = {
    #     k: np.stack([np.squeeze(p[k], axis=0) for p in all_posteriors], axis=0)
    #     for k in all_posteriors[0].keys()
    # }
    # # (1, 1000, 1) --squeeze axis=0--> (1000, 1) --stack 100x--> (100, 1000, 1) ✓
    # print("Final posterior shape (first key):",
    #     global_posterior[list(global_posterior.keys())[0]].shape)



    # def constrain(z, ):
    #     print(z.shape)
    #     params = workflow_global.approximator.standardize_layers["inference_variables"](z, forward=False)
        
    #     parameters_dict = {k: params[:, i] for i, k in enumerate(cfg.parameters_global)}
    #     jax.debug.print('params in c_ineq_raw: {parameters_dict}', parameters_dict={k: parameters_dict[k][:10] for k in cfg.parameters_global})
    #     jax.debug.print('true params for sample: {true_params}', true_params={k: test_data[k][:, 0] for k in cfg.parameters_global})
    #     quit()
    #     return params




    # global_posterior = workflow_global.compositional_sample(
    #                     num_samples=cfg.n_samples,
    #                     conditions={cfg.sim_data: test_data[cfg.sim_data], 
    #                                 "j": test_data["j"]},
    #                     compute_prior_score=prior_global_score,
    #                     batch_size = cfg.batch_size,
    #                     kwargs={'attention_mask': test_data['attention_mask']},
    #                     guidance_constraints=dict(constraints=constrain)
    #                     )
    
    # Use only a subset if desired
    # test_data = {k: test_data[k][:5] for k in test_data.keys()}

    # Split test set into chunks
    # ...existing code...

    eval_chunk_size = 5
    all_posteriors = []
    N_test = test_data[cfg.sim_data].shape[0]

    for start in range(0, N_test, eval_chunk_size):
        end = min(start + eval_chunk_size, N_test)
        print(f"\n--- Sampling batch {start}:{end} ---")

        # chunk-specific conditions
        conditions_batch = {
            cfg.sim_data: test_data[cfg.sim_data][start:end],
            "j": test_data["j"][start:end],
        }
        kwargs_batch = {
            "attention_mask": test_data["attention_mask"][start:end]
        }

        # chunk-specific reference circular velocities
        vel_circ_chunk = vel_circ_true[start:end]  # (chunk_size, 38)
        std_layer = workflow_global.approximator.standardize_layers["inference_variables"]

        def c_ineq_raw(z, vel_circ_ref=vel_circ_chunk, tol=0.003):
            # z: (B, n_params) where B = batch_size * n_samples
            params = std_layer(z, forward=True)
            jax.debug.print('z: {z}', z=z,)
            parameters_dict = {k: params[:, i] for i, k in enumerate(cfg.parameters_global)}
            jax.debug.print('params in c_ineq_raw: {parameters_dict}', parameters_dict={k: parameters_dict[k][:10] for k in cfg.parameters_global})
    #         jax.debug.print('true params for sample {j}: {true_params}', j=j, true_params={k: test_data[k][j, 0] for k in cfg.parameters_global})

            vel_circ_pred = jax.vmap(
                lambda p: circular_velocity_at_xyz(xyz_batch, config, p)
            )(parameters_dict)  # (B, 38)

            vel_circ_pred = jnp.nan_to_num(vel_circ_pred, nan=0.0, posinf=0.0, neginf=0.0)

            # vel_circ_ref: (chunk_size, 38)
            # vel_circ_pred: (B, 38) where B = chunk_size * n_samples_per_chunk
            # Reshape vel_circ_ref to (chunk_size, 1, 38) then broadcast to (chunk_size, n_samples_per_chunk, 38)
            chunk_size = vel_circ_ref.shape[0]
            n_samples_per_chunk = z.shape[0] // chunk_size
            
            vel_circ_ref_expanded = jnp.repeat(vel_circ_ref, n_samples_per_chunk, axis=0)  # (B, 38)
            
            # Now both have shape (B, 38)
            rel_dev = (vel_circ_pred - vel_circ_ref_expanded) / (jnp.abs(vel_circ_ref_expanded) + 1e-6)
            abs_rel = jnp.sqrt(rel_dev**2 + 1e-8)

            # one scalar per posterior sample
            c_per_sample = jnp.mean(abs_rel - tol, axis=-1)  # (B,)
            return c_per_sample

        def mild_scaling_function(t):
            return keras.ops.clip(0.05 * (1.0 - t), 1e-3, 0.05)

        posterior_batch = workflow_global.compositional_sample(
            num_samples=cfg.n_samples,
            conditions=conditions_batch,
            compute_prior_score=prior_global_score,
            batch_size=(end - start),
            kwargs=kwargs_batch,
            guidance_constraints=dict(
                constraints=c_ineq_raw,
                guidance_strength=0.0,
                # scaling_function=mild_scaling_function,
                # reduce="sum",
            ),
        )

        all_posteriors.append(posterior_batch)

    global_posterior = {
        k: np.concatenate([p[k] for p in all_posteriors], axis=0)
        for k in all_posteriors[0].keys()
}



    print("global_posterior shapes:")
    for k, v in global_posterior.items():
        print(k, v.shape)
    
        
    # → (N_test, n_samples)
    os.makedirs(name= os.path.join(cfg.base_dir, cfg.results_dir), exist_ok=True)
    ps = global_posterior.copy()
    if cfg.use_streamax_simulator:
        #let's extract q from the dirx, diry, dirz of the halo, to be able to plot it and compare with the true value
        q_min = 0.5
        q_max = 1.5
        r_posterior = np.sqrt(ps['dirx_Triaxial_rotated_halo']**2 + ps['diry_Triaxial_rotated_halo']**2 + ps['dirz_Triaxial_rotated_halo']**2)
        u_uniform_posterior = special.erf(r_posterior/np.sqrt(2)) - np.sqrt(2/np.pi)*r_posterior*np.exp(-(r_posterior**2)/2)
        ps['$q_{NFW}$'] = q_min + (q_max-q_min)*u_uniform_posterior
        r_test = np.sqrt(test_data['dirx_Triaxial_rotated_halo']**2 + test_data['diry_Triaxial_rotated_halo']**2 + test_data['dirz_Triaxial_rotated_halo']**2)
        u_uniform_test = special.erf(r_test/np.sqrt(2)) - np.sqrt(2/np.pi)*r_test*np.exp(-(r_test**2)/2)
        test_data['$q_{NFW}$'] = q_min + (q_max-q_min)*u_uniform_test
        param_names_global = cfg.parameters_global + ['$q_{NFW}$']
        cfg.paramater_global_pretty = cfg.paramater_global_pretty + ['$q_{NFW}$']
        #apply flipping to have all halos with dirz > 0, to avoid the degeneracy in the definition of the angles of the halo and make the plots easier to interpret
        #only needed for the 100 epochs models with onlyhalo
        # mask_posterior = ps['dirz_Triaxial_rotated_halo'] < 0
        # ps['dirz_Triaxial_rotated_halo'][mask_posterior] *= -1
        # ps['dirx_Triaxial_rotated_halo'][mask_posterior] *= -1
        # ps['diry_Triaxial_rotated_halo'][mask_posterior] *= -1
        
    np.savez(os.path.join(cfg.base_dir, cfg.results_dir, 'global_posterior.npz'), **ps)

    ###############
    # PLOTS GLOBAL#
    ###############
    #true vs predicted recovery plots
    fig = bf.diagnostics.recovery(
        estimates=ps,
        targets=test_data,
        variable_names=cfg.paramater_global_pretty
        # variable_names = param_names_global
    )
    for ax in fig.get_axes():
        ax.grid(False)
        for txt in ax.texts:
            txt.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.3'))
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'global_recovery.pdf'))
    print('Saved global recovery plot')
    plt.show()
    #corner plot
    dataset_id = np.array([0])
    fig = bf.diagnostics.plots.pairs_posterior(
        estimates=ps,
        targets=test_data,
        dataset_id=dataset_id,
        variable_names=cfg.paramater_global_pretty,
        # variable_names = param_names_global,
    )
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, f'global_cornerplot_datasetid_{dataset_id}.pdf'))
    print(f'Saved global corner plot for dataset id {dataset_id}')
    plt.show()
    #calibration plot
    fig = bf.diagnostics.calibration_ecdf(
        estimates=ps,
        targets=test_data,
        difference=True,
        variable_names=cfg.paramater_global_pretty
    )
    for ax in fig.get_axes():
        ax.grid(False)
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'global_calibration.pdf'))
    print('Saved global calibration plot')
    plt.show()
    #calibration plot stacked
    from utils.utils_plot import calibration_ecdf
    fig = calibration_ecdf(
        estimates=ps,
        targets=test_data,
        difference=True,
        variable_names=cfg.paramater_global_pretty,
        stacked = True,
        rank_ecdf_color=plt.cm.magma(np.linspace(0, 1, len(cfg.paramater_global_pretty))),

    )
    for ax in fig.get_axes():
        ax.grid(False)
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'global_calibration_stacked.pdf'))
    plt.show()
    fig = calibration_ecdf(
        estimates=ps,
        targets=test_data,
        difference=False,
        variable_names=cfg.paramater_global_pretty,
        stacked = True,
        rank_ecdf_color=plt.cm.magma(np.linspace(0, 1, len(cfg.paramater_global_pretty))),

    )
    for ax in fig.get_axes():
        ax.grid(False)
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'global_calibration_stacked_no_diff.pdf'))
    #calibration plot without diff
    fig = bf.diagnostics.calibration_ecdf(
        estimates=ps,
        targets=test_data,
        difference=False,
        variable_names=cfg.paramater_global_pretty
    )
    for ax in fig.get_axes():
        ax.grid(False)
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'global_calibration_no_diff.pdf'))
    plt.show()
    #histograms
    global_posterior_stream_1 = {k: ps[k] for k in list(ps.keys())[:4]}
    test_data_stream_1 = {k: test_data[k] for k in list(global_posterior_stream_1.keys())}
    fig_1 = bf.diagnostics.plots.calibration_histogram(
        estimates=global_posterior_stream_1, 
        targets=test_data_stream_1,
        variable_names=cfg.paramater_global_pretty[:4]
        # variable_names = param_names_global
    )
    for ax in fig_1.get_axes():
        ax.grid(False)
    fig_1.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'global_histograms_1.pdf'))
    plt.show()
    global_posterior_stream_2 = {k: ps[k] for k in list(ps.keys())[4:]}
    test_data_stream_2 = {k: test_data[k] for k in list(global_posterior_stream_2.keys())}
    fig_2 = bf.diagnostics.plots.calibration_histogram(
        estimates=global_posterior_stream_2, 
        targets=test_data_stream_2,
        variable_names=cfg.paramater_global_pretty[4:]
        # variable_names = param_names_global
    )
    for ax in fig_2.get_axes():
        ax.grid(False)
    fig_2.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'global_histograms_2.pdf'))
    plt.show()
    print('Saved global histograms plot')

    # z_score contraction
    fig = bf.diagnostics.plots.z_score_contraction(
        estimates=ps, 
        targets=test_data,
        variable_names=cfg.paramater_global_pretty
        # variable_names = param_names_global
    )
    for ax in fig.get_axes():
        ax.grid(False)
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, 'global_z_score_contraction.pdf'))
    print('Saved global z-score contraction plot')
    plt.show()
    print('Finished evaluation with composition')
    print('Results calibration saved in ', os.path.join(cfg.base_dir, cfg.results_dir, 'global_calibration.pdf'))
    

    ###############
    # local model # 
    ###############

if __name__ == "__main__":
    main()