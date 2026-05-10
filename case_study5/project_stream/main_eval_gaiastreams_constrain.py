from autocvd import autocvd
autocvd(num_gpus = 1)
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import yaml
import matplotlib.pyplot as plt
from functools import partial

from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
from chainconsumer import Chain, ChainConsumer, ChainConfig
import pandas as pd
from scipy import special
import jax.numpy as jnp 

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"
import keras
import bayesflow as bf


import logging
logging.getLogger('bayesflow').setLevel(logging.DEBUG)

from config.EvalConfig import EvalConfig
from utils.utils_train_jax import AugmentationsClass #we will need to use the augmentations on the test_set

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
MSUNPC3_to_MSUNKPC3 = float((u.Msun / u.pc**3).to(u.Msun / u.kpc**3))

#as in the prior used for the multistream gala
BULGE_M     = 4501365375.06545 * MSUN_TO_CODE_MASS
BULGE_ALPHA = 1.8
BULGE_RC    = 1.9 * KPC_TO_CODE_LENGTH

def s(x):
    """Squeeze array params to scalars for JAX grad compatibility."""
    return jnp.squeeze(x)

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

    # Extract parameters
    rho_thin = s(p['rho_thin_disk']) * MSUNPC3_to_MSUNKPC3
    hr_thin  = s(p['hr_thin_disk'])
    hz_thin  = s(p['hz_thin_disk'])

    rho_thick = s(p['rho_thick_disk']) * MSUNPC3_to_MSUNKPC3
    hr_thick  = s(p['hr_thick_disk'])
    hz_thick  = s(p['hz_thick_disk'])

    # Mass formula
    M_thin  = 4 * jnp.pi * rho_thin  * hr_thin**2  * hz_thin
    M_thick = 4 * jnp.pi * rho_thick * hr_thick**2 * hz_thick

    return {
        "M_thin_disk": M_thin,
        "M_thick_disk": M_thick,
    }

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

@hydra.main(version_base=None, config_path="config", config_name="eval_config_gaia",)
def main(cfg: EvalConfig):

    print(cfg)
    print("##############")
    model_path = os.path.join(cfg.base_dir, cfg.model_dir, 'global_model.keras' )
    print('Loading model from ', model_path)
    # Fix ArrayImpl serialization issue in the .keras file
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
    # with open(os.path.join(cfg.base_dir, cfg.model_dir, '.hydra', 'config.yaml'), "r") as f:
    #     model_config = yaml.safe_load(f)
    # print(model_config)
    model_config = {'global_model':
                    {
                        'inference_mlp_width': cfg.global_model.inference_mlp_width,
                        'inference_mlp_depth': cfg.global_model.inference_mlp_depth,
                        'inference_time_embedding_dim': cfg.global_model.inference_time_embedding_dim,
                        'summary_dim': cfg.global_model.summary_dim,
                        'num_heads': cfg.global_model.num_heads,
                        'embed_dims': cfg.global_model.embed_dims,
                        'mlp_depths': cfg.global_model.mlp_depths,
                        'mlp_widths': cfg.global_model.mlp_widths,
                        'dropout': cfg.global_model.dropout,
                    }
                }

    test_data_path = '/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz'
    print('Loading test data from ', test_data_path)
    test_data = dict(np.load(test_data_path, allow_pickle=True))
    test_data = {k: test_data[k] for k in [cfg.sim_data, "j", "attention_mask", "magnitudes"] }
    for k in [cfg.sim_data, "attention_mask", "magnitudes"]:
        print(f"{k} shape before truncation: {test_data[k].shape}")
        if len(test_data[k].shape) == 2:
            test_data[k] = test_data[k][:, :300]
        elif len(test_data[k].shape) == 3:
            test_data[k] = test_data[k][:, :, :300]
        elif len(test_data[k].shape) == 4:
            test_data[k] = test_data[k][:, :, :300]
        print(f"{k} shape after truncation: {test_data[k].shape}")

    # print('Test data keys and shape: ', test_data.keys(), test_data[list(test_data.keys())[0]].shape)
    other_things = ['attention_mask', 'magnitudes', 'vloss_mask']
    keys_to_drop = set(test_data.keys()) - set(param_names_global) - {sim_data} - set(inference_conditions) -  set(other_things)
    keys_to_drop = list(keys_to_drop) 

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
        inference_network=inference_network,
        standardize=["inference_variables", "summary_variables"]
    )
    workflow_global.approximator = keras.models.load_model(model_path)
    # Augmentation
    augmentations_class = AugmentationsClass(cfg)
    augmentations = []
    if "remove_los_velocity" in cfg.augmentations:
        augmentations.append(augmentations_class.remove_los_velocity)
    if "sample_obs_error" in cfg.augmentations:
        augmentations.append(augmentations_class.sample_obs_error)
    if "observational_window" in cfg.augmentations:
        augmentations.append(augmentations_class.observational_window)
    if "observational_window_random" in cfg.augmentations:
        augmentations.append(augmentations_class.observational_window_random)
    if "mask_vlos" in cfg.augmentations:
        augmentations.append(augmentations_class.mask_vlos)
    if "concatentate_sigma_error_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatentate_sigma_error_to_sim_data)
    if "concatenate_magnitudes_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_magnitudes_to_sim_data)
    if "concatenate_vlos_mask_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_vlos_mask_to_sim_data)
    if "concatenate_j_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_j_to_sim_data)

    #reshape the streams dimensions
    test_data[cfg.sim_data] = test_data[cfg.sim_data].reshape(-1, test_data[cfg.sim_data].shape[-2], test_data[cfg.sim_data].shape[-1])
    test_data['j'] = test_data['j'].reshape(-1, 1)
    print('Test data sim shape before augmentation: ', test_data[cfg.sim_data].shape)
    for aug in augmentations:
        print(f"Applying augmentation: {aug.__name__}")
        test_data = aug(test_data)

    test_data[cfg.sim_data] = test_data[cfg.sim_data].reshape(-1, len(cfg.target_streams.keys()), test_data[cfg.sim_data].shape[-2], test_data[cfg.sim_data].shape[-1])
    test_data['j'] = test_data['j'].reshape(-1,len(cfg.target_streams.keys()), 1)
    print('Test data sim shape after augmentation: ', test_data[cfg.sim_data].shape)
    print('Test data keys: ', test_data.keys())
    for k in test_data.keys():
        print('##########')
        print(f"{k} shape: {test_data[k].shape}")
    with open(os.path.join(cfg.base_dir, cfg.data_dir, '.hydra', 'config.yaml'), "r") as f:
        test_sim_config = yaml.safe_load(f)

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
    
    #constrain
    vel_circ_true = jnp.array([220.0]) #at k=8 kpc, 220 km/s
    std_layer = workflow_global.approximator.standardize_layers["inference_variables"]

    # --- helper: constraint for a SINGLE parameter vector ---
    def _constraint_single(z_single: jnp.ndarray) -> jnp.ndarray:
        """
        z_single: shape (n_params,) — one sample in latent space.
        Returns: shape (1,) — constraint residual for that sample.
        """
        # std_layer expects a batch dim → add/remove it
        params = std_layer(z_single[None], forward=True)   # (1, n_params)
        parameters_dict = {
            k: params[0, i]                                # scalar (0-d array)
            for i, k in enumerate(cfg.parameters_global)
        }
        vel_circ_pred = (
            circular_velocity_at_xyz(
                xyz=jnp.array([[8.0, 0.0, 0.0]]),
                config=config,
                p=parameters_dict,
            )
            * CODE_VEL_TO_KMS
        )                                                  # shape (1,)

        rel_diff = (vel_circ_pred - vel_circ_true) / vel_circ_true
        return jnp.sqrt(rel_diff**2 + 1e-8)               # shape (1,)


    # --- batched constraint passed to BayesFlow ---
    def constraint(z: jnp.ndarray) -> jnp.ndarray:
        """
        z: shape (batch, n_params) — full particle batch from the sampler.
        Returns: shape (batch, 1).
        """
        return jax.vmap(_constraint_single)(z)



    logging.info("Starting Partial-Pooling (global) inference...")
    workflow_global.approximator.inference_network.integrate_kwargs.update({
        'method': cfg.method,
        'steps': cfg.steps,
        'compositional_bridge_d1': 1/cfg.inverse_compositional_bridge_d1,
        'mini_batch_size': cfg.mini_batch_size,
        "max_steps": cfg.max_steps,
        })
    global_posterior = workflow_global.compositional_sample(
                        num_samples=cfg.n_samples,
                        conditions={cfg.sim_data: test_data[cfg.sim_data], 
                                    "j": test_data["j"]},
                        compute_prior_score=prior_global_score,
                        batch_size = cfg.batch_size,
                        kwargs={'attention_mask': test_data['attention_mask']},
                        guidance_constraints=dict(
                                constraints=constraint,
                                guidance_strength=0.6,
                                # scaling_function=mild_scaling_function,
                                # reduce="sum",
                            ),
                        )
    os.makedirs(name= os.path.join(cfg.base_dir, cfg.results_dir), exist_ok=True)
    # v shape: (n_obs, n_samples, n_params)
    # collapse obs (axis 0) and params (axis 2) → mask over samples only
    nan_mask = {
        k: ~np.isnan(v).any(axis=(0, 2))   # shape: (n_samples,)
        for k, v in global_posterior.items()
    }
    combined_mask = np.logical_and.reduce(list(nan_mask.values()))  # (n_samples,)

    global_posterior = {
        k: v[:, combined_mask, :]           # keep obs and params dims intact
        for k, v in global_posterior.items()
    }

    n_removed = (~combined_mask).sum()
    print(f"Removed {n_removed} / {len(combined_mask)} samples containing NaNs")
    ps = global_posterior.copy()
    q_min = 0.5
    q_max = 1.5
    if cfg.use_streamax_simulator:
        r_posterior = np.sqrt(ps['dirx_Triaxial_rotated_halo']**2 + ps['diry_Triaxial_rotated_halo']**2 + ps['dirz_Triaxial_rotated_halo']**2)
        u_uniform_posterior = special.erf(r_posterior/np.sqrt(2)) - np.sqrt(2/np.pi)*r_posterior*np.exp(-(r_posterior**2)/2)
        ps['$q_{NFW}$'] = q_min + (q_max-q_min)*u_uniform_posterior
        param_names_global = cfg.parameters_global + ['$q_{NFW}$']
        cfg.paramater_global_pretty = cfg.paramater_global_pretty + ['$q_{NFW}$']
    np.savez(os.path.join(cfg.base_dir, cfg.results_dir, 'global_posterior.npz'), **ps)
    ###############
    # PLOTS GLOBAL#
    ###############
    print('shapes of posterior samples: ', {k: v.shape for k, v in ps.items()})
    for k in ps.keys():
        ps[k] = ps[k].reshape(-1,)
    df = pd.DataFrame(ps) 
    print('Df columns before renaming: ', df.columns)
    df.columns = list(cfg.paramater_global_pretty)
    print('Df columns after renaming: ', df.columns)
    c = ChainConsumer()
    c.add_chain(Chain(samples=df, name="Global"))
    # fig = c.plotter.plot()
    # fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, f'global_cornerplot.pdf'))
    # print(f'Saved global corner plot')
    # plt.show()

    #Single stream posteriors
    for stream_name in cfg.target_streams.keys():
        print(f"Starting inference for stream {stream_name}...")
        test_data_stream = {cfg.sim_data: test_data[cfg.sim_data][:, cfg.target_streams[stream_name], :, :], 
                            "j": test_data["j"][:, cfg.target_streams[stream_name], :],
                            }
        print('test data stream shapes: ', {k: v.shape for k, v in test_data_stream.items()})
        print('we should see also the magnitude and sigma concatenated, and vlos_mask if used')
        attention_mask_stream = test_data['attention_mask'][cfg.target_streams[stream_name], :, :].reshape(1, -1)
        print('attention mask stream shape: ', attention_mask_stream.shape)
        posterior_stream = workflow_global.sample(
                            num_samples=cfg.n_samples,
                            conditions=test_data_stream,
                            kwargs={'attention_mask': attention_mask_stream}
                            )
        ps_stream = posterior_stream.copy()
        if cfg.use_streamax_simulator:
            r_posterior = np.sqrt(ps_stream['dirx_Triaxial_rotated_halo']**2 + ps_stream['diry_Triaxial_rotated_halo']**2 + ps_stream['dirz_Triaxial_rotated_halo']**2)
            u_uniform_posterior = special.erf(r_posterior/np.sqrt(2)) - np.sqrt(2/np.pi)*r_posterior*np.exp(-(r_posterior**2)/2)
            ps_stream['$q_{NFW}$'] = q_min + (q_max-q_min)*u_uniform_posterior
        np.savez(os.path.join(cfg.base_dir, cfg.results_dir, f'{stream_name}_posterior.npz'), **ps_stream)
        print(f'Saved posterior samples for stream {stream_name}')
        for k in ps_stream.keys():
            ps_stream[k] = ps_stream[k].reshape(-1,)
        df_stream = pd.DataFrame(ps_stream) 
        df_stream.columns = list(cfg.paramater_global_pretty)
        c.add_chain(Chain(samples=df_stream, name=f"{stream_name}"))
    c.set_override(ChainConfig(shade=False))
    fig = c.plotter.plot()
    fig.savefig(os.path.join(cfg.base_dir, cfg.results_dir, f'global_cornerplot.pdf'))
    print(f'Saved global corner plot with all streams in pathc: {os.path.join(cfg.base_dir, cfg.results_dir, "global_cornerplot.pdf")}')


if __name__ == "__main__":
    main()