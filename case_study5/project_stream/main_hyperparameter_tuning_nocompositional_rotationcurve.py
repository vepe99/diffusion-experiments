from autocvd import autocvd
autocvd(num_gpus = 1, interval=1)
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  
from tqdm import tqdm

import numpy as np

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"
import gc

import bayesflow as bf
import keras
from bayesflow.diagnostics import metrics as bf_metrics
import yaml
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import logging
logging.getLogger('bayesflow').setLevel(logging.DEBUG)

from utils.utils_train_jax_new_rotationcurve import (AugmentationsClass,
                                                     compute_standardization, 
                                                     apply_standardization,
                                                     save_stats)
from utils.custom_summary_network import SetTransformer, FusionNetwork
from utils.utils_plot import calibration_ecdf
import matplotlib.pyplot as plt
from chainconsumer import Chain, ChainConsumer, ChainConfig
import pandas as pd

import jax
import jax.numpy as jnp

def clear_gpu_memory():
    """Helper function to aggressively clear GPU memory."""
    gc.collect()



def objective(trial, cfg):
    augmentations_class.key = jax.random.PRNGKey(42)
    # Clear memory at the start of each trial
    clear_gpu_memory()
    results_dir =os.path.join(storage_path, f'model_{trial.number}')
    os.makedirs(results_dir, exist_ok=True)

    #We start to sample the hyperaramas
    #f1 irst SetTransformer
    summary_dim_SetTransformer = trial.suggest_int("SetTransformer_summary_dim", 32, 64)
    num_attention_blocks_SetTransformer = trial.suggest_int("SetTransformer_num_attention_blocks", 1, 4)
    num_heads_SetTransformer = trial.suggest_int("SetTransformer_num_heads", 1, 8)
    embed_dim_multiplier_SetTransformer = trial.suggest_int("SetTransformer_embed_dim_multiplier", 4, 16)
    embed_dims_SetTransformer = embed_dim_multiplier_SetTransformer * num_heads_SetTransformer  # always divisible, range ~16-128
    num_seeds = trial.suggest_int("SetTransformer_num_seeds", 2, 6)
    #go to tuples
    embed_dims_SetTransformer = (embed_dims_SetTransformer,) * num_attention_blocks_SetTransformer
    num_heads_SetTransformer = (num_heads_SetTransformer,) * num_attention_blocks_SetTransformer
    #we define the summary_netwrok_a
    summary_network_a = SetTransformer(
        summary_dim=summary_dim_SetTransformer,
        embed_dims=embed_dims_SetTransformer,
        num_heads=num_heads_SetTransformer,
        num_seeds=num_seeds,
        dropout=0.05,
    )

    #2 second the TimeSeriesTransformer
    summary_dim_TimeSeriesTransformer = trial.suggest_int("TimeSeriesTransformer_summary_dim", 32, 64)
    num_attention_blocks_TimeSeriesTransformer = trial.suggest_int("TimeSeriesTransformer_num_attention_blocks", 1, 4)
    num_heads_TimeSeriesTransformer = trial.suggest_int("TimeSeriesTransformer_num_heads", 1, 8)
    embed_dim_multiplier_TimeSeriesTransformer = trial.suggest_int("TimeSeriesTransformer_embed_dim_multiplier", 4, 16)
    embed_dims_TimeSeriesTransformer = embed_dim_multiplier_TimeSeriesTransformer * num_heads_TimeSeriesTransformer  # always divisible
    #go to tuples
    embed_dims_TimeSeriesTransformer = (embed_dims_TimeSeriesTransformer,) * num_attention_blocks_TimeSeriesTransformer
    num_heads_TimeSeriesTransformer = (num_heads_TimeSeriesTransformer,) * num_attention_blocks_TimeSeriesTransformer
    #we define the summary_netwrok_b
    summary_network_b = bf.networks.TimeSeriesTransformer(
        summary_dim=summary_dim_TimeSeriesTransformer,
        embed_dims=embed_dims_TimeSeriesTransformer,
        num_heads=num_heads_TimeSeriesTransformer,
        dropout=0.05,
    )

    #3 we sample the head 
    width_head = trial.suggest_int("FusionNetwork_width_head", 32, 256, step=16)
    length_head = trial.suggest_int("FusionNetwork_length_head", 1, 4)
    final_summary_dimension = trial.suggest_int("FusionNetwork_final_summary_dimension", 16, 64,)
    head = keras.Sequential(
        [bf.networks.MLP(widths=[width_head] * length_head), keras.layers.Dense(units=final_summary_dimension)]
    )

    #4 we create the summary network with the two backbones and the head
    summary_network = FusionNetwork(
        backbones={"input_a": summary_network_a, "input_b": summary_network_b},
        head=head,
    )

    #5 we sample the hyperparameters for the inference network
    inference_mlp_depth = trial.suggest_int("inference_mlp_depth", 2, 8)
    inference_mlp_width = trial.suggest_int("inference_mlp_width", 32, 256, step=16)
    time_embedding_dim = trial.suggest_int("inference_time_embedding_dim", 16, 64, step=2)
    inference_network = bf.networks.DiffusionModel(
        subnet_kwargs={
            "widths": [inference_mlp_width] * inference_mlp_depth,
            "time_embedding_dim": time_embedding_dim,
        }
    )


    param_names_global = list(cfg.parameters_global)
    sim_data = 'sim_data_projected'
    inference_conditions = 'j' #just 1
    keys_to_drop = (
        set(training_data.keys())
        - set(param_names_global)
        - {sim_data}
        - set(inference_conditions)
        - {"vcirc_kms"}
    )
    keys_to_drop = list(keys_to_drop)

        
    try:
        adapter = (
            bf.adapters.Adapter()
            .to_array()
            .convert_dtype("float64", "float32")
            .drop(keys_to_drop)
            .rename(inference_conditions, "inference_conditions")
            .concatenate(param_names_global, into="inference_variables")
            .rename(sim_data, "input_a")
            .rename('attention_mask', 'summary_attention_mask')
            .rename("vcirc_kms", "input_b")
            .group(
            ["input_a", "input_b",], into="summary_variables")   
            )

        workflow_global = bf.CompositionalWorkflow(
            adapter=adapter,
            summary_network=summary_network,
            inference_network=inference_network,
            standardize=["inference_variables", "summary_variables"],
        )

        # --- Training with batch-size retry ---
        batch_size_training = 1024
        for attempt in range(2):
            try:
                history = workflow_global.fit_offline(
                    training_data,
                    epochs=2,
                    batch_size=batch_size_training,
                    verbose=2,
                    augmentations=augmentations,
                )
                loss_plot = bf.diagnostics.plots.loss(
                    history,
                )
                loss_plot.savefig(os.path.join(results_dir, "loss_plot.pdf"))
                break  # success
            except Exception as e:
                err_str = str(e).lower()
                is_oom = "out of memory" in err_str or "cuda" in err_str or "resource exhausted" in err_str
                if is_oom and attempt == 0:
                    logging.warning(f"Trial {trial.number} OOM during training, halving batch size and retrying...")
                    batch_size_training //= 2
                    clear_gpu_memory()
                else:
                    raise  # non-OOM or second failure — re-raise

        # --- Sampling with batch-size retry ---
        batch_size_sampling = 250
        conditions = {
            "input_a": test_data[cfg.sim_data],         # (300, 300, 15)
            "input_b": test_data["vcirc_kms"],            # (300, 34, 1)  <-- missing
            "summary_attention_mask": test_data["attention_mask"],      # (300, 1, 300)
            "j": test_data["j"],                            # (300, 1)
        }
        for attempt in range(3):
            try:
                global_posterior = workflow_global.sample(
                    num_samples=1000,
                    conditions=conditions,
                    batch_size=batch_size_sampling,
                    kwargs={'summary_attention_mask': test_data['attention_mask']},
                )
                break
            except Exception as e:
                err_str = str(e).lower()
                is_oom = "out of memory" in err_str or "cuda" in err_str or "resource exhausted" in err_str
                if is_oom and attempt == 0:
                    logging.warning(f"Trial {trial.number} OOM during sampling, halving batch size and retrying...")
                    batch_size_sampling //= 2
                    clear_gpu_memory()
                else:
                    raise

        root_mean_squared_error = bf_metrics.root_mean_squared_error(
            estimates=global_posterior,
            targets=test_data,
            variable_keys=param_names_global,
            variable_names=param_names_global,
        )
        calibration_errors = bf_metrics.calibration_error(
            estimates=global_posterior,
            targets=test_data,
            variable_keys=param_names_global,
            variable_names=param_names_global,
        )
        workflow_global.approximator.save(os.path.join(results_dir, "global_model.keras"))
        workflow_global.approximator.save_weights(os.path.join(results_dir, "global_model.weights.h5"))
        #calibration plot with diff
        fig = bf.diagnostics.calibration_ecdf(
            estimates=global_posterior,
            targets=test_data,
            difference=True,
            variable_names=cfg.paramater_global_pretty,
        )
        for ax in fig.get_axes():
            ax.grid(False)
        fig.savefig(os.path.join(results_dir, 'calibration.pdf'))
        plt.show()
        fig = bf.diagnostics.calibration_ecdf(
            estimates=global_posterior,
            targets=test_data,
            difference=False,
            variable_names=cfg.paramater_global_pretty,
        )
        fig.savefig(os.path.join(results_dir, 'calibration_nodiff.pdf'))
        plt.show()
        fig = bf.diagnostics.recovery(
            estimates=global_posterior,
            targets=test_data,
            variable_names=cfg.paramater_global_pretty,
        )
        fig.savefig(os.path.join(results_dir, 'recovery.pdf'))
        plt.show()


        #now we do the non composition of the multistream
        conditions_multistream = {
            "input_a": test_data_multistream[cfg.sim_data],         # (300, 
            "input_b": test_data_multistream["vcirc_kms"],            # (300, 34, 1)  <-- missing
            "summary_attention_mask": test_data_multistream["attention_mask"],      # (300, 1, 300)
            "j": test_data_multistream["j"],                            # (300, 1)
        }

        for attempt in range(3):
            try:
                global_posterior_multistream = workflow_global.sample(
                    num_samples=1000,
                    conditions= conditions_multistream,
                    batch_size=batch_size_sampling,
                    kwargs={'summary_attention_mask': test_data_multistream['attention_mask']},
                )
                break
            except Exception as e:
                err_str = str(e).lower()
                is_oom = "out of memory" in err_str or "cuda" in err_str or "resource exhausted" in err_str
                if is_oom and attempt == 0:
                    logging.warning(f"Trial {trial.number} OOM during sampling, halving batch size and retrying...")
                    batch_size_sampling //= 2
                    clear_gpu_memory()
                else:
                    raise

        #true vs predicted recovery plots
        fig = bf.diagnostics.recovery(
            estimates=global_posterior_multistream,
            targets=test_data_multistream,
            variable_names=cfg.paramater_global_pretty
            # variable_names = param_names_global
        )
        for ax in fig.get_axes():
            ax.grid(False)
            for txt in ax.texts:
                txt.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.3'))
        fig.savefig(os.path.join(results_dir, f'recovery_multistream_{N_multistream}_nocomposition.pdf'))
        #stacked
        fig = calibration_ecdf(
            estimates=global_posterior_multistream,
            targets=test_data_multistream,
            difference=True,
            variable_names=cfg.paramater_global_pretty,
            stacked = True,
            rank_ecdf_color=plt.cm.magma(np.linspace(0, 1, len(cfg.paramater_global_pretty))),

        )
        for ax in fig.get_axes():
            ax.grid(False)
        fig.savefig(os.path.join(results_dir, f'calibration_stacked_multistream_{N_multistream}_nocomposition.pdf'))
        plt.show()
        fig = calibration_ecdf(
            estimates=global_posterior_multistream,
            targets=test_data_multistream,
            difference=False,
            variable_names=cfg.paramater_global_pretty,
            stacked = True,
            rank_ecdf_color=plt.cm.magma(np.linspace(0, 1, len(cfg.paramater_global_pretty))),
        )
        for ax in fig.get_axes():
            ax.grid(False)
        fig.savefig(os.path.join(results_dir, f'calibration_stacked_no_diff_multistream_{N_multistream}_nocomposition.pdf'))


        #now we do the compositional sampling 
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

        test_data_multistream[cfg.sim_data] = test_data_multistream[cfg.sim_data].reshape(-1, len(cfg.target_streams.keys()), test_data_multistream[cfg.sim_data].shape[-2], test_data_multistream[cfg.sim_data].shape[-1])
        test_data_multistream['vcirc_kms'] = test_data_multistream['vcirc_kms'].reshape(-1, len(cfg.target_streams.keys()), test_data_multistream['vcirc_kms'].shape[-2], test_data_multistream['vcirc_kms'].shape[-1])
        test_data_multistream['j'] = test_data_multistream['j'].reshape(-1,len(cfg.target_streams.keys()), 1)

        # Reshape all data to have a single batch dimension (n_sims * n_streams)
        n_sims = test_data_multistream[cfg.sim_data].shape[0]
        n_streams = test_data_multistream[cfg.sim_data].shape[1]
        
        flat_input_a = test_data_multistream[cfg.sim_data].reshape(
            n_sims * n_streams, 
            test_data_multistream[cfg.sim_data].shape[-2], 
            test_data_multistream[cfg.sim_data].shape[-1]
        )
        # Reshape vcirc_kms to match the flattened batch dimension
        flat_input_b = test_data_multistream["vcirc_kms"].reshape(
            n_sims * n_streams,
            test_data_multistream["vcirc_kms"].shape[-2],
            test_data_multistream["vcirc_kms"].shape[-1]
        )
        flat_j = test_data_multistream["j"].reshape(n_sims * n_streams, 1)
        
        # Reshape attention mask to be (batch_size, 1, n_stars)
        flat_mask = test_data_multistream["attention_mask"].reshape(
            n_sims * n_streams, 1, test_data_multistream["attention_mask"].shape[-1]
        )

        # 1. Create a raw conditions dictionary using pre-adapter keys
        raw_conditions = {
            cfg.sim_data: flat_input_a,
            "vcirc_kms": flat_input_b,
            "attention_mask": flat_mask,
            "j": flat_j,
        }

        # 2. Use _prepare_conditions to process data through Adapter + SummaryNetwork + Standardizers
        resolved_flat, _, _ = workflow_global.approximator._prepare_conditions(data=raw_conditions)
        
        # Convert fully prepared tensor to numpy array before reshaping
        resolved_flat = np.array(resolved_flat)

        # 3. Reshape standard-resolved conditions back to compositional shape (n_datasets, n_compositional_conditions, ...)
        final_summary_outputs = resolved_flat.reshape(
            n_sims, n_streams, resolved_flat.shape[-1]
        )
        print('Final summary outputs shape after reshape: ', final_summary_outputs.shape)

        # 4. Run compositional sampling using only summary_outputs (this perfectly bypasses the reshape bug)
        batch_size_sampling = 30
        for attempt in range(3):
            try:
                global_posterior_compositional = workflow_global.compositional_sample(
                                        num_samples=1000,
                                        summaries=final_summary_outputs,
                                        method="two_step_adaptive",
                                        steps="adaptive",
                                        compositional_bridge_d1=1/6., 
                                        compute_prior_score=prior_global_score,
                                        batch_size= batch_size_sampling,
                                        )
                break
            except Exception as e:
                err_str = str(e).lower()
                is_oom = "out of memory" in err_str or "cuda" in err_str or "resource exhausted" in err_str
                if is_oom and attempt == 0:
                    logging.warning(f"Trial {trial.number} OOM during sampling, halving batch size and retrying...")
                    batch_size_sampling //= 2
                    clear_gpu_memory()
                else:
                    raise
        
        #plotting compositional
        #true vs predicted recovery plots
        fig = bf.diagnostics.recovery(
            estimates=global_posterior_compositional,
            targets=test_data_multistream,
            variable_names=cfg.paramater_global_pretty
            # variable_names = param_names_global
        )
        for ax in fig.get_axes():
            ax.grid(False)
            for txt in ax.texts:
                txt.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.3'))
        fig.savefig(os.path.join(results_dir, f'recovery_multistream_{N_multistream}.pdf'))
        #stacked
        fig = calibration_ecdf(
            estimates=global_posterior_compositional,
            targets=test_data_multistream,
            difference=True,
            variable_names=cfg.paramater_global_pretty,
            stacked = True,
            rank_ecdf_color=plt.cm.magma(np.linspace(0, 1, len(cfg.paramater_global_pretty))),

        )
        for ax in fig.get_axes():
            ax.grid(False)
        fig.savefig(os.path.join(results_dir, f'calibration_stacked_multistream_{N_multistream}.pdf'))
        plt.show()
        fig = calibration_ecdf(
            estimates=global_posterior_compositional,
            targets=test_data_multistream,
            difference=False,
            variable_names=cfg.paramater_global_pretty,
            stacked = True,
            rank_ecdf_color=plt.cm.magma(np.linspace(0, 1, len(cfg.paramater_global_pretty))),
        )
        for ax in fig.get_axes():
            ax.grid(False)
        fig.savefig(os.path.join(results_dir, f'calibration_stacked_no_diff_multistream_{N_multistream}.pdf'))

        
        #finally Gaia data
        other_things = ['attention_mask', 'magnitudes', 'vlos_mask', 'vlos_error', 'vlos_mask', 'vcirc_kms', ]
        keys_to_drop = set(test_data.keys()) - set(param_names_global) - {sim_data} - set(inference_conditions) -  set(other_things)
        keys_to_drop = list(keys_to_drop) 

        adapter = (
            bf.adapters.Adapter()
            .to_array()
            .convert_dtype("float64", "float32")
            .drop(keys_to_drop)
            .rename(inference_conditions, "inference_conditions")
            .concatenate(param_names_global, into="inference_variables")
            .rename(sim_data, "input_a")
            .rename('attention_mask', 'summary_attention_mask')
            .rename("vcirc_kms", "input_b")
            .group(
            ["input_a", "input_b",], into="summary_variables")   
            )


        # Reshape all data to have a single batch dimension (n_sims * n_streams)
        n_sims = test_data_gaia[cfg.sim_data].shape[0]
        n_streams = test_data_gaia[cfg.sim_data].shape[1]
        
        flat_input_a = test_data_gaia[cfg.sim_data].reshape(
            n_sims * n_streams, 
            test_data_gaia[cfg.sim_data].shape[-2], 
            test_data_gaia[cfg.sim_data].shape[-1]
        )
        # Reshape vcirc_kms to match the flattened batch dimension
        flat_input_b = test_data_gaia["vcirc_kms"].reshape(
            n_sims * n_streams,
            test_data_gaia["vcirc_kms"].shape[-2],
            test_data_gaia["vcirc_kms"].shape[-1]
        )
        flat_j = test_data_gaia["j"].reshape(n_sims * n_streams, 1)
        
        # Reshape attention mask to be (batch_size, 1, n_stars)
        flat_mask = test_data_gaia["attention_mask"].reshape(
            n_sims * n_streams, 1, test_data_gaia["attention_mask"].shape[-1]
        )

        # 1. Create a raw conditions dictionary using pre-adapter keys
        raw_conditions = {
            cfg.sim_data: flat_input_a,
            "vcirc_kms": flat_input_b,
            "attention_mask": flat_mask,
            "j": flat_j,
        }

        # 2. Use _prepare_conditions to process data through Adapter + SummaryNetwork + Standardizers
        resolved_flat, _, _ = workflow_global.approximator._prepare_conditions(data=raw_conditions)
        
        # Convert fully prepared tensor to numpy array before reshaping
        resolved_flat = np.array(resolved_flat)

        # 3. Reshape standard-resolved conditions back to compositional shape (n_datasets, n_compositional_conditions, ...)
        final_summary_outputs = resolved_flat.reshape(
            n_sims, n_streams, resolved_flat.shape[-1]
        )
        print('Final summary outputs shape after reshape: ', final_summary_outputs.shape)

        # 4. Run compositional sampling using only summary_outputs (this perfectly bypasses the reshape bug)
        batch_size_sampling = 30
        for attempt in range(3):
            try:
                global_posterior_gaia = workflow_global.compositional_sample(
                                        num_samples=1000,
                                        # conditions=None,
                                        summaries=final_summary_outputs,
                                        method="two_step_adaptive",
                                        steps="adaptive",
                                        compositional_bridge_d1=1/6., 
                                        compute_prior_score=prior_global_score,
                                        batch_size= batch_size_sampling,
                                        )
                break
            except Exception as e:
                err_str = str(e).lower()
                is_oom = "out of memory" in err_str or "cuda" in err_str or "resource exhausted" in err_str
                if is_oom and attempt == 0:
                    logging.warning(f"Trial {trial.number} OOM during sampling, halving batch size and retrying...")
                    batch_size_sampling //= 2
                    clear_gpu_memory()
                else:
                    raise
        ps = global_posterior_gaia.copy()

        np.savez(os.path.join(results_dir, 'global_posterior.npz'), **ps)
        ###############
        # PLOTS GLOBAL#
        ###############
        import gala.potential as gp
        from gala.units import galactic
        from astropy import units as u
        obs_R   = np.array([5.24,5.74,6.25,6.77,7.23,7.83,8.21,8.78,9.26,9.75,
                        10.25,10.75,11.25,11.75,12.24,12.74,13.25,13.74,14.23,14.74,
                        15.23,15.74,16.24,16.74,17.23,17.74,18.35,18.90,19.50,20.41,
                        21.28,22.39,23.16,24.00])          # kpc
        obs_Vc  = np.array([225.10,233.53,234.30,233.17,236.19,236.00,233.19,233.15,232.15,231.24,
                        230.34,230.54,229.11,227.48,226.69,225.56,224.90,223.57,221.10,220.19,
                        219.59,217.36,216.61,217.28,216.25,213.81,217.53,212.10,210.46,206.69,
                        207.71,203.72,205.20,200.64]) 
        obs_sVc = np.array([0.69,0.68,0.62,0.60,0.45,0.29,0.26,0.22,0.17,0.16,
                        0.17,0.18,0.19,0.20,0.25,0.27,0.27,0.31,0.40,0.43,
                        0.50,0.68,0.74,0.87,1.02,1.15,1.45,1.58,1.32,1.71,
                        1.69,2.01,2.50,4.94])              # km/s  (1σ)
        r_array = obs_R * u.kpc
        N_sample = 1000
        N_obs = len(obs_R)
        params = ['m_Triaxial_halo','r_Triaxial_halo','q2_Triaxial_halo',
                'rho_thin_disk','hr_thin_disk','hz_thin_disk',
                'rho_thick_disk','hr_thick_disk','hz_thick_disk',]
        all_vcirc  = np.zeros((N_sample, N_obs))   # km/s, stored for reuse
        for k in params:
            print('Shape posterior samples for ', k, ': ', ps[k].shape)

        for i in tqdm(range(N_sample)):
            p = {k: ps[k][0][i] for k in params}
            # print(f"Sample {i}: ", p)

            pot = gp.CCompositePotential()
            pot['halo']       = gp.NFWPotential(
                                    m=p['m_Triaxial_halo'][0], r_s=p['r_Triaxial_halo'][0],
                                    a=1, b=1, c=p['q2_Triaxial_halo'][0], units=galactic)
            pot['thin_disk']  = gp.MN3ExponentialDiskPotential(
                                    m=4*np.pi*p['rho_thin_disk'][0]*p['hr_thin_disk'][0]**2*p['hz_thin_disk'][0],
                                    h_R=p['hr_thin_disk'][0], h_z=p['hz_thin_disk'][0],
                                    units=galactic, positive_density=True)
            pot['thick_disk'] = gp.MN3ExponentialDiskPotential(
                                    m=4*np.pi*p['rho_thick_disk'][0]*p['hr_thick_disk'][0]**2*p['hz_thick_disk'][0],
                                    h_R=p['hr_thick_disk'][0], h_z=p['hz_thick_disk'][0],
                                    units=galactic, positive_density=True)
            pot['bulge']      = gp.PowerLawCutoffPotential(
                                    m=test_sim_config['priors_global']['m_bulge']['prior_parameters'][0],
                                    r_c=test_sim_config['priors_global']['r_bulge']['prior_parameters'][0],
                                    alpha=test_sim_config['priors_global']['alpha_bulge']['prior_parameters'][0], 
                                    units=galactic)

            v_circ = pot.circular_velocity(R=r_array, z=np.zeros_like(r_array))
            v_circ_kms = v_circ.to(u.km/u.s).value
            all_vcirc[i] = v_circ_kms
            # ── Rotation-curve plot ───────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(10, 6))

            for i in range(N_sample):
                ax.plot(obs_R, all_vcirc[i], color='grey', alpha=0.5, lw=0.4)
                    
            ax.errorbar(obs_R, obs_Vc, yerr=3*obs_sVc, fmt='o', color='red',
                        ms=3, lw=1, capsize=2, label='Observed ±3σ', zorder=5)
            ax.set_xlabel('Radius (kpc)')
            ax.set_ylabel('Circular Velocity (km/s)')
            fig.savefig(os.path.join(results_dir, f'global_rotation_curve.pdf'))
            plt.close(fig)


        print('shapes of posterior samples: ', {k: v.shape for k, v in ps.items()})
        for k in ps.keys():
            ps[k] = ps[k].reshape(-1,)
        df = pd.DataFrame(ps) 
        print('Df columns before renaming: ', df.columns)
        df.columns = list(cfg.paramater_global_pretty)
        print('Df columns after renaming: ', df.columns)
        c = ChainConsumer()
        c.add_chain(Chain(samples=df, name="Global"))


        #Single stream posteriors
        for stream_name in cfg.target_streams.keys():
            print(f"Starting inference for stream {stream_name}...")
            test_data_stream = {cfg.sim_data: test_data[cfg.sim_data][:, cfg.target_streams[stream_name], :, :], 
                                "j": test_data["j"][:, cfg.target_streams[stream_name], :],
                                "vcirc_kms": test_data["vcirc_kms"][:, cfg.target_streams[stream_name], :, :],
                                "attention_mask": test_data["attention_mask"][:, cfg.target_streams[stream_name], :],
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

            np.savez(os.path.join(cfg.base_dir, cfg.results_dir, f'{stream_name}_posterior.npz'), **ps_stream)
            print(f'Saved posterior samples for stream {stream_name}')
            for k in ps_stream.keys():
                ps_stream[k] = ps_stream[k].reshape(-1,)
            df_stream = pd.DataFrame(ps_stream) 
            df_stream.columns = list(cfg.paramater_global_pretty)
            c.add_chain(Chain(samples=df_stream, name=f"{stream_name}"))
        c.set_override(ChainConfig(shade=False))
        fig = c.plotter.plot()
        fig.savefig(os.path.join(results_dir, f'global_cornerplot.pdf'))



        return root_mean_squared_error["values"].mean(), calibration_errors["values"].mean()

    except optuna.TrialPruned:
        raise  # let Optuna handle it cleanly

    except Exception as e:
        err_str = str(e).lower()
        is_oom = "out of memory" in err_str or "cuda" in err_str or "resource exhausted" in err_str

        logging.error(f"Trial {trial.number} failed: {type(e).__name__}: {e}")

        # Cleanup
        for var_name in ("workflow_global", "global_posterior", "history"):
            try:
                del locals()[var_name]
            except KeyError:
                pass
        clear_gpu_memory()

        if is_oom:
            raise optuna.TrialPruned(f"OOM: {e}")
        else:
            # Non-OOM failures (e.g. bad hyperparam combos) get pruned too,
            # so the study continues rather than crashing entirely.
            raise optuna.TrialPruned(f"{type(e).__name__}: {e}")


from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from config.TrainConfig import TrainConfig
from optuna.storages import JournalStorage, JournalFileStorage


if __name__ == "__main__":
    # Initialize Hydra config without the decorator
    cs = ConfigStore.instance()
    cs.store(name="train_config", node=TrainConfig)

    # 2. Point to the directory containing train_config.yaml
    config_path = '/export/data/vgiusepp/latest_bayesflow/diffusion-experiments/case_study5/project_stream/config/'

    with initialize_config_dir(version_base=None, config_dir=config_path):
        # 3. Compose: loads train_config.yaml, validated against TrainConfig schema
        cfg = compose(config_name="train_config_new_rotationcurve")
    
    base_dir =  '/export/data/vgiusepp/latest_bayesflow/diffusion-experiments/case_study5/project_stream/data/'
    data_dir = 'streams/data_gala/'
    N_simulations = 300_000


    train_data_path = os.path.join(base_dir, data_dir, f"training_data_local_{N_simulations}.npz")
    training_data = dict(np.load(train_data_path, allow_pickle=True))
    training_data = {k: training_data[k][:290_000] for k in training_data.keys()}
    training_data_rotation_curve = dict(np.load('./data/plots/gala_rotcurv/rotation_curves.npz'))
    training_data['vcirc_kms'] = training_data_rotation_curve['vcirc_kms'][:290_000, :, None] #extra dimension (n_observation, len_r_kpc, 1)


    augmentations_class = AugmentationsClass(cfg)
    augmentations_class.key = jax.random.PRNGKey(42)
    augmentations = []

    
    if "cut_to_300_particles" in cfg.augmentations:
        augmentations.append(augmentations_class.cut_to_300_particles)
    # --- Coordinate transforms (must be first, before any masking) ---
    if "remove_los_velocity" in cfg.augmentations:
        augmentations.append(augmentations_class.remove_los_velocity)
    if "convert_distance_to_parallax" in cfg.augmentations:
        augmentations.append(augmentations_class.convert_distance_to_parallax)

    # --- Observational selection (window → subsample → compact) ---
    if "observational_window" in cfg.augmentations:
        augmentations.append(augmentations_class.observational_window)
    if "observational_window_spline" in cfg.augmentations:
        augmentations.append(augmentations_class.observational_window_spline)
    if "observational_window_random" in cfg.augmentations:
        augmentations.append(augmentations_class.observational_window_random)
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

    if "add_noise_to_vcirc" in cfg.augmentations:
        augmentations.append(augmentations_class.add_noise_to_vcirc)
    if "log10_vcirc" in cfg.augmentations:
        augmentations.append(augmentations_class.log10_vcirc)

        # --- Feature concatenations (must be last) ---
    if "concatentate_sigma_error_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatentate_sigma_error_to_sim_data)
    if "concatenate_magnitudes_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_magnitudes_to_sim_data)
    if "concatenate_vlos_mask_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_vlos_mask_to_sim_data)
    if "concatenate_j_to_sim_data" in cfg.augmentations:
        augmentations.append(augmentations_class.concatenate_j_to_sim_data)


    test_data = dict(np.load(train_data_path, allow_pickle=True))
    test_data = {k: test_data[k][-10_000:] for k in test_data.keys()}
    test_data['vcirc_kms'] = training_data_rotation_curve['vcirc_kms'][-10_000:, :, None] #extra dimension (n_observation, len_r_kpc, 1)
    
    del training_data_rotation_curve

    for aug in augmentations:
        print(f"Applying augmentation {aug.__name__} to test data...")
        test_data = aug(test_data)
    for k in test_data.keys():
        test_data[k] = np.array(test_data[k])

    augmentations_class.key = jax.random.PRNGKey(42)
    
    #now we load the multistream test set
    data_dir_multistream = 'streams/data_multistream_gala_new/'
    N_multistream = 100
    test_data_multistream_path = os.path.join(base_dir, data_dir_multistream, f"simulation_multistream_{N_multistream}.npz")
    test_data_multistream = dict(np.load(test_data_multistream_path, allow_pickle=True))
    test_data_multistream_rotation_curve = dict(np.load(f'./data/plots/gala_rotcurv_multistream/{N_multistream}/rotation_curves.npz'))
    test_data_multistream['vcirc_kms'] = test_data_multistream_rotation_curve['vcirc_kms'][:, :, None] #extra dimension (n_observation, len_r_kpc, 1)

    test_data_multistream[cfg.sim_data] = test_data_multistream[cfg.sim_data].reshape(-1, test_data_multistream[cfg.sim_data].shape[-2], test_data_multistream[cfg.sim_data].shape[-1])
    test_data_multistream['vcirc_kms'] = np.tile(test_data_multistream['vcirc_kms'], (3, 1)).reshape(-1, test_data_multistream['vcirc_kms'].shape[-2], test_data_multistream['vcirc_kms'].shape[-1])
    test_data_multistream['j'] = test_data_multistream['j'].reshape(-1, 1)
    print('Test data sim shape before augmentation: ', test_data_multistream[cfg.sim_data].shape)
    for aug in augmentations:
        print(f"Applying augmentation: {aug.__name__}")
        test_data_multistream = aug(test_data_multistream)
    for k in cfg.parameters_global:
        test_data_multistream[k] = np.repeat(test_data_multistream[k], 3, axis=0).reshape(-1, 1)

    
    #now we load gaiastream
    test_data_path_gaia = '/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/gaia_observed_streams_6Dwitherrors_cutNGC3201.npz'
    print('Loading test data from ', test_data_path_gaia)
    test_data_gaia = dict(np.load(test_data_path_gaia, allow_pickle=True))
    test_data_gaia = {k: test_data_gaia[k] for k in [cfg.sim_data, "j", "attention_mask", "magnitudes", 'vlos_error', 'vlos_mask'] }
    for k in [cfg.sim_data, "attention_mask", "magnitudes", 'vlos_error', 'vlos_mask']:
        print(f"{k} shape before truncation: {test_data_gaia[k].shape}")
        if len(test_data_gaia[k].shape) == 2:
            test_data_gaia[k] = test_data_gaia[k][:, :300]
            if k == "vlos_mask":
                test_data_gaia[k] = test_data_gaia[k][:, None, :]
        elif len(test_data_gaia[k].shape) == 3:
            test_data_gaia[k] = test_data_gaia[k][:, :, :300]
        elif len(test_data_gaia[k].shape) == 4:
            test_data_gaia[k] = test_data_gaia[k][:, :, :300]
        print(f"{k} shape after truncation: {test_data_gaia[k].shape}")

    test_data_gaia['vcirc_kms'] = augmentations_class.obs_Vc[None, :, None]
    print('Test data vcirc_kms shape after adding to test data: ', test_data_gaia['vcirc_kms'].shape)

    augmentations_gaia = []
    augmentations_gaia.append(augmentations_class.remove_los_velocity)
    augmentations_gaia.append(augmentations_class.sample_obs_error)
    augmentations_gaia.append(augmentations_class.concatentate_sigma_error_to_sim_data)
    augmentations_gaia.append(augmentations_class.concatenate_magnitudes_to_sim_data)
    augmentations_gaia.append(augmentations_class.concatenate_vlos_mask_to_sim_data)
    augmentations_gaia.append(augmentations_class.concatenate_j_to_sim_data)
    augmentations_gaia.append(augmentations_class.log10_vcirc)

    test_data_gaia[cfg.sim_data] = test_data_gaia[cfg.sim_data].reshape(-1, test_data_gaia[cfg.sim_data].shape[-2], test_data_gaia[cfg.sim_data].shape[-1])
    n_streams = len(cfg.target_streams.keys())
    test_data_gaia['vcirc_kms'] = np.repeat(test_data_gaia['vcirc_kms'], n_streams, axis=0)
    print('test data vcirc_kms shape after tiling: ', test_data_gaia['vcirc_kms'].shape)
    test_data['j'] = test_data_gaia['j'].reshape(-1, 1)
    print('Test data sim shape before augmentation: ', test_data_gaia[cfg.sim_data].shape)
    for aug in augmentations:
        print(f"Applying augmentation: {aug.__name__}")
        test_data_gaia = aug(test_data_gaia)
        # Manually override the vlos error immediately after sample_obs_error is applied
        if aug.__name__ == "sample_obs_error":
            # Match the exact shape of the newly created v_los sigma_errors (batch_size, n_particles)
            target_shape = test_data_gaia["sigma_errors"][:, :, -1].shape
            
            # Reshape the real data and masks to match
            vlos_mask_np = np.array(test_data_gaia["vlos_mask"]).reshape(target_shape)
            vlos_error_np = np.array(test_data_gaia["vlos_error"]).reshape(target_shape)
            
            # Use standard numpy where to safely overwrite
            test_data_gaia["sigma_errors"] = np.array(test_data_gaia["sigma_errors"])
            test_data_gaia["sigma_errors"][:, :, -1] = np.where(
                vlos_mask_np, 
                vlos_error_np, 
                test_data_gaia["sigma_errors"][:, :, -1]
            )
            print("Successfully executed manual override of real v_los errors.")

    test_data_gaia[cfg.sim_data] = test_data_gaia[cfg.sim_data].reshape(-1, len(cfg.target_streams.keys()), test_data_gaia[cfg.sim_data].shape[-2], test_data_gaia[cfg.sim_data].shape[-1])
    test_data_gaia['vcirc_kms'] = test_data_gaia['vcirc_kms'].reshape(-1, len(cfg.target_streams.keys()), test_data_gaia['vcirc_kms'].shape[-2], test_data_gaia['vcirc_kms'].shape[-1])
    test_data_gaia['j'] = test_data_gaia['j'].reshape(-1,len(cfg.target_streams.keys()), 1)




    




    print("Loaded config:", cfg)
    study_name = 'study_DiffusionModel'  # Unique identifier of the study.
    storage_path = "./data/hyperparameter_tuning/gala/rotationcurve/"
    storage_name = JournalStorage(JournalFileStorage(os.path.join(storage_path, "optuna_diffusionmodel_gala_rotationcurve.log")))
    study = optuna.create_study(study_name=study_name, storage=storage_name, directions=['minimize', 'minimize'], load_if_exists=True)
    study.optimize(
        lambda trial: objective(trial, cfg),
        callbacks=[MaxTrialsCallback(300, states=(TrialState.COMPLETE, TrialState.FAIL))],
    )