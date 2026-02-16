import os
import numpy as np
import json
from astropy.io import ascii
import astropy.units as u
import pandas as pd
# from scipy.stats import gaussian_kde
from jax.scipy.stats import gaussian_kde
# from scipy.interpolate import interp1d
from interpax import Interpolator1D as interp1d




def get_priorscore_from_simconfig(params_name, path_to_config):
    pass

def load_npz_as_dict(file):
    with np.load(file) as data:
        return dict(data)


class AugmentationsClass:
    def __init__(self, cfg):
        self.cfg = cfg
        self.idx_to_stream = {v: k for k, v in self.cfg.target_streams.items()} #flip between stream name and j index for easy access in the augmentations
        self.tbl_data = ascii.read( "/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/apjad382dt1_mrt.txt", format="cds")
        self.tbl_ids = pd.read_csv("/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/gaia_stream_id.csv", sep='\t')
        self.gaia_id = cfg.gaia_id #key are the str of stream, and values are the str used in the tbl

        #we need to extract the corresponding source_id from the tbl_id and pair it with 'j'
        self.j_to_source_id = {}
        for name_stream, name_id in self.gaia_id.items():
            source_id = self.tbl_ids.loc[self.tbl_ids['Name'] == name_id, 's_ID'].values[0]
            j = self.cfg.target_streams[name_stream]
            self.j_to_source_id[j] = source_id
        
        #we take the magnitudes of the stars in each stream
        self.observed_streams = {}
        for j, source_id in self.j_to_source_id.items():
            tbl_subset = self.tbl_data[self.tbl_data['Stream'] == source_id]
            self.observed_streams[j] = jnp.array(tbl_subset['Gmag'])
        
        #let's store also the clipping value for max and min magnitude for each stream, to use in the augmentation
        self.magnitude_clipping = {}
        for j, magnitudes in self.observed_streams.items():
            self.magnitude_clipping[j] = (magnitudes.min(), magnitudes.max())
        
        #we construct a kde of the magnitude for each of the stream
        self.kde_streams = {}
        for j, magnitudes in self.observed_streams.items():
            self.kde_streams[j] = gaussian_kde(magnitudes)


        #Create the interpolation function for errros:
        error_file = "/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/gaia_DR3_erorr.txt"
        # Read with astropy
        tbl = ascii.read(error_file, format='tab')
        # Remove the Unit column
        tbl.remove_column('Unit')
        # Extract magnitude bins from column names (skip 'Quantity')
        mag_bins = []
        for colname in tbl.colnames[1:]:  # Skip 'Quantity' column
            if '–' in colname or '−' in colname:
                # Handle range like "9–12"
                parts = colname.replace('−', '-').replace('–', '-').split('-')
                mag_bins.append((float(parts[0]) + float(parts[1])) / 2.0)
            else:
                mag_bins.append(float(colname))
        mag_bins = jnp.array(mag_bins)
        
        # print("Magnitude bins for interpolation:", mag_bins)        
        # Create interpolators for each quantity
        self.error_interpolators = {}
        
        for row in tbl:
            quantity = row['Quantity']
            # Extract error values (all columns except 'Quantity')
            values = jnp.array(np.array([row[col] for col in tbl.colnames[1:]], dtype=float))
            # print('Row:', quantity, 'Values:', values)  # Debug print
            
            # Create interpolator with simplified key names
            self.error_interpolators[quantity] = interp1d(
                mag_bins, values,
                method='linear',
                # fill_value='extrapolate'
            )

    def remove_los_velocity(self, batch):
        batch[self.cfg.sim_data] = batch[self.cfg.sim_data][:, :, :5]  # Keep only ra, dec, distance, pmra, pmdec
        return batch

    def convert_distance_to_parallax(self, batch):
        sim_data = batch[self.cfg.sim_data]  # shape (batch_size, n_particles, 6)
        distances = sim_data[:, :, 2] * u.kpc
        parallax = distances.to(u.mas, equivalencies=u.parallax())
        batch[self.cfg.sim_data][:, :, 2] = parallax.value
        return batch

    def observational_window(self, batch,):
        """
        Vectorized: Mask particles outside the observational window for each stream in the batch.
        Returns attention_mask: (batch_size, 1, n_particles) with False where particles are outside the window.
        """
        sim_data = batch[self.cfg.sim_data]  # shape (batch_size, n_particles, 5/6)
        j = batch['j']                       # shape (batch_size, 1)

        # Build arrays of window limits for each batch entry
        ra_min = np.array([self.cfg.observational_window[self.idx_to_stream[int(jj[0])]]['ra_min'] for jj in j])
        ra_max = np.array([self.cfg.observational_window[self.idx_to_stream[int(jj[0])]]['ra_max'] for jj in j])
        dec_min = np.array([self.cfg.observational_window[self.idx_to_stream[int(jj[0])]]['dec_min'] for jj in j])
        dec_max = np.array([self.cfg.observational_window[self.idx_to_stream[int(jj[0])]]['dec_max'] for jj in j])

        # Expand dims for broadcasting: (batch_size, 1)
        ra_min = ra_min[:, None]
        ra_max = ra_max[:, None]
        dec_min = dec_min[:, None]
        dec_max = dec_max[:, None]

        # Extract ra, dec: (batch_size, n_particles)
        ra = sim_data[..., 0]
        dec = sim_data[..., 1]

        # Vectorized mask: (batch_size, n_particles)
        mask = (
            (ra >= ra_min) & (ra <= ra_max) &
            (dec >= dec_min) & (dec <= dec_max)
        )

        # Add singleton dimension for compatibility: (batch_size, 1, n_particles)
        batch['attention_mask'] = mask[:, None, :]

        return batch
    
    def subsampling_to_observed_n_stars(self, batch):
        """
        For each batch entry, subsample the True entries in attention_mask
        to at most observed_n_stars for that stream.
        If fewer stars are available, keep all.
        """
        attention_mask = batch['attention_mask']  # shape (batch_size, 1, n_particles)
        j = batch['j']                            # shape (batch_size, 1)
        batch_size, _, n_particles = attention_mask.shape

        # Get observed_n_stars for each batch entry: (batch_size,)
        observed_n_stars = np.array([
            self.cfg.observed_n_stars[self.idx_to_stream[int(jj[0])]] for jj in j
        ])

        # Remove singleton: (batch_size, n_particles)
        mask = attention_mask[:, 0, :].astype(bool).copy()

        # Count how many True per batch entry: (batch_size,)
        n_true = mask.sum(axis=1)

        # How many to turn off per batch entry (0 if no subsampling needed)
        n_excess = np.maximum(n_true - observed_n_stars, 0)  # (batch_size,)

        # Assign random scores to True positions, +inf to False (so False sorts last)
        random_scores = np.full((batch_size, n_particles), np.inf)
        random_scores[mask] = np.random.random(mask.sum())

        # Sort ascending: lowest random scores first (True entries come first)
        sorted_indices = np.argsort(random_scores, axis=1)  # (batch_size, n_particles)

        # Compute cumulative rank per row
        rank = np.argsort(sorted_indices, axis=1)  # (batch_size, n_particles)

        # For each entry, keep only the first (n_true - n_excess) = min(n_true, observed_n_stars) True entries
        keep_count = (n_true - n_excess)[:, None]  # (batch_size, 1)

        # Turn off entries whose rank >= keep_count AND were originally True
        turn_off = mask & (rank >= keep_count)
        mask[turn_off] = False

        # Restore singleton dimension: (batch_size, 1, n_particles)
        batch['attention_mask'] = mask[:, None, :].astype(attention_mask.dtype)
        return batch

    def sample_magnitudes(self, batch):
        """
        Sample magnitudes for each of the stream, based on the kde of the observed magnitude in the stream
        """
        batch_size, n_particles, _ = batch[self.cfg.sim_data].shape
        j = batch['j'].reshape(-1)  # shape (batch_size,)
        unique_j, unique_counts = np.unique(j, return_counts=True)
        magnitudes = np.zeros((batch_size, n_particles))
        # print("unique_j:", unique_j)

        # print("unique_counts:", unique_counts)

        for jj, count in zip(unique_j, unique_counts):
            kde = self.kde_streams[int(jj)]
            samples_size = (count * n_particles) #later most of them will be masked out
            sampled_magnitudes = kde.resample(size=samples_size).reshape(count, n_particles)
            # Clip magnitudes to the observed range for this stream
            mag_min, mag_max = self.magnitude_clipping[int(jj)]
            sampled_magnitudes = np.clip(sampled_magnitudes, mag_min, mag_max)
            magnitudes[j == jj] = sampled_magnitudes
        batch["magnitudes"] = magnitudes
        return batch

    def sample_obs_error(self, batch):
        """
        Sample the observational error for each of the stream, based on the sampled magnitudes
        """
        """Load Gaia DR3 error table and create interpolation functions."""
        magnitudes = batch["magnitudes"]  # shape (batch_size, n_particles)        
        errors = np.zeros_like(batch[self.cfg.sim_data])  
        
        # Get interpolated sigmas
        sigma_ra = self.error_interpolators['ra'](magnitudes)
        sigma_dec = self.error_interpolators['dec'](magnitudes)
        sigma_parallax = self.error_interpolators['parallax'](magnitudes)
        sigma_pmra = self.error_interpolators['mu_ra'](magnitudes)
        sigma_pmdec = self.error_interpolators['mu_dec'](magnitudes)
        
        # Sample errors
        errors[..., 0] = np.random.normal(0, sigma_ra)
        errors[..., 1] = np.random.normal(0, sigma_dec)
        errors[..., 2] = np.random.normal(0, sigma_parallax)
        errors[..., 3] = np.random.normal(0, sigma_pmra)
        errors[..., 4] = np.random.normal(0, sigma_pmdec)
        
        batch["obs_errors"] = errors
        return batch

    def apply_obs_error(self, batch):
        """Apply the sampled observational error to the sim_data
        """
        # if self.cfg.test:
        #     import matplotlib.pyplot as plt
        #     fig_radec = plt.figure(figsize=(10, 5))
        #     ax1 = fig_radec.add_subplot(1, 3, 1)
        #     ax2 = fig_radec.add_subplot(1, 3, 2)
        #     ax3 = fig_radec.add_subplot(1, 3, 3)
        #     sim_data = batch[self.cfg.sim_data]
        #     j = batch['j'].reshape(-1)
        #     ax1.scatter(sim_data[j==0, :, 0], sim_data[j==0, :, 1], alpha=0.5, label='Before error', c='blue')
        #     ax2.scatter(sim_data[j==1, :, 0], sim_data[j==1, :, 1], alpha=0.5, label='Before error', c='blue')
        #     ax3.scatter(sim_data[j==2, :, 0], sim_data[j==2, :, 1], alpha=0.5, label='Before error', c='blue')
        #     fig_pm = plt.figure(figsize=(10, 5))
        #     ax_pm1 = fig_pm.add_subplot(1, 3, 1)
        #     ax_pm2 = fig_pm.add_subplot(1, 3, 2)
        #     ax_pm3 = fig_pm.add_subplot(1, 3, 3)
        #     ax_pm1.scatter(sim_data[j==0, :, 3], sim_data[j==0, :, 4], alpha=0.5, label='Before error', c='blue')
        #     ax_pm2.scatter(sim_data[j==1, :, 3], sim_data[j==1, :, 4], alpha=0.5, label='Before error', c='blue')
        #     ax_pm3.scatter(sim_data[j==2, :, 3], sim_data[j==2, :, 4], alpha=0.5, label='Before error', c='blue')   
        #     fig_ra_parallax = plt.figure(figsize=(10, 5))
        #     ax_rp1 = fig_ra_parallax.add_subplot(1, 3, 1)
        #     ax_rp2 = fig_ra_parallax.add_subplot(1, 3, 2)
        #     ax_rp3 = fig_ra_parallax.add_subplot(1, 3, 3)
        #     ax_rp1.scatter(sim_data[j==0, :, 0], sim_data[j==0, :, 2], alpha=0.5, label='Before error', c='blue')
        #     ax_rp2.scatter(sim_data[j==1, :, 0], sim_data[j==1, :, 2], alpha=0.5, label='Before error', c='blue')
        #     ax_rp3.scatter(sim_data[j==2, :, 0], sim_data[j==2, :, 2], alpha=0.5, label='Before error', c='blue')
        batch[self.cfg.sim_data] += batch["obs_errors"]
        # if self.cfg.test:
        #     sim_data = batch[self.cfg.sim_data]
        #     ax1.scatter(sim_data[j==0, :, 0], sim_data[j==0, :, 1], alpha=0.5, label='After error', c='red')
        #     ax1.set_title(self.idx_to_stream[0])
        #     ax2.scatter(sim_data[j==1, :, 0], sim_data[j==1, :, 1], alpha=0.5, label='After error', c='red')
        #     ax2.set_title(self.idx_to_stream[1])
        #     ax3.scatter(sim_data[j==2, :, 0], sim_data[j==2, :, 1], alpha=0.5, label='After error', c='red')
        #     ax3.set_title(self.idx_to_stream[2])
        #     ax1.legend()
        #     ax2.legend()
        #     ax3.legend()
        #     plt.show()
        #     fig_radec.savefig(os.path.join(self.cfg.base_dir, self.cfg.results_dir, 'apply_obs_error_test_radec.pdf'))
        #     ax_pm1.scatter(sim_data[j==0, :, 3], sim_data[j==0, :, 4], alpha=0.5, label='After error', c='red')
        #     ax_pm1.set_title(self.idx_to_stream[0])
        #     ax_pm2.scatter(sim_data[j==1, :, 3], sim_data[j==1, :, 4], alpha=0.5, label='After error', c='red')
        #     ax_pm2.set_title(self.idx_to_stream[1])
        #     ax_pm3.scatter(sim_data[j==2, :, 3], sim_data[j==2, :, 4], alpha=0.5, label='After error', c='red')
        #     ax_pm3.set_title(self.idx_to_stream[2])
        #     ax_pm1.legend()
        #     ax_pm2.legend()
        #     ax_pm3.legend()
        #     fig_pm.savefig(os.path.join(self.cfg.base_dir, self.cfg.results_dir, 'apply_obs_error_pm_test.pdf'))
        #     ax_rp1.scatter(sim_data[j==0, :, 0], sim_data[j==0, :, 2], alpha=0.5, label='After error', c='red')
        #     ax_rp1.set_title(self.idx_to_stream[0])
        #     ax_rp2.scatter(sim_data[j==1, :, 0], sim_data[j==1, :, 2], alpha=0.5, label='After error', c='red')
        #     ax_rp2.set_title(self.idx_to_stream[1])
        #     ax_rp3.scatter(sim_data[j==2, :, 0], sim_data[j==2, :, 2], alpha=0.5, label='After error', c='red')
        #     ax_rp3.set_title(self.idx_to_stream[2])
        #     ax_rp1.legend()
        #     ax_rp2.legend()
        #     ax_rp3.legend()
        #     fig_ra_parallax.savefig(os.path.join(self.cfg.base_dir, self.cfg.results_dir, 'apply_obs_error_ra_parallax_test.pdf'))
        return batch
        
        
        


