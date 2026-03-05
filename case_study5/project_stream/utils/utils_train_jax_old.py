import os
import numpy as np
import json
from astropy.io import ascii
import astropy.units as u
import pandas as pd
import jax.numpy as jnp
import jax
from jax import jit
from jax.scipy.stats import gaussian_kde
from functools import partial


def get_priorscore_from_simconfig(params_name, path_to_config):
    pass


def load_npz_as_dict(file):
    with np.load(file) as data:
        return dict(data)


class AugmentationsClass:
    def __init__(self, cfg, random_key=None):
        self.cfg = cfg
        if random_key is None:
            self.key = jax.random.PRNGKey(42)
        else:
            self.key = random_key

        self.idx_to_stream = {v: k for k, v in self.cfg.target_streams.items()}
        self.tbl_data = ascii.read(
            "/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/apjad382dt1_mrt.txt",
            format="cds",
        )
        self.tbl_ids = pd.read_csv(
            "/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/gaia_stream_id.csv",
            sep="\t",
        )
        self.gaia_id = cfg.gaia_id

        # Extract source_id mapping
        self.j_to_source_id = {}
        for name_stream, name_id in self.gaia_id.items():
            source_id = self.tbl_ids.loc[
                self.tbl_ids["Name"] == name_id, "s_ID"
            ].values[0]
            j = self.cfg.target_streams[name_stream]
            self.j_to_source_id[j] = source_id

        # Observed stream magnitudes
        self.observed_streams = {}
        for j, source_id in self.j_to_source_id.items():
            tbl_subset = self.tbl_data[self.tbl_data["Stream"] == source_id]
            self.observed_streams[j] = jnp.array(tbl_subset["Gmag"])

        # Magnitude clipping
        self.magnitude_clipping = {}
        for j, magnitudes in self.observed_streams.items():
            self.magnitude_clipping[j] = (
                float(magnitudes.min()),
                float(magnitudes.max()),
            )

        # Build KDE objects for each stream (1D data -> shape (1, n))
        self.kde_streams = {}
        for j, magnitudes in self.observed_streams.items():
            self.kde_streams[j] = gaussian_kde(magnitudes[None, :])

        # ---- Precompute lookup arrays for JIT ----
        self.n_streams = max(self.idx_to_stream.keys()) + 1

        # Observational window lookup arrays: shape (n_streams,)
        ra_min_list = []
        ra_max_list = []
        dec_min_list = []
        dec_max_list = []
        for idx in range(self.n_streams):
            stream_name = self.idx_to_stream[idx]
            window = self.cfg.observational_window[stream_name]
            ra_min_list.append(window["ra_min"])
            ra_max_list.append(window["ra_max"])
            dec_min_list.append(window["dec_min"])
            dec_max_list.append(window["dec_max"])

        self.ra_min_lookup = jnp.array(ra_min_list)
        self.ra_max_lookup = jnp.array(ra_max_list)
        self.dec_min_lookup = jnp.array(dec_min_list)
        self.dec_max_lookup = jnp.array(dec_max_list)

        # Observed n_stars lookup: shape (n_streams,)
        observed_n_stars_list = []
        for idx in range(self.n_streams):
            stream_name = self.idx_to_stream[idx]
            observed_n_stars_list.append(self.cfg.observed_n_stars[stream_name])
        self.observed_n_stars_lookup = jnp.array(observed_n_stars_list)

        # Magnitude clipping lookup: shape (n_streams,) each
        mag_min_list = []
        mag_max_list = []
        for idx in range(self.n_streams):
            mag_min, mag_max = self.magnitude_clipping[idx]
            mag_min_list.append(mag_min)
            mag_max_list.append(mag_max)
        self.mag_min_lookup = jnp.array(mag_min_list)
        self.mag_max_lookup = jnp.array(mag_max_list)

        # ---- Build KDE resample branch functions for jax.lax.switch ----
        # Each branch takes (key, n_particles) and returns (1, n_particles) samples
        # We build closures that capture the KDE object and clipping bounds
        self._kde_branches = []
        for idx in range(self.n_streams):
            kde = self.kde_streams[idx]
            mag_min = self.mag_min_lookup[idx]
            mag_max = self.mag_max_lookup[idx]

            def make_branch(kde_obj, m_min, m_max):
                def branch_fn(key_and_n):
                    key, n_particles = key_and_n
                    # resample returns (d,) + shape = (1, n_particles)
                    samples = kde_obj.resample(key, shape=(n_particles,))
                    # squeeze the d=1 dimension -> (n_particles,)
                    samples = samples[0]
                    samples = jnp.clip(samples, m_min, m_max)
                    return samples
                return branch_fn

            self._kde_branches.append(make_branch(kde, mag_min, mag_max))

        # ---- Error interpolators ----
        error_file = "/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/gaia_DR3_erorr.txt"
        tbl = ascii.read(error_file, format="tab")
        tbl.remove_column("Unit")

        mag_bins = []
        for colname in tbl.colnames[1:]:
            if "–" in colname or "−" in colname:
                parts = colname.replace("−", "-").replace("–", "-").split("-")
                mag_bins.append((float(parts[0]) + float(parts[1])) / 2.0)
            else:
                mag_bins.append(float(colname))
        self.error_interp_mag_bins = jnp.array(mag_bins)

        # Store raw values for JIT-compatible interpolation
        error_values = {}
        for row in tbl:
            quantity = row["Quantity"]
            values = jnp.array(
                np.array([row[col] for col in tbl.colnames[1:]], dtype=float)
            )
            error_values[quantity] = values

        # Stack error values for JIT: order = [ra, dec, parallax, mu_ra, mu_dec]
        self.error_keys = ["ra", "dec", "parallax", "mu_ra", "mu_dec"]
        self.error_values_stacked = jnp.stack(
            [error_values[k] for k in self.error_keys], axis=0
        )  # shape (5, n_mag_bins)

    # ----------------------------------------------------------------
    # Internal key management
    # ----------------------------------------------------------------

    def _split_key(self):
        """Split the internal PRNG key, update self.key, return a subkey."""
        self.key, subkey = jax.random.split(self.key)
        return subkey

    # ----------------------------------------------------------------
    # Augmentation functions — all take batch, return batch
    # ----------------------------------------------------------------

    def remove_los_velocity(self, batch):
        batch[self.cfg.sim_data] = batch[self.cfg.sim_data][:, :, :5]
        return batch

    def convert_distance_to_parallax(self, batch):
        sim_data = batch[self.cfg.sim_data]
        distances_kpc = sim_data[:, :, 2]
        parallax_mas = 1.0 / distances_kpc
        sim_data[:, :, 2] = parallax_mas
        batch[self.cfg.sim_data] = sim_data
        return batch

    def observational_window(self, batch):
        """
        Mask particles outside the observational window for each stream.
        Sets batch['attention_mask']: (batch_size, 1, n_particles) bool.
        """
        sim_data = batch[self.cfg.sim_data]
        j = batch["j"]
        batch["attention_mask"] = self._observational_window_jit(sim_data, j)
        return batch

    @partial(jit, static_argnums=(0,))
    def _observational_window_jit(self, sim_data, j):
        j_flat = j[:, 0].astype(jnp.int32)

        ra_min = self.ra_min_lookup[j_flat][:, None]
        ra_max = self.ra_max_lookup[j_flat][:, None]
        dec_min = self.dec_min_lookup[j_flat][:, None]
        dec_max = self.dec_max_lookup[j_flat][:, None]

        ra = sim_data[:, :, 0]
        dec = sim_data[:, :, 1]

        mask = (
            (ra >= ra_min) & (ra <= ra_max) &
            (dec >= dec_min) & (dec <= dec_max)
        )
        return mask[:, None, :]

    def subsampling_to_observed_n_stars(self, batch):
        subkey = self._split_key()
        attention_mask = batch["attention_mask"]
        j = batch["j"]
        batch["attention_mask"] = self._subsampling_jit(attention_mask, j, subkey)
        return batch

    @partial(jit, static_argnums=(0,))
    def _subsampling_jit(self, attention_mask, j, key):
        j_flat = j[:, 0].astype(jnp.int32)
        mask = attention_mask[:, 0, :]
        batch_size, n_particles = mask.shape

        max_keep = self.observed_n_stars_lookup[j_flat]

        random_scores = jax.random.uniform(key, shape=(batch_size, n_particles))
        random_scores = jnp.where(mask, random_scores, 2.0)

        sorted_indices = jnp.argsort(random_scores, axis=1)
        rank = jnp.argsort(sorted_indices, axis=1)

        keep = mask & (rank < max_keep[:, None])
        return keep[:, None, :]

    def sample_magnitudes(self, batch):
        """
        Sample magnitudes from KDE for each stream in the batch using
        jax.lax.switch to select the correct KDE per batch entry.
        Sets batch['magnitudes']: (batch_size, n_particles).
        """
        subkey = self._split_key()
        j = batch["j"]
        n_particles = batch[self.cfg.sim_data].shape[1]
        batch["magnitudes"] = self._sample_magnitudes_jit(j, n_particles, subkey)
        return batch

    def _build_kde_branches(self, n_particles):
        """
        Build branch functions for jax.lax.switch with n_particles
        baked in as a Python constant (not traced).
        """
        branches = []
        for idx in range(self.n_streams):
            kde = self.kde_streams[idx]
            mag_min = self.mag_min_lookup[idx]
            mag_max = self.mag_max_lookup[idx]

            def make_branch(kde_obj, m_min, m_max, n):
                def branch_fn(key):
                    # n is a Python int captured in the closure, not traced
                    samples = kde_obj.resample(key, shape=(n,))
                    # squeeze the d=1 dimension -> (n,)
                    samples = samples[0]
                    samples = jnp.clip(samples, m_min, m_max)
                    return samples
                return branch_fn

            branches.append(make_branch(kde, mag_min, mag_max, n_particles))
        return branches

    @partial(jit, static_argnums=(0, 2))
    def _sample_magnitudes_jit(self, j, n_particles, key):
        """
        JIT-compiled magnitude sampling using jax.lax.switch to dispatch
        to the correct stream's gaussian_kde.resample.

        Args:
            j: (batch_size, 1)
            n_particles: int (static)
            key: PRNGKey

        Returns:
            magnitudes: (batch_size, n_particles)
        """
        j_flat = j[:, 0].astype(jnp.int32)
        batch_size = j_flat.shape[0]

        # Build branches with n_particles baked in (it's static so this
        # runs at trace time, not at runtime)
        kde_branches = self._build_kde_branches(n_particles)

        # Generate one key per batch entry
        keys = jax.random.split(key, batch_size)

        def sample_single(j_idx, subkey):
            """Sample n_particles magnitudes for one batch entry using lax.switch."""
            return jax.lax.switch(
                j_idx,
                kde_branches,
                subkey,
            )  # returns (n_particles,)

        # vmap over the batch dimension
        magnitudes = jax.vmap(sample_single)(j_flat, keys)  # (batch_size, n_particles)

        return magnitudes
    
    def sample_obs_error(self, batch):
        subkey = self._split_key()
        magnitudes = batch["magnitudes"]
        batch["obs_errors"], batch["sigma_errors"] = self._sample_obs_error_jit(magnitudes, subkey)
        return batch

    @partial(jit, static_argnums=(0,))
    def _sample_obs_error_jit(self, magnitudes, key):
        mag_bins = self.error_interp_mag_bins
        values = self.error_values_stacked

        def interp_single_quantity(vals):
            return jnp.interp(magnitudes, mag_bins, vals)

        sigmas = jax.vmap(interp_single_quantity)(values)
        
        batch_size, n_particles = magnitudes.shape
        noise = jax.random.normal(key, shape=(5, batch_size, n_particles))

        errors = sigmas * noise
        errors = jnp.transpose(errors, (1, 2, 0))
        sigmas = jnp.transpose(sigmas, (1, 2, 0))

        return errors, sigmas

    def apply_obs_error(self, batch):
        batch[self.cfg.sim_data] = self._apply_obs_error_jit(
            batch[self.cfg.sim_data], batch["obs_errors"]
        )
        return batch

    @partial(jit, static_argnums=(0,))
    def _apply_obs_error_jit(self, sim_data, errors):
        return sim_data.at[:, :, :5].add(errors)

    @partial(jit, static_argnums=(0,))
    def flip_dirz(self, batch):
        dirz = batch['dirz_Triaxial_rotated_halo']
        mask = dirz < 0
        batch['dirz_Triaxial_rotated_halo'] = jnp.where(mask, -dirz, dirz)
        batch['dirx_Triaxial_rotated_halo'] = jnp.where(mask, -batch['dirx_Triaxial_rotated_halo'], batch['dirx_Triaxial_rotated_halo'])
        batch['diry_Triaxial_rotated_halo'] = jnp.where(mask, -batch['diry_Triaxial_rotated_halo'], batch['diry_Triaxial_rotated_halo'])
        return batch
    
    @partial(jit, static_argnums=(0,))
    def concatentate_sigma_error_to_sim_data(self, batch):
        """
        Concatenate the sigma_errors to the sim_data, so that the model can use them as input
        """
        batch[self.cfg.sim_data] = jnp.concatenate([batch[self.cfg.sim_data], batch["sigma_errors"]], axis=-1)
        return batch
    
    @partial(jit, static_argnums=(0,))
    def concatenate_magnitudes_to_sim_data(self, batch):
        """
        Concatenate the magnitudes to the sim_data, so that the model can use them as input
        """
        batch[self.cfg.sim_data] = jnp.concatenate([batch[self.cfg.sim_data], batch["magnitudes"][..., None]], axis=-1)
        return batch
    
    @partial(jit, static_argnums=(0,))
    def concatenate_j_to_sim_data(self, batch):
        """
        Concatenate the j to the sim_data, so that the model can use them as input
        """
        j_expanded = jnp.broadcast_to(batch['j'][:, None, :], (batch[self.cfg.sim_data].shape[0], batch[self.cfg.sim_data].shape[1], 1))
        batch[self.cfg.sim_data] = jnp.concatenate([batch[self.cfg.sim_data], j_expanded], axis=-1)
        return batch