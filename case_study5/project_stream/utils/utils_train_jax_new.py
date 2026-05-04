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
        error_file = "/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/gaia_DR3_erorr_6D.txt"
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
            quantity = row["Quantity"].strip()
            values = jnp.array(
                np.array([row[col] for col in tbl.colnames[1:]], dtype=float)
            )
            error_values[quantity] = values

        # Default behavior is to use only 5 dimensional errors, but allow config override
        self.error_keys = ["ra", "dec", "parallax", "mu_ra", "mu_dec"] if cfg.error_keys is None else cfg.error_keys
        self.error_values_stacked = jnp.stack(
            [error_values[k] for k in self.error_keys], axis=0
        )  # shape (5/6, n_mag_bins)
        # self.error_values_stacked = self.error_values_stacked.at[:2].set(
        #     self.error_values_stacked[:2] * u.mas.to(u.deg)
        # )
        # print("Error values stacked shape:", self.error_values_stacked)
        # v_los error values for mask_vlos interpolation
        self.vlos_median_std = error_values["v_los"]       # shape (n_mag_bins,)
        self.vlos_std_of_std = error_values["std_v_los"]   # shape (n_mag_bins,)
        
        #minimum number of stars with v_los information for each stream
        min_star_with_vlos = []
        for idx in range(self.n_streams):
            stream_name = self.idx_to_stream[idx]
            min_star_with_vlos.append(self.cfg.min_star_with_vlos[stream_name])
        self.min_star_with_vlos = jnp.array(min_star_with_vlos)

        # ---- Masked v_los values lookup ----
        vlos_mean_list = []
        vlos_std_list = []
        for idx in range(self.n_streams):
            stream_name = self.idx_to_stream[idx]
            vlos_mean_list.append(self.cfg.masked_value_vlos[stream_name]["mean"])
            vlos_std_list.append(self.cfg.masked_value_vlos[stream_name]["std"])
        self.vlos_mean_lookup = jnp.array(vlos_mean_list)
        self.vlos_std_lookup = jnp.array(vlos_std_list)


        #new observational windows using the track
        from scipy.interpolate import UnivariateSpline

        N_DENSE = 500  # dense enough for smooth interpolation, cheap to store

        track_ra_dense  = []
        track_dec_dense = []
        track_width_deg = []
        track_ra_min    = []
        track_ra_max    = []

        for idx in range(self.n_streams):
            stream_name = self.idx_to_stream[idx]
            track  = np.load(f"/export/home/vgiusepp/diffusion-experiments/case_study5/project_stream/data/{stream_name}_track.npz")

            ra  = track['ra']
            dec = track['dec']
            order = np.argsort(ra)
            ra, dec = ra[order], dec[order]

            spl      = UnivariateSpline(ra, dec, s=0, k=3)
            ra_dense = np.linspace(ra.min(), ra.max(), N_DENSE)
            dec_dense = spl(ra_dense)

            # 5-sigma width from observed stream members
            source_id  = self.j_to_source_id[idx]
            tbl_subset = self.tbl_data[self.tbl_data["Stream"] == source_id]
            obs_ra     = np.asarray(tbl_subset["RAdeg"], dtype=float)
            obs_dec    = np.asarray(tbl_subset["DEdeg"], dtype=float)
            in_ra      = (obs_ra >= ra.min()) & (obs_ra <= ra.max())
            residuals  = obs_dec[in_ra] - spl(obs_ra[in_ra])
            width_deg  = 5.0 * float(np.std(residuals))

            track_ra_dense.append(ra_dense)
            track_dec_dense.append(dec_dense)
            track_width_deg.append(width_deg)
            track_ra_min.append(ra.min())
            track_ra_max.append(ra.max())

        self.track_ra_dense  = jnp.array(track_ra_dense)   # (n_streams, N_DENSE)
        self.track_dec_dense = jnp.array(track_dec_dense)  # (n_streams, N_DENSE)
        self.track_width_deg = jnp.array(track_width_deg)  # (n_streams,)
        self.track_ra_min    = jnp.array(track_ra_min)     # (n_streams,)
        self.track_ra_max    = jnp.array(track_ra_max)     # (n_streams,)

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

    # @partial(jit, static_argnums=(0,))
    # def cut_to_300_particles(self, batch):
    #     """
    #     Randomly sample N particles for each batch entry.
    #     """
    #     subkey = self._split_key()
    #     sim_data = batch[self.cfg.sim_data]
    #     batch_size, n_particles, *rest = sim_data.shape

    #     # Sample indices for each batch entry
    #     # keys = jax.random.split(subkey, batch_size)
    #     # idx = jax.vmap(
    #     #     lambda k: jax.random.choice(k, n_particles, shape=(200,), replace=False)
    #     # )(keys)
    #     keys = jax.random.split(subkey, batch_size)
    #     idx = jax.vmap(lambda k: jax.random.permutation(k, n_particles)[:300])(keys)

    #     # Gather the selected particles for each batch entry
    #     batch[self.cfg.sim_data] = np.take_along_axis(
    #         sim_data, idx[..., None], axis=1
    #     )

    #     return batch

    # def cut_to_300_particles(self, batch):
    #     subkey = self._split_key()
    #     sim_data = batch[self.cfg.sim_data]
    #     batch_size, n_particles, _ = sim_data.shape
    #     keys = jax.random.split(subkey, batch_size)

    #     idx = jax.vmap(
    #         lambda k: jax.random.choice(k, n_particles, shape=(300,), replace=False)
    #     )(keys)  # (batch_size, 300)

    #     batch[self.cfg.sim_data] = jnp.take_along_axis(sim_data, idx[..., None], axis=1)
    #     return batch

    # @partial(jit, static_argnums=(0,))
    def cut_to_300_particles(self, batch):
        subkey = self._split_key()
        sim_data = batch[self.cfg.sim_data]
        batch_size, n_particles, _ = sim_data.shape

        # Single (batch_size, n_particles) random matrix -> one batched sort kernel
        random_scores = jax.random.uniform(subkey, shape=(batch_size, n_particles))
        idx = jnp.argsort(random_scores, axis=1)[:, :300]  # (batch_size, 300)

        batch[self.cfg.sim_data] = jnp.take_along_axis(sim_data, idx[..., None], axis=1)
        # print(f"cut_to_300_particles: reduced from {n_particles} to 300 particles")
        return batch
    
    @partial(jit, static_argnums=(0,))
    def remove_los_velocity(self, batch):
        batch[self.cfg.sim_data] = batch[self.cfg.sim_data][:, :, :5]
        return batch

    # def convert_distance_to_parallax(self, batch):
    #     sim_data = batch[self.cfg.sim_data]
    #     distances_kpc = sim_data[:, :, 2]
    #     parallax_mas = 1.0 / distances_kpc
    #     sim_data[:, :, 2] = parallax_mas
    #     batch[self.cfg.sim_data] = sim_data
    #     return batch
    
    def compact_to_attended(self, batch):
        """
        After attention_mask is finalised (window + subsampling), reorder
        sim_data so attended particles come first, then slice to max_particles.
        This replaces cut_to_300_particles and must run after observed_n_stars.
        """
        sim_data = batch[self.cfg.sim_data]
        attention_mask = batch["attention_mask"]  # (batch_size, 1, n_particles)
        batch[self.cfg.sim_data], batch["attention_mask"] = self._compact_jit(
            sim_data, attention_mask
        )
        return batch

    @partial(jit, static_argnums=(0,))
    def _compact_jit(self, sim_data, attention_mask):
        mask = attention_mask[:, 0, :]          # (batch_size, n_particles)  bool
        batch_size, n_particles, n_features = sim_data.shape

        # ~mask: attended=False(0), unattended=True(1)
        # argsort ascending => attended indices (0) come first
        sort_order = jnp.argsort(~mask, axis=1)          # (batch_size, n_particles)

        sim_data_sorted = jnp.take_along_axis(
            sim_data, sort_order[..., None], axis=1
        )
        mask_sorted = jnp.take_along_axis(mask, sort_order, axis=1)

        # Slice: your largest stream is M68 with 297 stars, so 300 is safe
        max_p = 300
        sim_data_out = sim_data_sorted[:, :max_p, :]
        mask_out     = mask_sorted[:, :max_p]

        return sim_data_out, mask_out[:, None, :]
    
    @partial(jit, static_argnums=(0,))
    def convert_distance_to_parallax(self, batch):
        sim_data = batch[self.cfg.sim_data]
        distances_kpc = sim_data[:, :, 2]
        parallax_mas = 1.0 / distances_kpc
        sim_data = sim_data.at[:, :, 2].set(parallax_mas)
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
    
    def observational_window_spline(self, batch):
        sim_data = batch[self.cfg.sim_data]
        j        = batch["j"]
        batch["attention_mask"] = self._observational_window_spline_jit(sim_data, j)
        return batch

    @partial(jit, static_argnums=(0,))
    def _observational_window_spline_jit(self, sim_data, j):
        j_flat = j[:, 0].astype(jnp.int32)

        ra  = sim_data[:, :, 0]  # (batch_size, n_particles)
        dec = sim_data[:, :, 1]

        # Index precomputed dense tracks by stream
        ra_dense  = self.track_ra_dense[j_flat]   # (batch_size, N_DENSE)
        dec_dense = self.track_dec_dense[j_flat]  # (batch_size, N_DENSE)
        width     = self.track_width_deg[j_flat]  # (batch_size,)
        ra_min    = self.track_ra_min[j_flat]     # (batch_size,)
        ra_max    = self.track_ra_max[j_flat]     # (batch_size,)

        # Interpolate track dec at each particle's RA — vmap over batch
        dec_center = jax.vmap(jnp.interp)(ra, ra_dense, dec_dense)  # (batch_size, n_particles)

        in_ra  = (ra  >= ra_min[:, None]) & (ra  <= ra_max[:, None])
        in_dec = jnp.abs(dec - dec_center) < width[:, None]

        return (in_ra & in_dec)[:, None, :]

    def observational_window_random(self, batch):
        """
        Create attention_mask by randomly selecting observed_n_stars particles
        from all available particles (no RA/Dec filtering).
        Must run after cut_to_300_particles.
        """
        subkey = self._split_key()
        sim_data = batch[self.cfg.sim_data]
        j = batch["j"]
        batch["attention_mask"] = self._observational_window_random_jit(sim_data, j, subkey)
        return batch

    @partial(jit, static_argnums=(0,))
    def _observational_window_random_jit(self, sim_data, j, key):
        j_flat = j[:, 0].astype(jnp.int32)
        batch_size, n_particles, _ = sim_data.shape

        max_keep = self.observed_n_stars_lookup[j_flat]  # (batch_size,)

        # All particles are candidates — uniform scores over all
        random_scores = jax.random.uniform(key, shape=(batch_size, n_particles))

        # Keep the max_keep lowest scores per row
        sorted_scores = jnp.sort(random_scores, axis=1)
        threshold = sorted_scores[jnp.arange(batch_size), max_keep - 1]  # (batch_size,)
        mask = random_scores <= threshold[:, None]  # (batch_size, n_particles)

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

        #old
        # sorted_indices = jnp.argsort(random_scores, axis=1)
        # rank = jnp.argsort(sorted_indices, axis=1)
        # keep = mask & (rank < max_keep[:, None])

        # One sort to find the threshold score at position max_keep
        sorted_scores = jnp.sort(random_scores, axis=1)  # (batch_size, n_particles)
        # Index the max_keep-th score per row (variable per row, so use advanced indexing)
        threshold = sorted_scores[jnp.arange(batch_size), max_keep - 1]  # (batch_size,)
        keep = mask & (random_scores <= threshold[:, None])
        return keep[:, None, :]
    
    def mask_vlos(self, batch):
        subkey = self._split_key()
        j = batch["j"]
        attention_mask = batch["attention_mask"]
        sim_data = batch[self.cfg.sim_data]
        sigma_errors = batch["sigma_errors"]

        sim_data, sigma_errors, vlos_mask = self._mask_vlos_jit(
            sim_data, sigma_errors, attention_mask, j, subkey
        )
        batch[self.cfg.sim_data] = sim_data
        batch["sigma_errors"] = sigma_errors
        batch["vlos_mask"] = vlos_mask
        return batch

    @partial(jit, static_argnums=(0,))
    def _mask_vlos_jit(self, sim_data, sigma_errors, attention_mask, j, key):
        j_flat = j[:, 0].astype(jnp.int32)
        mask = attention_mask[:, 0, :]          # (batch_size, n_particles)
        batch_size, n_particles = mask.shape

        # --- Decide which attended stars keep their real v_los ---
        n_keep_vlos = self.min_star_with_vlos[j_flat]   # (batch_size,)

        random_scores = jax.random.uniform(key, shape=(batch_size, n_particles))
        random_scores = jnp.where(mask, random_scores, 2.0)    # push unattended out

        sorted_scores = jnp.sort(random_scores, axis=1)
        threshold = sorted_scores[jnp.arange(batch_size), n_keep_vlos - 1]
        vlos_mask = mask & (random_scores <= threshold[:, None])  # (batch_size, n_particles)

        # --- v_los in sim_data ---
        # vlos_mask=True:  keep the already-noised simulated v_los  (apply_obs_error did this)
        # vlos_mask=False: fix to the stream's prior mean
        vlos_mean = self.vlos_mean_lookup[j_flat]               # (batch_size,)
        sim_data_vlos = jnp.where(vlos_mask, sim_data[:, :, -1], vlos_mean[:, None])
        sim_data = sim_data.at[:, :, -1].set(sim_data_vlos)

        # --- sigma_errors for v_los ---
        # vlos_mask=True:  keep the magnitude-interpolated sigma  (sample_obs_error did this)
        # vlos_mask=False: fix to the stream's prior std
        vlos_std = self.vlos_std_lookup[j_flat]                 # (batch_size,)
        sigma_vlos = jnp.where(vlos_mask, sigma_errors[:, :, -1], vlos_std[:, None])
        sigma_errors = sigma_errors.at[:, :, -1].set(sigma_vlos)

        return sim_data, sigma_errors, vlos_mask[:, None, :]

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
        noise = jax.random.normal(key, shape=(len(self.error_keys), batch_size, n_particles))

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
        return sim_data.at[:, :, :len(self.error_keys)].add(errors)
    

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

    @partial(jit, static_argnums=(0,))
    def concatenate_vlos_mask_to_sim_data(self, batch):
        """
        Concatenate the vlos_mask to the sim_data, so that the model can use it as input.
        vlos_mask shape: (batch_size, 1, n_particles) -> transpose to (batch_size, n_particles, 1)
        """
        vlos_mask = batch["vlos_mask"].transpose(0, 2, 1).astype(batch[self.cfg.sim_data].dtype)
        batch[self.cfg.sim_data] = jnp.concatenate([batch[self.cfg.sim_data], vlos_mask], axis=-1)
        return batch

    #used only for real observation were the spettroscopi error (which we will pass a vlos_mask) is known
    @partial(jit, static_argnums=(0,))
    def override_vlos_error_with_real(self, batch):
        """
        Override the last dimension of sigma_errors (v_los error) with the
        real observed vlos_error from the dataset, where vlos_mask == 1.
        Where vlos_mask == 0 (no real v_los), keep the sampled sigma_error.

        Expects:
          batch["sigma_errors"]: (batch_size, n_particles, n_error_keys)
          batch["vlos_error"]:   (batch_size, n_particles, 1)
          batch["vlos_mask"]:    (batch_size, n_particles, 1)  — 1 where real v_los exists
        """
        sigma_errors = batch["sigma_errors"]
        vlos_error = batch["vlos_error"][..., 0]    # (batch_size, n_particles)
        vlos_mask = batch["vlos_mask"][..., 0]      # (batch_size, n_particles)

        sigma_vlos = sigma_errors[:, :, -1]         # (batch_size, n_particles)
        sigma_vlos = jnp.where(vlos_mask, vlos_error, sigma_vlos)
        sigma_errors = sigma_errors.at[:, :, -1].set(sigma_vlos)

        batch["sigma_errors"] = sigma_errors
        return batch
    
    
    