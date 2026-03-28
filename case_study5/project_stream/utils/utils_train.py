import os
import numpy as np
import json
from astropy.io import ascii
import astropy.units as u
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d


def get_priorscore_from_simconfig(params_name, path_to_config):
    pass


def load_npz_as_dict(file):
    with np.load(file) as data:
        return dict(data)


class AugmentationsClass:
    def __init__(self, cfg, seed=42):
        self.cfg = cfg

        # Stateful RNG — use rng.normal(...), rng.uniform(...), etc.
        # The Generator updates its internal state after every call,
        # so sampling order must stay consistent for reproducibility.
        self.rng = np.random.default_rng(seed)

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

        # Map stream index j -> source_id
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
            self.observed_streams[j] = np.array(tbl_subset["Gmag"])

        # Magnitude clipping bounds per stream
        self.magnitude_clipping = {}
        for j, magnitudes in self.observed_streams.items():
            self.magnitude_clipping[j] = (float(magnitudes.min()), float(magnitudes.max()))

        # KDE of observed magnitudes per stream
        self.kde_streams = {}
        for j, magnitudes in self.observed_streams.items():
            self.kde_streams[j] = gaussian_kde(magnitudes)

        # ---- Precompute lookup arrays ----
        self.n_streams = max(self.idx_to_stream.keys()) + 1

        ra_min_list, ra_max_list, dec_min_list, dec_max_list = [], [], [], []
        for idx in range(self.n_streams):
            stream_name = self.idx_to_stream[idx]
            window = self.cfg.observational_window[stream_name]
            ra_min_list.append(window["ra_min"])
            ra_max_list.append(window["ra_max"])
            dec_min_list.append(window["dec_min"])
            dec_max_list.append(window["dec_max"])

        self.ra_min_lookup  = np.array(ra_min_list)
        self.ra_max_lookup  = np.array(ra_max_list)
        self.dec_min_lookup = np.array(dec_min_list)
        self.dec_max_lookup = np.array(dec_max_list)

        # Observed n_stars per stream
        observed_n_stars_list = []
        for idx in range(self.n_streams):
            stream_name = self.idx_to_stream[idx]
            observed_n_stars_list.append(self.cfg.observed_n_stars[stream_name])
        self.observed_n_stars_lookup = np.array(observed_n_stars_list)

        # Magnitude clipping lookup arrays
        mag_min_list, mag_max_list = [], []
        for idx in range(self.n_streams):
            mag_min, mag_max = self.magnitude_clipping[idx]
            mag_min_list.append(mag_min)
            mag_max_list.append(mag_max)
        self.mag_min_lookup = np.array(mag_min_list)
        self.mag_max_lookup = np.array(mag_max_list)

        # ---- Error interpolators (unified 6D table) ----
        error_file = (
            "/export/home/vgiusepp/diffusion-experiments/case_study5/"
            "project_stream/data/gaia_DR3_erorr_6D.txt"
        )
        tbl = ascii.read(error_file, format="tab")
        tbl.remove_column("Unit")

        mag_bins = []
        for colname in tbl.colnames[1:]:
            if "–" in colname or "−" in colname:
                parts = colname.replace("−", "-").replace("–", "-").split("-")
                mag_bins.append((float(parts[0]) + float(parts[1])) / 2.0)
            else:
                mag_bins.append(float(colname))
        self.error_interp_mag_bins = np.array(mag_bins)

        # Build one interpolator per quantity
        self.error_interpolators = {}
        for row in tbl:
            quantity = row["Quantity"].strip()
            values = np.array([row[col] for col in tbl.colnames[1:]], dtype=float)
            self.error_interpolators[quantity] = interp1d(
                self.error_interp_mag_bins,
                values,
                kind="linear",
                fill_value="extrapolate",
            )

        # Which error dimensions to use (default: 5D; override via cfg.error_keys)
        self.error_keys = (
            ["ra", "dec", "parallax", "mu_ra", "mu_dec"]
            if cfg.error_keys is None
            else cfg.error_keys
        )

        # Stack sigma values for vectorised interpolation: shape (n_keys, n_mag_bins)
        self.error_values_stacked = np.stack(
            [self.error_interpolators[k].y for k in self.error_keys], axis=0
        )

        # v_los spread statistics for mask_vlos
        self.vlos_median_std = self.error_interpolators["v_los"].y      # (n_mag_bins,)
        self.vlos_std_of_std = self.error_interpolators["std_v_los"].y  # (n_mag_bins,)

        # Minimum stars with v_los per stream
        min_star_with_vlos_list = []
        for idx in range(self.n_streams):
            stream_name = self.idx_to_stream[idx]
            min_star_with_vlos_list.append(self.cfg.min_star_with_vlos[stream_name])
        self.min_star_with_vlos = np.array(min_star_with_vlos_list)

        # Masked v_los mean and std per stream
        vlos_mean_list, vlos_std_list = [], []
        for idx in range(self.n_streams):
            stream_name = self.idx_to_stream[idx]
            vlos_mean_list.append(self.cfg.masked_value_vlos[stream_name]["mean"])
            vlos_std_list.append(self.cfg.masked_value_vlos[stream_name]["std"])
        self.vlos_mean_lookup = np.array(vlos_mean_list)
        self.vlos_std_lookup  = np.array(vlos_std_list)

    # ----------------------------------------------------------------
    # Augmentation methods — each takes batch dict, returns batch dict
    # ----------------------------------------------------------------

    def cut_to_300_particles(self, batch):
        """
        Randomly subsample to 300 particles per batch entry (without replacement).
        """
        sim_data = batch[self.cfg.sim_data]          # (batch_size, n_particles, d)
        batch_size, n_particles, *rest = sim_data.shape

        # Build index array: for each batch entry, a random permutation of
        # particle indices, keeping the first 300.
        idx = np.stack(
            [self.rng.permutation(n_particles)[:300] for _ in range(batch_size)],
            axis=0,
        )  # (batch_size, 300)

        # Gather selected particles
        batch[self.cfg.sim_data] = np.take_along_axis(
            sim_data, idx[..., None], axis=1
        )
        return batch

    def remove_los_velocity(self, batch):
        """Keep only the first 5 phase-space dimensions (drop v_los)."""
        batch[self.cfg.sim_data] = batch[self.cfg.sim_data][:, :, :5]
        return batch

    def convert_distance_to_parallax(self, batch):
        """
        Convert distance [kpc] to parallax [mas] in-place.
        parallax [mas] = 1 / distance [kpc]
        """
        sim_data = batch[self.cfg.sim_data]
        sim_data[:, :, 2] = 1.0 / sim_data[:, :, 2]
        batch[self.cfg.sim_data] = sim_data
        return batch

    def observational_window(self, batch):
        """
        Mask particles outside the RA/Dec observational window for each stream.
        Sets batch['attention_mask']: (batch_size, 1, n_particles) bool.
        """
        sim_data = batch[self.cfg.sim_data]  # (batch_size, n_particles, d)
        j = batch["j"]                       # (batch_size, 1)
        j_flat = j[:, 0].astype(int)        # (batch_size,)

        ra_min  = self.ra_min_lookup[j_flat][:, None]   # (batch_size, 1)
        ra_max  = self.ra_max_lookup[j_flat][:, None]
        dec_min = self.dec_min_lookup[j_flat][:, None]
        dec_max = self.dec_max_lookup[j_flat][:, None]

        ra  = sim_data[:, :, 0]  # (batch_size, n_particles)
        dec = sim_data[:, :, 1]

        mask = (
            (ra  >= ra_min)  & (ra  <= ra_max) &
            (dec >= dec_min) & (dec <= dec_max)
        )  # (batch_size, n_particles)

        batch["attention_mask"] = mask[:, None, :]  # (batch_size, 1, n_particles)
        return batch

    def subsampling_to_observed_n_stars(self, batch):
        """
        For each batch entry, subsample the True entries in attention_mask
        down to at most observed_n_stars for that stream.
        """
        attention_mask = batch["attention_mask"]  # (batch_size, 1, n_particles)
        j = batch["j"]                            # (batch_size, 1)
        j_flat = j[:, 0].astype(int)

        mask = attention_mask[:, 0, :].astype(bool).copy()  # (batch_size, n_particles)
        batch_size, n_particles = mask.shape

        max_keep = self.observed_n_stars_lookup[j_flat]  # (batch_size,)

        # Assign uniform random scores to True entries, 2.0 (> 1) to False
        random_scores = np.full((batch_size, n_particles), 2.0)
        random_scores[mask] = self.rng.uniform(size=int(mask.sum()))

        # Rank entries per row (ascending); keep only the first max_keep True ones
        sorted_indices = np.argsort(random_scores, axis=1)
        rank = np.argsort(sorted_indices, axis=1)

        keep = mask & (rank < max_keep[:, None])
        batch["attention_mask"] = keep[:, None, :]
        return batch

    def mask_vlos(self, batch):
        """
        After subsampling, randomly select min_star_with_vlos stars from the
        attention_mask to keep their true v_los (last dimension of sim_data).
        For all other stars, replace the last dim of sim_data with a sample
        from N(vlos_mean, vlos_std) and the last dim of sigma_errors with
        the interpolated v_los sigma at a random magnitude.
        """
        j = batch["j"]
        j_flat = j[:, 0].astype(int)             # (batch_size,)
        attention_mask = batch["attention_mask"]  # (batch_size, 1, n_particles)
        sim_data   = batch[self.cfg.sim_data]     # (batch_size, n_particles, d)
        sigma_errors = batch["sigma_errors"]      # (batch_size, n_particles, d)

        mask = attention_mask[:, 0, :].astype(bool)  # (batch_size, n_particles)
        batch_size, n_particles = mask.shape

        n_keep_vlos = self.min_star_with_vlos[j_flat]  # (batch_size,)

        # ---- Select which stars keep their real v_los ----
        random_scores = np.full((batch_size, n_particles), 2.0)
        random_scores[mask] = self.rng.uniform(size=int(mask.sum()))

        sorted_indices = np.argsort(random_scores, axis=1)
        rank = np.argsort(sorted_indices, axis=1)

        vlos_mask = mask & (rank < n_keep_vlos[:, None])  # (batch_size, n_particles)

        # ---- Replace last dim of sim_data for non-vlos stars ----
        vlos_mean = self.vlos_mean_lookup[j_flat]  # (batch_size,)
        vlos_std  = self.vlos_std_lookup[j_flat]   # (batch_size,)

        noise = self.rng.standard_normal(size=(batch_size, n_particles))
        fake_vlos = vlos_mean[:, None] + vlos_std[:, None] * noise  # (batch_size, n_particles)

        sim_data_last = sim_data[:, :, -1].copy()
        sim_data_last[~vlos_mask] = fake_vlos[~vlos_mask]
        sim_data[:, :, -1] = sim_data_last
        batch[self.cfg.sim_data] = sim_data

        # ---- Replace last dim of sigma_errors for non-vlos stars ----
        mag_min = self.mag_min_lookup[j_flat]  # (batch_size,)
        mag_max = self.mag_max_lookup[j_flat]

        rand_uniform = self.rng.uniform(size=(batch_size, n_particles))
        rand_mags = mag_min[:, None] + rand_uniform * (mag_max[:, None] - mag_min[:, None])

        # Interpolate v_los median sigma at the random magnitudes
        vlos_sigma_interp = np.interp(
            rand_mags, self.error_interp_mag_bins, self.vlos_median_std
        )  # (batch_size, n_particles)

        sigma_last = sigma_errors[:, :, -1].copy()
        sigma_last[~vlos_mask] = vlos_sigma_interp[~vlos_mask]
        sigma_errors[:, :, -1] = sigma_last
        batch["sigma_errors"] = sigma_errors

        # Store mask with singleton dim: (batch_size, 1, n_particles)
        batch["vlos_mask"] = vlos_mask[:, None, :]
        return batch

    def sample_magnitudes(self, batch):
        """
        Sample magnitudes from the per-stream KDE.
        Sets batch['magnitudes']: (batch_size, n_particles).
        """
        batch_size, n_particles, _ = batch[self.cfg.sim_data].shape
        j = batch["j"].reshape(-1).astype(int)  # (batch_size,)

        magnitudes = np.zeros((batch_size, n_particles))

        unique_j, inverse = np.unique(j, return_inverse=True)
        for jj in unique_j:
            idx_in_batch = np.where(j == jj)[0]   # which batch entries have this stream
            count = len(idx_in_batch)

            kde = self.kde_streams[int(jj)]
            mag_min, mag_max = self.magnitude_clipping[int(jj)]

            # scipy gaussian_kde.resample uses its own internal state; we seed it
            # via a fresh integer derived from our Generator so sampling remains
            # reproducible relative to self.rng's state.
            seed_for_kde = int(self.rng.integers(0, 2**31))
            kde.random_state = np.random.RandomState(seed_for_kde)

            sampled = kde.resample(size=count * n_particles)   # (1, count*n_particles)
            sampled = sampled.reshape(count, n_particles)
            sampled = np.clip(sampled, mag_min, mag_max)

            magnitudes[idx_in_batch] = sampled

        batch["magnitudes"] = magnitudes
        return batch

    def sample_obs_error(self, batch):
        """
        Sample observational errors for all dimensions listed in self.error_keys,
        based on interpolated Gaia DR3 sigmas at the sampled magnitudes.
        Sets batch['obs_errors'] and batch['sigma_errors']:
        both (batch_size, n_particles, n_error_keys).
        """
        magnitudes = batch["magnitudes"]  # (batch_size, n_particles)

        # Interpolate sigma for each error key: result shape (n_keys, batch_size, n_particles)
        sigmas = np.stack(
            [self.error_interpolators[k](magnitudes) for k in self.error_keys],
            axis=0,
        )  # (n_keys, batch_size, n_particles)

        noise = self.rng.standard_normal(size=sigmas.shape)  # same shape
        errors = sigmas * noise

        # Transpose to (batch_size, n_particles, n_keys)
        batch["sigma_errors"] = np.transpose(sigmas, (1, 2, 0))
        batch["obs_errors"]   = np.transpose(errors, (1, 2, 0))
        return batch

    def apply_obs_error(self, batch):
        """Add sampled observational errors to the first n_error_keys dims of sim_data."""
        n = len(self.error_keys)
        batch[self.cfg.sim_data][:, :, :n] += batch["obs_errors"]
        return batch

    def concatentate_sigma_error_to_sim_data(self, batch):
        """Concatenate sigma_errors to sim_data along the feature axis."""
        batch[self.cfg.sim_data] = np.concatenate(
            [batch[self.cfg.sim_data], batch["sigma_errors"]], axis=-1
        )
        return batch

    def concatenate_magnitudes_to_sim_data(self, batch):
        """Concatenate magnitudes (as an extra feature) to sim_data."""
        batch[self.cfg.sim_data] = np.concatenate(
            [batch[self.cfg.sim_data], batch["magnitudes"][..., None]], axis=-1
        )
        return batch

    def concatenate_j_to_sim_data(self, batch):
        """Concatenate stream index j (broadcast) to sim_data."""
        j_expanded = np.broadcast_to(
            batch["j"][:, None, :],
            (batch[self.cfg.sim_data].shape[0], batch[self.cfg.sim_data].shape[1], 1),
        )
        batch[self.cfg.sim_data] = np.concatenate(
            [batch[self.cfg.sim_data], j_expanded], axis=-1
        )
        return batch

    def concatenate_vlos_mask_to_sim_data(self, batch):
        """
        Concatenate vlos_mask to sim_data.
        vlos_mask shape: (batch_size, 1, n_particles) -> (batch_size, n_particles, 1)
        """
        vlos_mask = batch["vlos_mask"].transpose(0, 2, 1).astype(
            batch[self.cfg.sim_data].dtype
        )
        batch[self.cfg.sim_data] = np.concatenate(
            [batch[self.cfg.sim_data], vlos_mask], axis=-1
        )
        return batch

    def flip_dirz(self, batch):
        """
        Ensure the z-component of the direction vector is positive by flipping
        all three components when dirz < 0.
        """
        dirz = batch["dirz_Triaxial_rotated_halo"]
        mask = dirz < 0
        batch["dirz_Triaxial_rotated_halo"] = np.where(mask, -dirz, dirz)
        batch["dirx_Triaxial_rotated_halo"] = np.where(
            mask, -batch["dirx_Triaxial_rotated_halo"], batch["dirx_Triaxial_rotated_halo"]
        )
        batch["diry_Triaxial_rotated_halo"] = np.where(
            mask, -batch["diry_Triaxial_rotated_halo"], batch["diry_Triaxial_rotated_halo"]
        )
        return batch

    def override_vlos_error_with_real(self, batch):
        """
        Override the last dimension of sigma_errors (v_los error) with the
        real observed vlos_error from the dataset, where vlos_mask == 1.
        Where vlos_mask == 0 (no real v_los), keep the sampled sigma_error.

        Expects:
          batch["sigma_errors"]: (batch_size, n_particles, n_error_keys)
          batch["vlos_error"]:   (batch_size, n_particles, 1)
          batch["vlos_mask"]:    (batch_size, 1, n_particles)  — 1 where real v_los exists
        """
        sigma_errors = batch["sigma_errors"]                    # (batch_size, n_particles, d)
        vlos_error   = batch["vlos_error"][:, :, 0]            # (batch_size, n_particles)
        # vlos_mask stored as (batch_size, 1, n_particles) -> transpose
        vlos_mask    = batch["vlos_mask"][:, 0, :].astype(bool)  # (batch_size, n_particles)

        sigma_vlos = sigma_errors[:, :, -1].copy()
        sigma_vlos[vlos_mask] = vlos_error[vlos_mask]
        sigma_errors[:, :, -1] = sigma_vlos
        batch["sigma_errors"] = sigma_errors
        return batch