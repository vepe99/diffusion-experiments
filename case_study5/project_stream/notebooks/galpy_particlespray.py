"""
Stellar stream simulation pipeline – Palau & Miralda-Escudé (2023)
===================================================================
arXiv: 2212.03587  |  MNRAS (2023)

Milky Way potential (3 components):
  1. Thin  disc  – DoubleExponentialDiskPotential          (Eq. 1)
  2. Thick disc  – DoubleExponentialDiskPotential          (Eq. 1)
  3. Bulge       – PowerSphericalPotentialwCutoff           (Eq. 3, McMillan 2017 fixed params, Table 2)
  4. Dark halo   – TwoPowerTriaxialPotential (axisymmetric) (Eq. 6)

Best-fit values are from the "all three streams" model (Section 5.3 / Table F1).

Galpy unit conventions (the important ones)
-------------------------------------------
  ro   [kpc]   – Galactocentric radius of the Sun (= 8.178 kpc here)
  vo   [km/s]  – Circular velocity at ro (= 230.67 km/s here, = Θ₀)
  Time unit   = ro / vo ≈ 34.7 Myr  → pass t_integ in natural units (see below)
  Mass unit   = vo² ro / G  ≈ conv.mass_in_msol(vo, ro) × M_sun

Required packages
-----------------
  galpy >= 1.8
  astropy
  scipy  (only for _halo_rho0_from_M200 auto-calibration)
  numpy
"""

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.integrate import quad

from galpy.potential import (
    DoubleExponentialDiskPotential,
    PowerSphericalPotentialwCutoff,
    TwoPowerTriaxialPotential,
)
from galpy.orbit import Orbit
from galpy.df import streamspraydf
import galpy.util.conversion as conv
import matplotlib.pyplot as plt
import galpy.df

# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT PARAMETERS  –  Palau & Miralda-Escudé 2023, all-streams best fit
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_PARAMS = {

    # ── Reference frame ────────────────────────────────────────────────────────
    # These define galpy's natural unit system.  Must be consistent everywhere.
    "ro": 8.178 * u.kpc,           # R_sun   – Gravity Collaboration 2019 (Table 1)
    "vo": 230.67 * u.km / u.s,     # Θ₀ LSR  – Section 5.3 best fit
    "zo": 0.025 * u.kpc,           # z_sun   – Jurić et al. 2008 (Table 2)
    # Solar peculiar velocity (U toward GC, V along rotation, W toward NGP)
    # Schönrich et al. 2010 – Table 1
    "solarmotion": [-11.1, 12.24, 7.25] * u.km / u.s,

    # ── Thin disc  (double exponential, Eq. 1) ────────────────────────────────
    # M_thin from Section 5.3: M_d^n = (6.07 ± 0.39) × 10^10 M_sun
    "M_thin": 6.07e10 * u.Msun,
    "sigma_thin": 1.25 * 1e9 * u.Msun / u.kpc**2,   # Alternative value
    "h_thin": 2.6 * u.kpc,         # radial scale length  – Table 1 prior centroid
    "z_thin": 0.3 * u.kpc,         # vertical scale height

    # ── Thick disc  (double exponential, Eq. 1) ───────────────────────────────
    # Derived: M_bar=8.01e10, M_bulge=8.9e9, M_thin=6.07e10 → M_thick ≈ 1.05e10
    "sigma_thick": 3.77 * 1e8 * u.Msun / u.kpc**2,   # Alternative value
    "M_thick": 1.05e10 * u.Msun,
    "h_thick": 2.0 * u.kpc,
    "z_thick": 0.9 * u.kpc,

    # ── Bulge  (Eq. 3 / Table 2, all fixed except normalisation) ─────────────
    # ρ_b(s) = ρ₀_b × (1 + s/h_b)^{–α_b} × exp(–s²/a₁b²),  q_b=0.5 (oblate)
    # The (1 + s/h_b)^{–α_b} core is negligible for s >> h_b = 75 pc, so
    # PowerSphericalPotentialwCutoff (ρ ∝ r^{–α_b} exp(–r²/a₁b²)) is an
    # excellent approximation at the stream radii (> 10 kpc).
    # The oblate flattening q_b = 0.5 is also negligible beyond a few kpc.
    "M_bulge": 8.9e9 * u.Msun,     # Table 1 prior  (8.9 ± 0.89) × 10⁹ M_sun
    "rho_bulge": 5.0E10 * u.Msun / u.kpc**3,    # Alternative value
    "alpha_bulge": 1.8,             # power-law slope  – Table 2
    "rc_bulge": 2.1 * u.kpc,       # Gaussian truncation radius a₁b – Table 2

    # ── Dark halo  (axisymmetric two-power-law, Eq. 6) ────────────────────────
    # ρ_h(s) = ρ_h0 × (s/a₁)^{–α} × (1 + s/a₁)^{α–β}
    # with  s² = R² + z²/q_h²   (iso-density ellipsoids, Eq. 4)
    # Best-fit (all streams, Section 5.3 / Table F1):
    "rho_h0": 1.84*1e7 * u.Msun / u.kpc**3,   # None → auto-computed to match M_h200 below
    "M_h200": 1.08e12 * u.Msun,    # M_h^200 = (1.08 ± 0.22) × 10¹² M_sun
    "r200": 215.3 * u.kpc,         # r₂₀₀ (Callingham et al. 2019, Eq. 10)
    "alpha_h": 0.06,                # inner slope (≈ flat core, Table F1)
    "beta_h": 3.3,                  # outer slope
    "a1_h": 17.0 * u.kpc,          # scale radius  (17⁺¹⁰₋₃ kpc)
    "q_h": 1.06,                    # z-axis ratio: >1 = prolate (Section 5.3)
    #   NOTE on axis-ratio convention:
    #   Paper Eq. 4:  s² = R² + z²/q_h²
    #   Galpy m:      m  = √(R² + (z/c)²)  →  c = q_h
    #   q_h = 1.06 > 1 → iso-density ellipsoid elongated along z (prolate halo).

    # ── Progenitor  (must be provided by the user) ────────────────────────────
    "prog_mass": None,              # e.g.  5e4 * u.Msun   (sets tidal radius)
    "prog_plummer_a": None,         # Plummer scale radius, e.g.  5.0 * u.pc
    "prog_ic": None,                # astropy SkyCoord with full 6-D kinematics

    # ── Simulation control ────────────────────────────────────────────────────
    "t_integ": 3.0 * u.Gyr,        # integration time (backward in time)
    "n_particles": 1000,            # number of spray particles to sample
    "n_times": 3000,                # number of timesteps for orbit integration
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# def _disk_rho0(M_disc: u.Quantity, h_r: u.Quantity, z_h: u.Quantity) -> u.Quantity:
def _disk_rho0(params, disc_type) -> u.Quantity:
    """
    Central volume density of a double-exponential disc.

        ρ(R, z) = ρ₀ × exp(–R/h_R) × exp(–|z|/h_z)
        Σ₀      = M / (2π h_R²)          (surface density at R=0)
        ρ₀      = Σ₀ / (2 h_z)

    Parameters
    ----------
    M_disc : astropy Quantity [mass]
    h_r    : astropy Quantity [length]  – radial scale length
    z_h    : astropy Quantity [length]  – vertical scale height

    Returns
    -------
    rho0 : astropy Quantity in M_sun / kpc³
    """
    h_r = params[f"h_{disc_type}"]
    z_h = params[f"z_{disc_type}"]
    Sigma0 = params[f"M_{disc_type}"] / (2.0 * np.pi * h_r**2)
    # Sigma0 = params[f"sigma_{disc_type}"]
    rho0 = Sigma0 / (2.0 * z_h)
    return rho0.to(u.Msun / u.kpc**3)


def _halo_rho0_from_M200(
    M200: u.Quantity,
    r200: u.Quantity,
    alpha: float,
    beta: float,
    a1: u.Quantity,
    q_h: float,
) -> u.Quantity:
    """
    Numerically find ρ_h0 such that M_halo(<r₂₀₀) = M200.

    Following Eq. 8 of the paper, mass is integrated over ellipsoidal shells:

        M(<r₂₀₀) = 4π q_h ∫₀^{r₂₀₀} s² ρ_h(s) ds

    where ρ_h(s) = ρ_h0 × (s/a₁)^{–α} × (1 + s/a₁)^{α–β}.

    Parameters
    ----------
    M200, r200 : astropy Quantities [mass, length]
    alpha, beta, q_h : float  – halo profile parameters
    a1 : astropy Quantity [length]

    Returns
    -------
    rho_h0 : astropy Quantity in M_sun / kpc³
    """
    a1_kpc = a1.to(u.kpc).value
    r200_kpc = r200.to(u.kpc).value
    M200_msun = M200.to(u.Msun).value

    # Small lower limit to avoid numerical issues when alpha > 0
    s_min = 1e-4 * a1_kpc

    def integrand(s):
        return 4.0 * np.pi * q_h * s**2 * (s / a1_kpc)**(-alpha) * (1.0 + s / a1_kpc)**(alpha - beta)

    norm, _ = quad(integrand, s_min, r200_kpc, limit=200)
    rho_h0_val = M200_msun / norm
    return rho_h0_val * u.Msun / u.kpc**3


# ══════════════════════════════════════════════════════════════════════════════
# POTENTIAL BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_potential(params: dict) -> tuple:
    """
    Build the Palau & Miralda-Escudé (2023) MW potential.

    The returned list is ready to pass directly to galpy's ``Orbit.integrate``
    and ``streamspraydf``.

    Parameters
    ----------
    params : dict
        Parameter dictionary (keys from DEFAULT_PARAMS; all values must carry
        astropy units where applicable).

    Returns
    -------
    pot : list of galpy Potential objects
        [thin_disc, thick_disc, bulge, halo]
    ro  : astropy Quantity [kpc]
    vo  : astropy Quantity [km/s]
    """
    ro = params["ro"]
    vo = params["vo"]

    # ── 1.  Thin disc ─────────────────────────────────────────────────────────
    rho0_thin = _disk_rho0(params, "thin")
    thin_disc = DoubleExponentialDiskPotential(
        amp=rho0_thin,              # M_sun / kpc³  →  galpy central density
        # amp = params["M_thin"],         # M_sun           →  galpy normalises the profile to match this total mass
        hr=params["h_thin"],
        hz=params["z_thin"],
        ro=ro, vo=vo,
    )

    # ── 2.  Thick disc ────────────────────────────────────────────────────────
    rho0_thick = _disk_rho0(params, "thick")
    thick_disc = DoubleExponentialDiskPotential(
        amp=rho0_thick,
        # amp = params["M_thick"],
        hr=params["h_thick"],
        hz=params["z_thick"],
        ro=ro, vo=vo,
    )

    # ── 3.  Bulge  (spherical approximation) ──────────────────────────────────
    #   ρ ∝ r^{–α_b} × exp(–r²/rc²)
    #   amp = M_bulge → galpy normalises the total enclosed mass accordingly.
    #   The paper's oblate shape (q_b = 0.5) and the (1 + r/h_b)^{–α} core
    #   (h_b = 75 pc) are negligible for stream orbits beyond ~10 kpc.
    bulge = PowerSphericalPotentialwCutoff(
        amp=params["rho_bulge"],   # M_sun / kpc³  (alternative to M_bulge)
        # amp = params["M_bulge"],     # M_sun          → galpy normalises the profile to match this total mass
        alpha=params["alpha_bulge"],
        rc=params["rc_bulge"],
        ro=ro, vo=vo,
    )

    # ── 4.  Dark halo (axisymmetric two-power-law) ────────────────────────────
    #   ρ_h(s) = ρ_h0 × (s/a₁)^{–α} × (1 + s/a₁)^{α–β}
    #   s = √(R² + z²/q_h²)  →  galpy TwoPowerTriaxialPotential with c = q_h
    #
    #   Amplitude convention:
    #     galpy density: ρ = amp × m^{–α} × (1 + m)^{α–β},  m = s/a₁
    #     Paper density: ρ = ρ_h0 × (s/a₁)^{–α} × (1 + s/a₁)^{α–β}
    #     → amp = ρ_h0  (when amp is passed in density units M_sun/kpc³)
    rho_h0 = params.get("rho_h0")
    if rho_h0 is None:
        print(
            "[build_potential] rho_h0 not provided – auto-calibrating to "
            f"M_h200 = {params['M_h200']:.3e} within r200 = {params['r200']:.1f} …"
        )
        rho_h0 = _halo_rho0_from_M200(
            params["M_h200"],
            params["r200"],
            params["alpha_h"],
            params["beta_h"],
            params["a1_h"],
            params["q_h"],
        )
        print(f"    → ρ_h0 = {rho_h0:.4e}")

    halo = TwoPowerTriaxialPotential(
        # amp=params['rho_h0']*(4*np.pi*params["a1_h"]**3), # M_sun / kpc³  (= ρ_h0 in paper notation)
        amp = params['rho_h0'] * u.kpc**3,  # M_sun  (galpy normalises the profile to match this mass at large radii)
        a=params["a1_h"],           # scale radius  [kpc]
        alpha=params["alpha_h"],    # inner slope
        beta=params["beta_h"],      # outer slope
        b=1.0,                      # axisymmetric: y = x (no φ-flattening)
        c=params["q_h"],            # z-axis ratio  (c = q_h,  see NOTE above)
        ro=ro, vo=vo,
    )

    pot = [thin_disc, thick_disc, bulge, halo]
    return pot, ro, vo


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(params: dict) -> dict:
    """
    End-to-end stellar stream particle-spray simulation.

    Steps
    -----
    1. Merge user params with DEFAULT_PARAMS.
    2. Build the Palau & Miralda-Escudé (2023) MW potential.
    3. Integrate the progenitor orbit **backward** in time.
    4. Run ``streamspraydf`` (Küpper+2012 particle-spray with self-gravity
       encoded in the Jacobi radius via ``prog_mass``).
    5. Convert sampled particles to observable sky coordinates.

    Parameters
    ----------
    params : dict
        Overrides for any key in DEFAULT_PARAMS.  The three keys below are
        **required** (no default):

        prog_mass      : astropy Quantity [mass]    – progenitor mass today
                         e.g.  5e4 * u.Msun
                         Sets the tidal radius → controls stream width.

        prog_plummer_a : astropy Quantity [length]  – Plummer scale radius
                         e.g.  5.0 * u.pc
                         Sets internal velocity dispersion of ejected stars.

        prog_ic        : astropy SkyCoord with full 6-D kinematics
                         (ra, dec, distance, pm_ra_cosdec, pm_dec,
                          radial_velocity)

    Returns
    -------
    result : dict with keys
        "potential"     – list of galpy Potential objects
        "prog_orbit"    – integrated progenitor Orbit
        "stream_df"     – streamspraydf instance (re-sample with .sample())
        "stream_ra"     – RA of stream stars           [deg]
        "stream_dec"    – Dec of stream stars          [deg]
        "stream_dist"   – heliocentric distance        [kpc]
        "stream_pmra"   – µ_α* proper motion           [mas/yr]
        "stream_pmdec"  – µ_δ  proper motion           [mas/yr]
        "stream_vlos"   – line-of-sight velocity       [km/s]
        "ro", "vo"      – galpy reference units used
        "rho_h0"        – halo scale density that was used  [M_sun/kpc³]
        "t_unit_Gyr"    – time unit (1 natural unit in Gyr)

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> from stream_pipeline import run_simulation, DEFAULT_PARAMS
    >>>
    >>> sc = SkyCoord(
    ...     ra=154.4 * u.deg, dec=-46.4 * u.deg, distance=4.9 * u.kpc,
    ...     pm_ra_cosdec=8.32 * u.mas/u.yr, pm_dec=-1.99 * u.mas/u.yr,
    ...     radial_velocity=-494.3 * u.km/u.s,   # NGC 3201 kinematics
    ...     frame="icrs",
    ... )
    >>> result = run_simulation({
    ...     "prog_mass"     : 6.47e4 * u.Msun,   # Table 2
    ...     "prog_plummer_a": 4.9 * u.pc,         # Table 2
    ...     "prog_ic"       : sc,
    ...     "t_integ"       : 1.5 * u.Gyr,
    ...     "n_particles"   : 2000,
    ... })
    """

    # ── 1.  Merge params ──────────────────────────────────────────────────────
    p = {**DEFAULT_PARAMS, **params}

    for key in ("prog_mass", "prog_plummer_a", "prog_ic"):
        if p[key] is None:
            raise ValueError(
                f"Required parameter '{key}' is not set in params.\n"
                "See run_simulation docstring for examples."
            )

    ro = p["ro"]
    vo = p["vo"]
    ro_kpc = ro.to(u.kpc).value
    vo_kms = vo.to(u.km / u.s).value

    # ── 2.  Build potential ───────────────────────────────────────────────────
    pot, ro, vo = build_potential(p)
    pot = galpy.potential.mwpotentials.Cautun20
    ro = pot._ro
    vo = pot._vo
    ro_kpc = ro * u.kpc
    vo_kms = vo * u.km / u.s

    # ── 3.  Time array  (IMPORTANT: negative = integrate backward) ────────────
    #
    #   Time unit: t₀ = ro / vo ≈ 34.7 Myr  (conv.time_in_Gyr gives Gyr/unit)
    #
    #   t_integ = 3 Gyr → t_back = –3.0 / 0.0347 ≈ –86.5 in natural units
    #
    t_unit_Gyr = conv.time_in_Gyr(vo, ro)   # e.g. ~0.0347 Gyr per unit
    t_back_nat = -(p["t_integ"].to(u.Gyr).value) / t_unit_Gyr   # negative!
    print(f"Integrating backward for {p['t_integ']} → t_back_nat = {t_back_nat:.1f} in natural units")
    ts = np.linspace(0.0, t_back_nat, int(p["n_times"]))

    # ── 4.  Progenitor orbit from SkyCoord ────────────────────────────────────
    prog_orbit = Orbit(
        p["prog_ic"],
        ro=ro,
        vo=vo,
        zo=p["zo"],
        solarmotion=p["solarmotion"],
    )
    # prog_orbit.integrate(ts, pot)

    # ── 5.  streamspraydf  (Küpper+ 2012 particle spray) ─────────────────────
    #
    #   prog_mass       → Jacobi/tidal radius → ejection speed from L1/L2
    #   prog_plummer_a  → internal velocity dispersion of the cluster
    #   tdisrupt        → time over which stripping occurred  (|t_back_nat|)
    #
    progpot = galpy.potential.PlummerPotential(6.47*10**4.0 * u.Msun, 4.9 * u.pc, ro=ro, vo=vo)  # Progenitor's self-potential (for Jacobi radius)
    spdf = galpy.df.chen24spraydf(
        progenitor_mass=6.47*10**4.0 * u.Msun,
        progenitor=prog_orbit,
        pot=pot,
        tail="both",
        progpot=progpot,
        tdisrupt=np.abs(t_back_nat),    # positive natural time units
        ro=ro,
        vo=vo,
    )

    # ── 6.  Sample stream particles ───────────────────────────────────────────
    #
    #   RvR shape: (6, N) – [R, vR, vT, z, vz, phi] in natural units
    #   dt       : stripping time of each particle (natural units, negative)
    #
    RvR, dt = spdf.sample(n=int(p["n_particles"]), returndt=True, integrate=True)

    # ── 7.  Convert to sky coordinates ───────────────────────────────────────
    #
    #   Pass RvR.T  (N × 6)  to Orbit;  galpy interprets it as natural units.
    #
    # stream_orbits = Orbit(
    #     RvR.Tp,
    #     ro=ro, vo=vo,
    #     zo=p["zo"],
    #     solarmotion=p["solarmotion"],
    # )
    stream_orbits = RvR

    result = {
        "potential"    : pot,
        "prog_orbit"   : prog_orbit,
        "stream_df"    : spdf,
        # ── Observable phase-space ──────────────────────────────────────────
        "stream_ra"    : stream_orbits.ra()    * u.deg,
        "stream_dec"   : stream_orbits.dec()   * u.deg,
        "stream_dist"  : stream_orbits.dist()  * u.kpc,
        "stream_pmra"  : stream_orbits.pmra()  * u.mas / u.yr,
        "stream_pmdec" : stream_orbits.pmdec() * u.mas / u.yr,
        "stream_vlos"  : stream_orbits.vlos()  * u.km / u.s,

        "stream_orbits" : stream_orbits,  # galpy Orbit instance with full phase-space info
        # ── Metadata ────────────────────────────────────────────────────────
        "ro"           : ro,
        "vo"           : vo,
        "t_unit_Gyr"   : t_unit_Gyr * u.Gyr,
        "rho_h0"       : p.get("rho_h0"),       # None if auto-computed
    }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK  (run as script: python stream_pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from galpy.potential import evaluatePotentials, evaluateDensities
    import astropy.constants as const

    print("=" * 60)
    print("Building Palau & Miralda-Escudé 2023 potential …")
    pot, ro, vo = build_potential(DEFAULT_PARAMS)
    pot = galpy.potential.mwpotentials.Cautun20
    ro = pot._ro
    vo = pot._vo

    # ro_kpc = ro.to(u.kpc).value
    # vo_kms = vo.to(u.km / u.s).value
    ro_kpc = ro * u.kpc
    vo_kms = vo * u.km / u.s

    # ── Circular velocity at R_sun should recover vo ──────────────────────────
    from galpy.potential import vcirc
    # vc_sun = vcirc(pot, 1.0) * vo_kms   # R=1 in natural units = ro
    vc_sun = vcirc(pot, 1.0) 
    print(f"  v_circ(R_sun) = {vc_sun:.2f} km/s  (should be ≈ {vo_kms:.2f} km/s)")

    # ── Time unit ─────────────────────────────────────────────────────────────
    t_unit = conv.time_in_Gyr(vo_kms, ro_kpc)
    print(f"  1 natural time unit = {t_unit*1e3:.2f} Myr  (should be ≈ 34.7 Myr)")

    # ── NGC 3201 mini-simulation ──────────────────────────────────────────────
    print("\nRunning NGC 3201 stream simulation (500 particles, 1.5 Gyr) …")
    sc_ngc3201 = SkyCoord(
        ra=154.403 * u.deg,
        dec=-46.412 * u.deg,
        distance=4.9 * u.kpc,
        pm_ra_cosdec=8.324 * u.mas / u.yr,
        pm_dec=-1.991 * u.mas / u.yr,
        radial_velocity=-494.34 * u.km / u.s,
        frame="icrs",
    )

    result = run_simulation({
        "prog_mass"     : 6.47e4 * u.Msun,   # Table 2 – Sollima & Baumgardt 2017
        "prog_plummer_a": 4.9    * u.pc,      # Table 2 – a_gc
        "prog_ic"       : sc_ngc3201,
        "t_integ"       : 1.5 * u.Gyr,
        "n_particles"   : 500,
        "n_times"       : 1000,
    })

    ra  = result["stream_ra"]
    dec = result["stream_dec"]
    d   = result["stream_dist"]
    print(f"  Stream RA  range: {ra.min():.1f} – {ra.max():.1f}")
    print(f"  Stream Dec range: {dec.min():.1f} – {dec.max():.1f}")
    print(f"  Stream dist range: {d.min():.2f} – {d.max():.2f}")
    print("t_unit_Gyr =", result["t_unit_Gyr"])
    print("Done ✓")
    plt.plot(ra, dec, "o", markersize=2, alpha=0.5)
    plt.xlabel("RA [deg]")
    plt.ylabel("Dec [deg]")
    plt.savefig("ngc3201_stream.png", dpi=150)