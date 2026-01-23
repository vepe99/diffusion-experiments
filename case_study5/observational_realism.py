from autocvd import autocvd 
autocvd(num_gpus = 1)

from jax import jit, vmap
import jax.numpy as jnp

import coordinax as cx
import coordinax.vecs as cxv
import unxt as u


# Create the transform operator once (outside JIT for efficiency)
_galc_to_icrs_op = cx.frames.frame_transform_op(cx.frames.Galactocentric(), cx.frames.ICRS())


@jit
def galc_to_icrs(pos_vel_gal):
    """
    Transform Galactocentric coordinates to ICRS.
    
    Args:
        pos_vel_gal: Array of shape (N, 6) with columns [x, y, z, vx, vy, vz]
                     positions in kpc, velocities in km/s
    
    Returns:
        ICRS coordinate object
    """
    pos = cxv.CartesianPos3D(
        x=u.Quantity(pos_vel_gal[:, 0], "kpc"),
        y=u.Quantity(pos_vel_gal[:, 1], "kpc"),
        z=u.Quantity(pos_vel_gal[:, 2], "kpc")
    )
    vel = cxv.CartesianVel3D(
        x=u.Quantity(pos_vel_gal[:, 3], "km/s"),
        y=u.Quantity(pos_vel_gal[:, 4], "km/s"),
        z=u.Quantity(pos_vel_gal[:, 5], "km/s")
    )
    gal_coord = cx.Coordinate(
        {"length": pos, "speed": vel}, 
        frame=cx.frames.Galactocentric()
    )
    
    # Apply the pre-compiled transform operator
    icrs_coord = _galc_to_icrs_op(gal_coord)
    return icrs_coord


@jit
def galc_to_icrs_spherical(pos_vel_gal):
    """
    Transform Galactocentric to ICRS and return spherical coordinates.
    
    Args:
        pos_vel_gal: Array of shape (N, 6) with columns [x, y, z, vx, vy, vz]
                     positions in kpc, velocities in km/s
    
    Returns:
        Array of shape (N, 6) with columns [ra, dec, distance, pm_ra_cosdec, pm_dec, radial_velocity]
        - ra in deg, dec in deg, distance in kpc
        - pm_ra_cosdec in mas/yr, pm_dec in mas/yr, radial_velocity in km/s
    """
    icrs_coord = galc_to_icrs(pos_vel_gal)
    
    # Convert to spherical representation for observational quantities
    pos_sph = icrs_coord.data["length"].vconvert(cxv.LonLatSphericalPos)
    vel_sph = icrs_coord.data["speed"].vconvert(cxv.LonLatSphericalVel, pos_sph)
    
    # Extract values with proper unit conversions
    ra = pos_sph.lon.to("deg").value       # RA in degrees
    dec = pos_sph.lat.to("deg").value      # Dec in degrees
    dist = pos_sph.distance.to("kpc").value # Distance in kpc
    
    pm_lon = vel_sph.lon.to("mas/yr").value    # PM in lon direction (mas/yr)
    pm_lat = vel_sph.lat.to("mas/yr").value    # PM in lat direction (mas/yr)
    rv = vel_sph.distance.to("km/s").value     # Radial velocity (km/s)
    
    # Apply cos(dec) correction to match Astropy's pm_ra_cosdec convention
    dec_rad = pos_sph.lat.to("rad").value
    pm_lon_cosdec = pm_lon * jnp.cos(dec_rad)
    
    # Stack into (N, 6) array
    return jnp.stack([ra, dec, dist, pm_lon_cosdec, pm_lat, rv], axis=-1)


if __name__ == "__main__":
    # Example usage
    pos_vel = jnp.array([[8.0, 0.0, 0.0, 0.0, 220.0, 0.0],
                         [10.0, 5.0, 2.0, 50.0, 180.0, 30.0]])  # Test with 2 particles

    # Compare with Astropy
    import numpy as np
    from astropy import units as au
    from astropy.coordinates import Galactocentric, ICRS, SkyCoord
    
    pos_vel_np = np.array(pos_vel)
    
    galcen_astropy = SkyCoord(
        x=pos_vel_np[:, 0] * au.kpc,
        y=pos_vel_np[:, 1] * au.kpc,
        z=pos_vel_np[:, 2] * au.kpc,
        v_x=pos_vel_np[:, 3] * au.km / au.s,
        v_y=pos_vel_np[:, 4] * au.km / au.s,
        v_z=pos_vel_np[:, 5] * au.km / au.s,
        frame=Galactocentric()
    )
    
    icrs_astropy = galcen_astropy.transform_to(ICRS())
    
    print("Astropy ICRS Coordinate:")
    print(f"  RA: {icrs_astropy.ra}")
    print(f"  Dec: {icrs_astropy.dec}")
    print(f"  Distance: {icrs_astropy.distance}")
    print(f"  PM RA: {icrs_astropy.pm_ra_cosdec}")
    print(f"  PM Dec: {icrs_astropy.pm_dec}")
    print(f"  Radial velocity: {icrs_astropy.radial_velocity}")
    
    # Get coordinax spherical output
    pos_vel_sph = galc_to_icrs_spherical(pos_vel)
    print("\n--- Coordinax ICRS Spherical (N, 6) ---")
    print(f"Shape: {pos_vel_sph.shape}")
    print(f"Columns: [RA(deg), Dec(deg), Dist(kpc), PM_RA(mas/yr), PM_Dec(mas/yr), RV(km/s)]")
    print(f"Output:\n{pos_vel_sph}")
    
    # Numerical comparison
    print("\n--- Numerical Comparison ---")
    for i in range(pos_vel.shape[0]):
        print(f"\nParticle {i}:")
        print(f"  RA:   coordinax={pos_vel_sph[i, 0]:.6f} deg, Astropy={icrs_astropy.ra.deg[i]:.6f} deg, diff={pos_vel_sph[i, 0] - icrs_astropy.ra.deg[i]:.6f} deg")
        print(f"  Dec:  coordinax={pos_vel_sph[i, 1]:.6f} deg, Astropy={icrs_astropy.dec.deg[i]:.6f} deg, diff={pos_vel_sph[i, 1] - icrs_astropy.dec.deg[i]:.6f} deg")
        print(f"  Dist: coordinax={pos_vel_sph[i, 2]:.6f} kpc, Astropy={icrs_astropy.distance.kpc[i]:.6f} kpc, diff={pos_vel_sph[i, 2] - icrs_astropy.distance.kpc[i]:.6f} kpc")
        print(f"  PM_RA:  coordinax={pos_vel_sph[i, 3]:.6f} mas/yr, Astropy={icrs_astropy.pm_ra_cosdec.value[i]:.6f} mas/yr, diff={pos_vel_sph[i, 3] - icrs_astropy.pm_ra_cosdec.value[i]:.6f}")
        print(f"  PM_Dec: coordinax={pos_vel_sph[i, 4]:.6f} mas/yr, Astropy={icrs_astropy.pm_dec.value[i]:.6f} mas/yr, diff={pos_vel_sph[i, 4] - icrs_astropy.pm_dec.value[i]:.6f}")
        print(f"  RV:   coordinax={pos_vel_sph[i, 5]:.6f} km/s, Astropy={icrs_astropy.radial_velocity.value[i]:.6f} km/s, diff={pos_vel_sph[i, 5] - icrs_astropy.radial_velocity.value[i]:.6f}")