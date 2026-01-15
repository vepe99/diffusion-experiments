import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic

_ = coord.galactocentric_frame_defaults.set('v4.0')

c = coord.ICRS(ra=229 * u.deg, dec=-0.124 * u.deg,
               distance=22.9 * u.kpc,
               pm_ra_cosdec=-2.296 * u.mas/u.yr,
               pm_dec=-2.257 * u.mas/u.yr,
               radial_velocity=-58.7 * u.km/u.s)

c_gc = c.transform_to(coord.Galactocentric()).cartesian
pal5_w0 = gd.PhaseSpacePosition(c_gc)

print(pal5_w0)