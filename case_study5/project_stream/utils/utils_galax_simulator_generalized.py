from functools import partial
import astropy.units as u
import astropy.coordinates as coord
import numpy as np
from jax import jit
import jax.random as jr
import jax.numpy as jnp
from jax.scipy import special


import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
from unxt import Quantity
import coordinax as cx
EPSILON       = 1e-12  # Small constant to avoid division by zero


@jit
def get_mat(x, y, z):
    v1 = jnp.array([0.0, 0.0, 1.0])
    I3 = jnp.eye(3)

    # Create a fixed-shape vector from inputs
    v2 = jnp.array([x, y, z])
    # Normalize v2 in one step
    v2 = v2 / (jnp.linalg.norm(v2) + EPSILON)

    # Compute the angle using a fused dot and clip operation
    angle = jnp.arccos(jnp.clip(jnp.dot(v1, v2), -1.0, 1.0))

    # Compute normalized rotation axis
    v3 = jnp.cross(v1, v2)
    v3 = v3 / (jnp.linalg.norm(v3) + EPSILON)

    # Build the skew-symmetric matrix K for Rodrigues' formula
    K = jnp.array([
        [0, -v3[2], v3[1]],
        [v3[2], 0, -v3[0]],
        [-v3[1], v3[0], 0]
    ])

    sin_angle = jnp.sin(angle)
    cos_angle = jnp.cos(angle)

    # Compute rotation matrix using Rodrigues' formula
    rot_mat = I3 + sin_angle * K + (1 - cos_angle) * jnp.dot(K, K)
    return rot_mat

@partial(jit, static_argnames=('config', 'code_units'))
def simulate_stream_galax(parameters_dict, config, code_units, random_seed:int):
    '''
    Docstring for simulate_stream_galax
    code_units it is not used 
    
    :param parameters_dict: Description
    :param config: Description
    :type config: SimulationConfig
    :param code_units: Description
    :type code_units: CodeUnits
    :param random_seed: Description
    :type random_seed: int
    '''
    q_min = 0.5
    q_max = 1.5
    flip = jnp.where(parameters_dict['dirz_Triaxial_rotated_halo'][0] < 0, -1.0, 1.0)
    dirx = parameters_dict['dirx_Triaxial_rotated_halo'][0] * flip
    diry = parameters_dict['diry_Triaxial_rotated_halo'][0] * flip
    dirz = parameters_dict['dirz_Triaxial_rotated_halo'][0] * flip
    r = jnp.sqrt(dirx**2 + diry**2 + dirz**2)
    u_from_r = special.erf(r/jnp.sqrt(2)) - jnp.sqrt(2/jnp.pi)*r*jnp.exp(-(r**2)/2)
    q = q_min + (q_max - q_min) * u_from_r
    R = get_mat(dirx, diry, dirz)

    base_potential = gp.TriaxialNFWPotential( m     = parameters_dict['m_Triaxial_halo'][0],
                                    r_s   = parameters_dict['r_Triaxial_halo'][0],
                                    q1     = parameters_dict['q1_Triaxial_halo'][0],
                                    q2     = q,
                                    units ="galactic")
    op = cx.ops.GalileanRotation(R)

    pot = gp.CompositePotential(

    halo = gp.TransformedPotential(base_potential=base_potential, xop=op),

    thin_disk = gp.MN3ExponentialPotential(m_tot = 4 * np.pi * parameters_dict['rho_thin_disk'][0]*parameters_dict['hr_thin_disk'][0]**2 * parameters_dict['hz_thin_disk'][0],
                                                    h_R=parameters_dict['hr_thin_disk'][0],
                                                    h_z=parameters_dict['hz_thin_disk'][0],
                                                    units="galactic",
                                                    positive_density=True),
    thick_disk = gp.MN3ExponentialPotential(m_tot = 4 * np.pi * parameters_dict['rho_thick_disk'][0] *parameters_dict['hr_thick_disk'][0]**2 * parameters_dict['hz_thick_disk'][0],
                                                    h_R=parameters_dict['hr_thick_disk'][0],
                                                    h_z=parameters_dict['hz_thick_disk'][0],
                                                    units="galactic",
                                                    positive_density=True),
    bulge = gp.PowerLawCutoffPotential(m_tot=parameters_dict['m_bulge'][0],
                                        r_c=parameters_dict['r_bulge'][0],
                                        alpha=parameters_dict['alpha_bulge'][0],
                                        units="galactic")
    )

    # w0 = coord.Galactocentric(x=parameters_dict['x'][0], y=parameters_dict['y'][0], z=parameters_dict['z'][0],
    #                         v_x=parameters_dict['vx'][0]*u.km/u.s, v_y=parameters_dict['vy'][0]*u.km/u.s, v_z=parameters_dict['vz'][0]*u.km/u.s)
    w0 = gc.PhaseSpacePosition(q = Quantity([parameters_dict['x'][0], parameters_dict['y'][0], parameters_dict['z'][0]], "kpc"),
                                p = Quantity([parameters_dict['vx'][0], parameters_dict['vy'][0],parameters_dict['vz'][0]], "km/s"),
                                )

    t_end = parameters_dict['t_end'][0] * u.Gyr.to(u.Myr)
    t_array = Quantity(-jnp.linspace(0, t_end, int(config.n_timesteps/2)), "Myr")


    prog_mass = Quantity(parameters_dict['m_progenitor'][0], "Msun")

    if config.df_type == "ChenStreamDF":
        df = gd.ChenStreamDF()
    elif config.df_type == "FardalStreamDF": 
        df = gd.FardalStreamDF()
    gen = gd.MockStreamGenerator(df, pot, )
    func = lambda k, t, w0, prog_mass: gen.run(k, t, w0, prog_mass, )
    stream, _ = func(jr.key(random_seed), t_array, w0, prog_mass)

    return jnp.array([stream.q.x.value, stream.q.y.value,stream.q.z.value, stream.p.x.value, stream.p.y.value, stream.p.z.value]).T




