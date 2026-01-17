from autocvd import autocvd
autocvd(num_gpus = 1)

import os
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

# import bayesflow as bf


import numpy as np
import jax
from jax import jit, random
import jax.numpy as jnp
from astropy import units as u
from functools import partial
import astropy.coordinates as coord


import odisseo
from odisseo import construct_initial_state
from odisseo.integrators import leapfrog
from odisseo.dynamics import direct_acc, DIRECT_ACC, DIRECT_ACC_LAXMAP, DIRECT_ACC_FOR_LOOP, DIRECT_ACC_MATRIX
from odisseo.option_classes import SimulationConfig, SimulationParams, MNParams, NFWParams, PlummerParams, PSPParams, MN_POTENTIAL, NFW_POTENTIAL, PSP_POTENTIAL
from odisseo.initial_condition import Plummer_sphere, ic_two_body, sample_position_on_sphere, inclined_circular_velocity, sample_position_on_circle, inclined_position
from odisseo.utils import center_of_mass
from odisseo.time_integration import time_integration
from odisseo.units import CodeUnits
from odisseo.visualization import create_3d_gif, create_projection_gif, energy_angular_momentum_plot
from odisseo.potentials import MyamotoNagai, NFW
from odisseo.option_classes import DIFFRAX_BACKEND, TSIT5



from odisseo.utils import halo_to_gd1_velocity_vmap, halo_to_gd1_vmap, projection_on_GD1

code_length = 10 * u.kpc
code_mass = 1e4 * u.Msun
G = 1
code_time = 1 * u.Gyr
code_units = CodeUnits(code_length, code_mass, G=1, unit_time = code_time )  



##############
# simulation #
##############
@jit
def stream_codeunits2realunits(stream):
    pos = stream[:, :3] * code_units.code_length.to('kpc')
    vel = stream[:, 3:] * code_units.code_velocity.to('km/s')
    pos_vel = jnp.concatenate([pos, vel], axis=1)
    return pos_vel

@partial(jit, static_argnames=['n_stars'])
def run_simulation(prog_mass, t_end, x_c, y_c, z_c, v_xc, v_yc, v_zc, m_nfw, r_s, key, n_stars):


    config = SimulationConfig(N_particles = n_stars, 
                            return_snapshots = False, 
                            num_timesteps = 1000, 
                            external_accelerations=(NFW_POTENTIAL, MN_POTENTIAL, PSP_POTENTIAL), 
                            acceleration_scheme = DIRECT_ACC_MATRIX,
                            softening = (0.1 * u.pc).to(code_units.code_length).value,
                            integrator=DIFFRAX_BACKEND,
                            fixed_timestep=False,
                            diffrax_solver=TSIT5
                            ) #default values
    
    params = SimulationParams(t_end = t_end * (u.Myr).to(code_units.code_time),  
                          Plummer_params= PlummerParams(Mtot=prog_mass * u.Msun.to(code_units.code_mass),
                                                        a=8 * u.pc.to(code_units.code_length)
                                                        ),
                           MN_params= MNParams(M = 68_193_902_782.346756 * u.Msun.to(code_units.code_mass),
                                              a = 3.0 * u.kpc.to(code_units.code_length),
                                              b = 0.280 * u.kpc.to(code_units.code_length)
                                              ),
                          NFW_params= NFWParams(Mvir=m_nfw * u.Msun.to(code_units.code_mass),
                                               r_s= r_s * u.kpc.to(code_units.code_length)
                                               ),      
                          PSP_params= PSPParams(M = 4501365375.06545 * u.Msun.to(code_units.code_mass),
                                                alpha = 1.8, 
                                                r_c = 1.9 * u.kpc.to(code_units.code_length)),                    
                          G=code_units.G, ) 
    key = random.PRNGKey(0)

    #set up the particles in the initial state
    positions, velocities, mass = Plummer_sphere(key=key, params=params, config=config)
    #the center of mass needs to be integrated backwards in time first 
    config_com = config._replace(N_particles=1,)
    params_com = params._replace(t_end=-params.t_end,)


    #this is the final position of the cluster, we need to integrate backwards in time 
    pos_com_final = jnp.array([[x_c, y_c, z_c]]) * u.kpc.to(code_units.code_length)
    vel_com_final = jnp.array([[v_xc, v_yc, v_zc]]) * (u.km/u.s).to(code_units.code_velocity)
    mass_com = jnp.array([params_com.Plummer_params.Mtot])
    final_state_com = construct_initial_state(pos_com_final, vel_com_final) # state is a (N_particles x 2 x 3)

    #evolution in time
    snapshots_com = time_integration(final_state_com, mass_com, config_com, params_com)
    pos_com, vel_com = snapshots_com[:, 0], snapshots_com[:, 1]

    # Add the center of mass position and velocity to the Plummer sphere particles
    positions = positions + pos_com
    velocities = velocities + vel_com

    #initialize the initial state
    initial_state_stream = construct_initial_state(positions, velocities)

    #run the simulation
    final_state = time_integration(initial_state_stream, mass, config, params)
    # Reshape from (n_stars, 2, 3) to (n_stars, 6)
    # final_state[:, 0, :] is positions (n_stars, 3)
    # final_state[:, 1, :] is velocities (n_stars, 3)
    final_state = jnp.concatenate([final_state[:, 0, :], final_state[:, 1, :]], axis=1)  # (n_stars, 6)
    
    final_state = stream_codeunits2realunits(final_state)
    
    return final_state

@partial(jit, static_argnames=['n_stars', 'n_streams'])
def simulate_stream(prog_mass, t_end, 
                    x_c, y_c, z_c, v_xc, v_yc, v_zc, 
                    m_nfw, r_s, 
                    j, 
                    n_streams=1, n_stars=500, key=random.PRNGKey(0)):
    if isinstance(prog_mass, (float, int)):
        prog_mass = jnp.ones((n_streams,)) * prog_mass
        t_end = jnp.ones((n_streams,)) * t_end
        x_c = jnp.ones((n_streams,)) * x_c
        y_c = jnp.ones((n_streams,)) * y_c
        z_c = jnp.ones((n_streams,)) * z_c   
        v_xc = jnp.ones((n_streams,)) * v_xc
        v_yc = jnp.ones((n_streams,)) * v_yc
        v_zc = jnp.ones((n_streams,)) * v_zc
        m_nfw = jnp.ones((n_streams,)) * m_nfw
        r_s = jnp.ones((n_streams,)) * r_s
    
    #generate data array
    # TO DO IS TO IMPLEMENT A VMAP AND BATCHED VERSION
    # key = random.PRNGKey(0)
    data = jnp.zeros((n_streams, n_stars, 6))
    if n_streams == 1:
        for s in range(n_streams):
            key = random.split(key, 2)[1]
            data = data.at[s].set(jnp.array(
                run_simulation(
                prog_mass=prog_mass, t_end=t_end,
                x_c=x_c, y_c=y_c, z_c=z_c,
                v_xc=v_xc, v_yc=v_yc, v_zc=v_zc,
                m_nfw=m_nfw, r_s=r_s, 
                key=key, n_stars=n_stars)
                )
            )
            data = data[0]
    else:
        for s in range(n_streams):          #TODO: INSTEAD OF FOR LOOP USE A VMAP
            key = random.split(key, 2)[1]
            data = data.at[s].set(jnp.array(
                run_simulation(
                prog_mass=prog_mass[s], t_end=t_end[s],
                x_c=x_c[s], y_c=y_c[s], z_c=z_c[s],
                v_xc=v_xc[s], v_yc=v_yc[s], v_zc=v_zc[s],
                m_nfw=m_nfw, r_s=r_s, #this are not array ! they are the shared parameters
                key=key, n_stars=n_stars)
                )
            )
    return dict(sim_data=data)


##########
# priors #
##########

#score
def prior_global_score(x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    m_nfw = x["m_nfw"]
    r_s = x["r_s"]

    score = {
        "m_nfw": np.zeros_like(m_nfw),
        "r_s": np.zeros_like(r_s),
    }

    return score


def sample_hierarchical_stream_priors(n_streams=1):

    # Group level: 
    # - stream index j (what stream I am simulating) 
    # - Galaxy potential parameters
    
    if n_streams == 1:
        #if I ask to generate a single stream I return single values
        j = np.random.randint(0, 2)
    else:
        # I need all the streams
        # TO DO IS TO TAKE A SUBET OF INDICES WHEN n_stream < maximum number of streams simulated
        j = np.arange(0, n_streams, dtype=int)

    # Galaxy potential parameters
    m_nfw = np.random.uniform(0.5e12, 2.0e12,)  # in Msun
    r_s = np.random.uniform(10.0, 30.0,)  # in kpc

    if n_streams == 1:
    # Subject level: stream/progenitor parameters
        if j == 0:
            # GD-1 like stream
            samples_gd1_prior = sample_gd1_priors()
            prog_mass = samples_gd1_prior['prog_mass']
            t_end = samples_gd1_prior['t_end']
            x_c = samples_gd1_prior['x_c']
            y_c = samples_gd1_prior['y_c']
            z_c = samples_gd1_prior['z_c']
            v_xc = samples_gd1_prior['v_xc']
            v_yc = samples_gd1_prior['v_yc']    
            v_zc = samples_gd1_prior['v_zc']
        elif j == 1:
            # Pal 5 like stream
            samples_pal5_prior = sample_pal5_priors()
            prog_mass = samples_pal5_prior['prog_mass']
            t_end = samples_pal5_prior['t_end']
            x_c = samples_pal5_prior['x_c']
            y_c = samples_pal5_prior['y_c']
            z_c = samples_pal5_prior['z_c']
            v_xc = samples_pal5_prior['v_xc']
            v_yc = samples_pal5_prior['v_yc']    
            v_zc = samples_pal5_prior['v_zc']
    else:
        prog_mass = np.zeros((n_streams,))
        t_end = np.zeros((n_streams,))
        x_c = np.zeros((n_streams,))
        y_c = np.zeros((n_streams,))
        z_c = np.zeros((n_streams,))
        v_xc = np.zeros((n_streams,))
        v_yc = np.zeros((n_streams,))
        v_zc = np.zeros((n_streams,))
        for s in range(n_streams):
            if j[s] == 0:
                # GD-1 like stream
                samples_gd1_prior = sample_gd1_priors()
                prog_mass[s] = samples_gd1_prior['prog_mass']
                t_end[s] = samples_gd1_prior['t_end']
                x_c[s] = samples_gd1_prior['x_c']
                y_c[s] = samples_gd1_prior['y_c']
                z_c[s] = samples_gd1_prior['z_c']
                v_xc[s] = samples_gd1_prior['v_xc']
                v_yc[s] = samples_gd1_prior['v_yc']    
                v_zc[s] = samples_gd1_prior['v_zc']
            elif j[s] == 1:
                # Pal 5 like stream
                samples_pal5_prior = sample_pal5_priors()
                prog_mass[s] = samples_pal5_prior['prog_mass']
                t_end[s] = samples_pal5_prior['t_end']
                x_c[s] = samples_pal5_prior['x_c']
                y_c[s] = samples_pal5_prior['y_c']
                z_c[s] = samples_pal5_prior['z_c']
                v_xc[s] = samples_pal5_prior['v_xc']
                v_yc[s] = samples_pal5_prior['v_yc']    
                v_zc[s] = samples_pal5_prior['v_zc']

    return dict(
        j=j,
        m_nfw=m_nfw,
        r_s=r_s,
        prog_mass=prog_mass,
        t_end=t_end,
        x_c=x_c,
        y_c=y_c,
        z_c=z_c,
        v_xc=v_xc,
        v_yc=v_yc,
        v_zc=v_zc
    )


# prior GD1
def sample_gd1_priors():
    #from https://arxiv.org/pdf/2304.02032
    prog_mass = np.random.uniform(1e3, 10**4.5,)  #
    t_end = np.random.uniform(3000.0, 5000.0,)  # in Myr
    pos = (11.8,0.79,6.4) # kpc
    vel = (109.5, - 254.5, -90.3) #km / s
    x_c = np.random.uniform(pos[0] - 0.1*pos[0], pos[0] + 0.1*pos[0])  # in kpc
    y_c = np.random.uniform(pos[1] - 0.1*pos[1], pos[1] + 0.1*pos[1])  # in kpc
    z_c = np.random.uniform(pos[2] - 0.1*pos[2], pos[2] + 0.1*pos[2])  # in kpc
    v_xc = np.random.uniform(vel[0] - 0.1*vel[0], vel[0] + 0.1*vel[0])  # in km/s
    v_yc = np.random.uniform(vel[1] - 0.1*vel[1], vel[1] + 0.1*vel[1])  # in km/s
    v_zc = np.random.uniform(vel[2] - 0.1*vel[2], vel[2] + 0.1*vel[2])  # in km/s

    return dict(
        prog_mass=prog_mass,
        t_end=t_end,
        x_c=x_c,
        y_c=y_c,
        z_c=z_c,
        v_xc=v_xc,
        v_yc=v_yc,
        v_zc=v_zc
    )

#prior Pal5
def sample_pal5_priors():
    prog_mass = np.random.uniform(1e4, 5e4,)  #
    t_end = np.random.uniform(2000.0, 4000.0,)  # in Myr

    #from https://gala.adrian.pw/en/latest/tutorials/mock-stream-heliocentric.html#
    pos=(7.86390455, 0.22748727, 16.41622487) # kpc
    vel=(-42.35458106, -103.69384675, -15.48729026) #km / s
    x_c = np.random.uniform(pos[0] - 0.1*pos[0], pos[0] + 0.1*pos[0])  # in kpc
    y_c = np.random.uniform(pos[1] - 0.1*pos[1], pos[1] + 0.1*pos[1])  # in kpc
    z_c = np.random.uniform(pos[2] - 0.1*pos[2], pos[2] + 0.1*pos[2])  # in kpc
    v_xc = np.random.uniform(vel[0] - 0.1*vel[0], vel[0] + 0.1*vel[0])  # in km/s
    v_yc = np.random.uniform(vel[1] - 0.1*vel[1], vel[1] + 0.1*vel[1])  # in km/s
    v_zc = np.random.uniform(vel[2] - 0.1*vel[2], vel[2] + 0.1*vel[2])  # in km/s
    return dict(
        prog_mass=prog_mass,
        t_end=t_end,
        x_c=x_c,
        y_c=y_c,
        z_c=z_c,
        v_xc=v_xc,
        v_yc=v_yc,
        v_zc=v_zc
    )

if __name__ == "__main__":
    #test simulation
    key = random.PRNGKey(0)
    sim_data = simulate_stream(
        prog_mass=5e3,
        t_end=4000.0,
        x_c=11.8,
        y_c=0.79,
        z_c=6.4,
        v_xc=109.5,
        v_yc=-254.5,
        v_zc=-90.3,
        m_nfw=1e12,
        r_s=20.0,
        n_streams=1,
        n_stars=1000,
        key=key
    )
    print("sim_data shape:", sim_data['sim_data'].shape)  # (n_stars, 6)





    

    
