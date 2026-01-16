from autocvd import autocvd
autocvd(num_gpus = 1)

import os
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

# import bayesflow as bf
from functools import partial

import numpy as np
import jax
from jax import jit, random
import jax.numpy as jnp

from unxt import Quantity
import galax as gx
import galax.coordinates as gc
import galax.potential as gp
import galax.dynamics as gd

from astropy import units as u


##############
# simulation #
##############
@jit
def stream_to_array(stream):
    pos = jnp.array([stream.q.x.to('kpc').value, stream.q.y.to('kpc').value, stream.q.z.to('kpc').value]).T
    vel = jnp.array([stream.p.x.to('km/s').value, stream.p.y.to('km/s').value, stream.p.z.to('km/s').value]).T
    pos_vel = jnp.concatenate([pos, vel], axis=1)
    return pos_vel

@partial(jit, static_argnames=['n_stars'])
def run_simulation(prog_mass, t_end, x_c, y_c, z_c, v_xc, v_yc, v_zc, m_nfw, r_s, q1, q2, key, n_stars):

    #initial phase-space position of the progenitor
    w = gc.PhaseSpacePosition(q=Quantity([x_c, y_c, z_c], unit="kpc"),
                              p=Quantity([v_xc, v_yc, v_zc], unit="km/s"))
    
    #progenitor mass
    prog_mass = Quantity(prog_mass, "Msun")
    
    #total time integration
    t_array = Quantity(jnp.linspace(0, -t_end, int(n_stars/2)), "Myr")

    #Milky Way base potential
    milky_way_pot = gp.BovyMWPotential2014()

    pot = gp.CompositePotential(

            halo = gp.TriaxialNFWPotential(m=m_nfw, 
                                           r_s=r_s,
                                            q1=q1,
                                            q2=q2,           
                                            units="galactic"),
            disk = milky_way_pot,
            bulge=milky_way_pot.bulge,
        )
    
    # simulate stream
    df = gd.ChenStreamDF()
    gen = gd.MockStreamGenerator(df, pot)
    stream, _ = gen.run(key, t_array, w, prog_mass)
    stream_array = stream_to_array(stream)
    return stream_array

@partial(jit, static_argnames=['n_stars', 'n_streams'])
def simulate_stream(prog_mass, t_end, 
                    x_c, y_c, z_c, v_xc, v_yc, v_zc, 
                    m_nfw, r_s, q1, q2, j,
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
        q1 = jnp.ones((n_streams,)) * q1
        q2 = jnp.ones((n_streams,)) * q2
        
    
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
                m_nfw=m_nfw, r_s=r_s, q1=q1, q2=q2, 
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
                m_nfw=m_nfw, r_s=r_s, q1=q1, q2=q2,  #this are not array ! they are the shared parameters
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
    q1 = x["q1"]
    q2 = x["q2"]

    score = {
        "m_nfw": np.zeros_like(m_nfw),
        "r_s": np.zeros_like(r_s),
        "q1": np.zeros_like(q1),
        "q2": np.zeros_like(q2),
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
    q1 = np.random.normal(1.0, 0.05)
    q2 = np.random.uniform(0.5, 1.5)

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
        q1=q1,
        q2=q2,
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
    x_c = np.random.uniform(pos[0] - pos[0]*0.1, pos[0] + 0.1*pos[0])  # in kpc
    y_c = np.random.uniform(pos[1] - 0.1*pos[1]*0.1, pos[1] + 0.1*pos[1]*0.1 )  # in kpc
    z_c = np.random.uniform(pos[2] - 0.1*pos[2]*0.1, pos[2] + 0.1*pos[2]*0.1)  # in kpc
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
    x_c = np.random.uniform(pos[0] - pos[0]*0.1, pos[0] + 0.1*pos[0])  # in kpc
    y_c = np.random.uniform(pos[1] - 0.1*pos[1]*0.1, pos[1] + 0.1*pos[1]*0.1 )  # in kpc
    z_c = np.random.uniform(pos[2] - 0.1*pos[2]*0.1, pos[2] + 0.1*pos[2]*0.1)  # in kpc
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
        q1=1.0,
        q2=1.0,
        n_streams=1,
        n_stars=1000,
        key=key
    )
    print("sim_data shape:", sim_data['sim_data'].shape)  # (n_stars, 6)
    

    

    






    

    
