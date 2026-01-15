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

@jit
def run_simulation(prog_mass, t_end, x_c, y_c, z_c, v_xc, v_yc, v_zc, m_nfw, r_s, gamma, key, n_stars):

    #initial phase-space position of the progenitor
    w = gc.PhaseSpacePosition(q=Quantity([x_c, y_c, z_c], unit="kpc"),
                              p=Quantity([v_xc, v_yc, v_zc], unit="km/s"))
    
    #progenitor mass
    prog_mass = Quantity(prog_mass, "Msun")
    
    #total time integration
    t_array = Quantity(-jnp.linspace(0, -t_end, n_stars), "Myr")

    #Milky Way base potential
    milky_way_pot = gp.BovyMWPotential2014()

    pot = gp.CompositePotential(
            halo = gp.gNFWPotential(m=m_nfw, 
                                   r_s=r_s, 
                                   gamma=gamma, units="galactic"),
            disk = milky_way_pot,
            bulge=milky_way_pot.bulge,
        )
    #
    # simulate stream
    df = gd.ChenStreamDF()
    gen = gd.MockStreamGenerator(df, pot)
    stream, _ = gen.run(key, t_array, w, prog_mass)
    stream_array = stream_to_array(stream)
    return stream_array

@jit
def simulate_stream(prog_mass, t_end, 
                    x_c, y_c, z_c, v_xc, v_yc, v_zc, 
                    m_nfw, r_s, gamma, 
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
        gamma = jnp.ones((n_streams,)) * gamma
    
    #generate data array
    # TO DO IS TO IMPLEMENT A VMAP AND BATCHED VERSION
    # key = random.PRNGKey(0)
    data = jnp.zeros((n_streams, 2*n_stars, 6))
    for s in range(n_streams):
        key = random.split(key, 2)[1]
        data[s] = jnp.array(
            run_simulation(
            prog_mass[s], t_end[s],
            x_c[s], y_c[s], z_c[s],
            v_xc[s], v_yc[s], v_zc[s],
            m_nfw[s], r_s[s], gamma[s],
            key, n_stars)
            )
    if n_streams == 1:
        data = data[0]
    return dict(sim_data=data)


##########
# priors #
##########

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
    gamma = np.random.uniform(0.3, 1.9,)  # unitless
    r_s = np.random.uniform(10.0, 30.0,)  # in kpc

    # Subject level: stream/progenitor parameters
    if j == 0:
        # GD-1 like stream
        prog_mass = np.random.uniform(1e4, 5e4,)  #
        t_end = np.random.uniform(3000.0, 5000.0,)  # in Myr
        x_c = np.random.uniform(-15.0, -5.0,)  # in kpc
        y_c = np.random.uniform(5.0, 15.0,)  # in kpc
        z_c = np.random.uniform(10.0, 20.0,)  # in kpc
        v_xc = np.random.uniform(-100.0, 0.0,)  # in km/s
        v_yc = np.random.uniform(100.0, 200.0,)  # in km/s
        v_zc = np.random.uniform(-200.0, -100.0,)  # in km/s
    elif j == 1:
        # Pal 5 like stream
        prog_mass = np.random.uniform(1e4, 5e4,)  #
        t_end = np.random.uniform(2000.0, 4000.0,)  # in Myr
        x_c = np.random.uniform(5.0, 15.0,)  # in kpc
        y_c = np.random.uniform(-15.0, -5.0,)  # in kpc
        z_c = np.random.uniform(-10.0, 0.0,)  # in kpc
        v_xc = np.random.uniform(0.0, 100.0,)  # in km/s
        v_yc = np.random.uniform(-200.0, -100.0,)  # in km/s
        v_zc = np.random.uniform(100.0, 200.0,)  # in km/s

    return dict(
        j=j,
        m_nfw=m_nfw,
        r_s=r_s,
        gamma=gamma,
        prog_mass=prog_mass,
        t_end=t_end,
        x_c=x_c,
        y_c=y_c,
        z_c=z_c,
        v_xc=v_xc,
        v_yc=v_yc,
        v_zc=v_zc
    )

# simulator_hierarchical = bf.make_simulator([sample_hierarchical_stream_priors, simulate_stream]) #not really used

    
    
    






    

    
