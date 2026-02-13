from functools import partial
import astropy.units as u
import astropy.coordinates as coord
import numpy as np

import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic
from gala.dynamics import mockstream as ms



def simulate_stream_gala(parameters_dict, config, code_units, random_seed:int):
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
    pot = gp.CCompositePotential()

    pot['halo'] = gp.NFWPotential(m     = parameters_dict['m_Triaxial_halo'][0],
                                r_s   = parameters_dict['r_Triaxial_halo'][0],
                                a     = 1,
                                b     = parameters_dict['q1_Triaxial_halo'][0],
                                  c     = parameters_dict['q2_Triaxial_halo'][0],
                                units =galactic)

    pot['thin_disk'] = gp.MN3ExponentialDiskPotential(m = 4 * np.pi * parameters_dict['rho_thin_disk'][0]*parameters_dict['hr_thin_disk'][0]**2 * parameters_dict['hz_thin_disk'][0],
                                                    h_R=parameters_dict['hr_thin_disk'][0],
                                                    h_z=parameters_dict['hz_thin_disk'][0],
                                                    units=galactic,
                                                    positive_density=True)
    pot['thick_disk'] = gp.MN3ExponentialDiskPotential(m = 4 * np.pi * parameters_dict['rho_thick_disk'][0] *parameters_dict['hr_thick_disk'][0]**2 * parameters_dict['hz_thick_disk'][0],
                                                    h_R=parameters_dict['hr_thick_disk'][0],
                                                    h_z=parameters_dict['hz_thick_disk'][0],
                                                    units=galactic,
                                                    positive_density=True)
    pot['bulge'] = gp.PowerLawCutoffPotential(m=parameters_dict['m_bulge'][0],
                                            r_c=parameters_dict['r_bulge'][0],
                                            alpha=parameters_dict['alpha_bulge'][0],
                                            units=galactic)

    w0 = coord.Galactocentric(x=parameters_dict['x'][0]*u.kpc, y=parameters_dict['y'][0]*u.kpc, z=parameters_dict['z'][0]*u.kpc,
                            v_x=parameters_dict['vx'][0]*u.km/u.s, v_y=parameters_dict['vy'][0]*u.km/u.s, v_z=parameters_dict['vz'][0]*u.km/u.s)
    w0 = gd.PhaseSpacePosition(w0)

    prog_mass = parameters_dict['m_progenitor'][0] * u.Msun
    b_prog = parameters_dict['a_progenitor'][0] * u.pc
    prog_pot = gp.PlummerPotential(m=prog_mass, b=b_prog, units=galactic)
    if config.df_type == "ChenStreamDF":
        df = gd.ChenStreamDF()
    elif config.df_type == "FardalStreamDF":
        df = gd.FardalStreamDF()
    if config.use_prog_potential:
        gen = gd.MockStreamGenerator(df, pot, progenitor_potential=prog_pot)
    else:
        gen = gd.MockStreamGenerator(df, pot)
    stream, _ = gen.run(w0, prog_mass,
                        # n_particles=config.N_particles/2,
                        dt=-(parameters_dict['t_end']*u.Gyr.to(u.Myr)/config.n_timesteps), 
                        n_steps=config.n_timesteps, 
                        )
    return np.array([stream.x.to(u.kpc).value, stream.y.to(u.kpc).value,stream.z.to(u.kpc).value, 
                     stream.vel._d_x.to(u.km/u.s).value, stream.vel._d_y.to(u.km/u.s).value, stream.vel._d_z.to(u.km/u.s).value]).T
                     


# Add this helper function at the top of your file
def _run_gala_single(args):
    """Wrapper for single gala simulation - needed for multiprocessing"""
    param_dict, config, code_units, seed = args
    try:
        return simulate_stream_gala(param_dict, config, code_units, seed)
    except Exception as e:
        print(f"Gala simulation failed for seed {seed}: {e}")
        return None
    




    # prog_mass = Quantity(parameters_dict['m_progenitor'][0], "Msun")

    # if config.df_type == "ChenStreamDF":
    #     df = gd.ChenStreamDF()
    # elif config.df_type == "FardalStreamDF": 
    #     df = gd.FardalStreamDF()
    # gen = gd.MockStreamGenerator(df, pot)
    # func = lambda k, t, w0, prog_mass: gen.run(k, t, w0, prog_mass)
    # stream, _ = func(jr.key(random_seed), t_array, w0, prog_mass)

    # return jnp.array([stream.q.x.value, stream.q.y.value,stream.q.z.value, stream.p.x.value, stream.p.y.value, stream.p.z.value]).T




