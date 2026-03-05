from functools import partial

from jax import jit, random
import jax.numpy as jnp
import astropy.units as u

from odisseo import construct_initial_state
from odisseo.units import CodeUnits
from odisseo.time_integration import time_integration
from odisseo.initial_condition import Plummer_sphere
from odisseo.option_classes import SimulationConfig, SimulationParams
from odisseo.option_classes import (PlummerParams, 
                                    MNParams, 
                                    NFWParams, 
                                    PSPParams, 
                                    TriaxialNFWParams, 
                                    TwoPowerTriaxialParams, 
                                    ThinMN3DiskParams, 
                                    ThickMN3DiskParams)
from odisseo.option_classes import (TRIAXIAL_NFW_POTENTIAL,
                                    TWO_POWER_TRIAXIAL, 
                                    THIN_MN3_DISK,
                                    THICK_MN3_DISK, 
                                    PSP_POTENTIAL,
                                    DIRECT_ACC_MATRIX, 
                                    DIFFRAX_BACKEND,
                                    TSIT5,
)

def convert_to_integer_externalacc(ext_acc):
    external_acc_int = []
    for acc in ext_acc:
        if acc == 'PSP_POTENTIAL':
            external_acc_int.append(PSP_POTENTIAL)
        elif acc == 'TWO_POWER_TRIAXIAL':
            external_acc_int.append(TWO_POWER_TRIAXIAL)
        elif acc == 'THIN_MN3_DISK':
            external_acc_int.append(THIN_MN3_DISK)
        elif acc == 'THICK_MN3_DISK':
            external_acc_int.append(THICK_MN3_DISK)
        elif acc == 'TRIAXIAL_NFW_POTENTIAL':
            external_acc_int.append(TRIAXIAL_NFW_POTENTIAL)
    return tuple(external_acc_int)

def convert_to_integer_config(config):
    if config == 'DIRECT_ACC_MATRIX':
        return DIRECT_ACC_MATRIX
    elif config == 'DIFFRAX_BACKEND':
        return DIFFRAX_BACKEND
    elif config == 'TSIT5':
        return TSIT5
    
@partial(jit, static_argnames=('config', 'code_units'))
def simulate_stream_odisseo(parameters_dict, config: SimulationConfig, code_units: CodeUnits, random_seed:int):
    params = SimulationParams(
        t_end = parameters_dict['t_end'][0] * u.Gyr.to(code_units.code_time),  
        Plummer_params = PlummerParams(Mtot=parameters_dict['m_progenitor'] * u.Msun.to(code_units.code_mass), 
                                      a=parameters_dict['a_progenitor'] * u.pc.to(code_units.code_length)),  
        PSP_params = PSPParams(M = parameters_dict['m_bulge'] * u.Msun.to(code_units.code_mass),
                              alpha = parameters_dict['alpha_bulge'], 
                              r_c = parameters_dict['r_bulge'] * u.kpc.to(code_units.code_length)),  
        TriaxialNFW_params = TriaxialNFWParams(Mvir = parameters_dict['m_Triaxial_halo'] * u.Msun.to(code_units.code_mass),
                                                r_s = parameters_dict['r_Triaxial_halo'] * u.kpc.to(code_units.code_length),
                                                q1 = parameters_dict['q1_Triaxial_halo'],
                                                q2 = parameters_dict['q2_Triaxial_halo'],),
        TwoPowerTriaxial_params = TwoPowerTriaxialParams(rho = parameters_dict['rho_TwoPowerTriaxial_halo']* (u.Msun / u.kpc**3).to(code_units.code_mass / code_units.code_length**3),
                                                        a = parameters_dict['a_TwoPowerTriaxial_halo'] * u.kpc.to(code_units.code_length),
                                                        b = parameters_dict['b_TwoPowerTriaxial_halo'] , 
                                                        c = parameters_dict['c_TwoPowerTriaxial_halo'],
                                                        alpha = parameters_dict['alpha_TwoPowerTriaxial_halo'],
                                                        beta = parameters_dict['beta_TwoPowerTriaxial_halo'],),
        ThinMN3Disk_params= ThinMN3DiskParams(M = (4 * jnp.pi * parameters_dict['rho_thin_disk'] * parameters_dict['hr_thin_disk']**2 * parameters_dict['hz_thin_disk']) * u.Msun.to(code_units.code_mass),
                                              hr = parameters_dict['hr_thin_disk'] * u.kpc.to(code_units.code_length),
                                              hz = parameters_dict['hz_thin_disk'] * u.kpc.to(code_units.code_length),),
        ThickMN3Disk_params= ThickMN3DiskParams(M = (4 * jnp.pi * parameters_dict['rho_thick_disk'] * parameters_dict['hr_thick_disk']**2 * parameters_dict['hz_thick_disk']) * u.Msun.to(code_units.code_mass),
                                                hr = parameters_dict['hr_thick_disk'] * u.kpc.to(code_units.code_length),
                                                hz = parameters_dict['hz_thick_disk'] * u.kpc.to(code_units.code_length),),
        G=code_units.G, ) 
    

    #the center of mass needs to be integrated backwards in time first 
    config_com = config._replace(N_particles=1,)
    params_com = params._replace(t_end=-params.t_end,)

    #this is the final position of the cluster, we need to integrate backwards in time 
    pos_com_final = jnp.array([[parameters_dict['x'].squeeze(), parameters_dict['y'].squeeze(), parameters_dict['z'].squeeze()]]) * u.kpc.to(code_units.code_length)
    vel_com_final = jnp.array([[parameters_dict['vx'].squeeze(), parameters_dict['vy'].squeeze(), parameters_dict['vz'].squeeze()]]) * (u.km/u.s).to(code_units.code_velocity)
    mass_com = jnp.array([params_com.Plummer_params.Mtot])
    #we construmt the initial state of the com 
    initial_state_com = construct_initial_state(pos_com_final, vel_com_final,)
    #we run the simulation backwards in time for the center of mass
    final_state_com = time_integration(initial_state_com, mass_com, config=config_com, params=params_com)
    #we calculate the final position and velocity of the center of mass
    pos_com = final_state_com[:, 0]
    vel_com = final_state_com[:, 1]
    
    key = random.PRNGKey(random_seed)
    #set up the particles in the initial state
    positions, velocities, mass = Plummer_sphere(key=key, params=params, config=config)
    #we add the center of mass position and velocity to the Plummer sphere particles
    positions = positions + pos_com
    velocities = velocities + vel_com
    #initialize the initial state
    initial_state_stream = construct_initial_state(positions, velocities, )
    #run the simulation
    stream = time_integration(initial_state_stream, mass, config=config, params=params) #this is in galactic coordinates and code units

    stream = stream.at[:, 0].set(stream[:, 0] * code_units.code_length.to(u.kpc))  # all the positions in kpc
    stream = stream.at[:, 1].set(stream[:, 1] * code_units.code_velocity.to(u.km/u.s))  # all the velocities in km/s
    
    return stream.reshape(-1, 6)