import agama
import numpy as np
import time

agama.setUnits(length=1, velocity=1, mass=1)
timeUnitGyr = agama.getUnits()['time'] / 1e3  # time unit is 1 kpc / (1 km/s)
# print('time unit: %.3f Gyr' % timeUnitGyr)



#### PArticle spray taken from AGAMA:

def get_rj_vj_R(pot_host, orbit_sat, mass_sat):
    """
    Compute the Jacobi radius, associated velocity, and rotation matrix
    for generating streams using particle-spray methods.
    Arguments:
        pot_host:  an instance of agama.Potential for the host galaxy.
        orbit_sat: the orbit of the satellite, an array of shape (N, 6).
        mass_sat:  the satellite mass (a single number or an array of length N).
    Return:
        rj:  Jacobi radius at each point on the orbit (length: N).
        vj:  velocity offset from the satellite at each point on the orbit (length: N).
        R:   rotation matrix converting from host to satellite at each point on the orbit (shape: N,3,3)
    """
    N = len(orbit_sat)
    x, y, z, vx, vy, vz = orbit_sat.T
    Lx = y * vz - z * vy
    Ly = z * vx - x * vz
    Lz = x * vy - y * vx
    r = (x*x + y*y + z*z)**0.5
    L = (Lx*Lx + Ly*Ly + Lz*Lz)**0.5
    # rotation matrices transforming from the host to the satellite frame for each point on the trajectory
    R = np.zeros((N, 3, 3))
    R[:,0,0] = x/r
    R[:,0,1] = y/r
    R[:,0,2] = z/r
    R[:,2,0] = Lx/L
    R[:,2,1] = Ly/L
    R[:,2,2] = Lz/L
    R[:,1,0] = R[:,0,2] * R[:,2,1] - R[:,0,1] * R[:,2,2]
    R[:,1,1] = R[:,0,0] * R[:,2,2] - R[:,0,2] * R[:,2,0]
    R[:,1,2] = R[:,0,1] * R[:,2,0] - R[:,0,0] * R[:,2,1]
    # compute  the second derivative of potential by spherical radius
    der = pot_host.eval(orbit_sat[:,0:3], der=True)
    d2Phi_dr2 = -(x**2  * der[:,0] + y**2  * der[:,1] + z**2  * der[:,2] +
                  2*x*y * der[:,3] + 2*y*z * der[:,4] + 2*z*x * der[:,5]) / r**2
    # compute the Jacobi radius and the relative velocity at this radius for each point on the trajectory
    Omega = L / r**2
    rj = (agama.G * mass_sat / (Omega**2 - d2Phi_dr2))**(1./3)
    vj = Omega * rj
    return rj, vj, R

def create_ic_particle_spray(orbit_sat, rj, vj, R, gala_modified=True):
    """
    Create initial conditions for particles escaping through Largange points,
    using the method of Fardal+2015
    Arguments:
        orbit_sat:  the orbit of the satellite, an array of shape (N, 6).
        rj:  Jacobi radius at each point on the orbit (length: N).
        vj:  velocity offset from the satellite at each point on the orbit (length: N).
        R:   rotation matrix converting from host to satellite at each point on the orbit (shape: N,3,3)
        gala_modified:  if True, use modified parameters as in Gala, otherwise the ones from the original paper.
    Return:
        initial conditions for stream particles, an array of shape (2*N, 6) - 
        two points for each point on the original satellite trajectory.
    """
    N = len(rj)
    # assign positions and velocities (in the satellite reference frame) of particles
    # leaving the satellite at both lagrange points (interleaving positive and negative offsets).
    rj = np.repeat(rj, 2) * np.tile([1, -1], N)
    vj = np.repeat(vj, 2) * np.tile([1, -1], N)
    R  = np.repeat(R, 2, axis=0)
    mean_x  = 2.0
    disp_x  = 0.5 if gala_modified else 0.4
    disp_z  = 0.5
    mean_vy = 0.3
    disp_vy = 0.5 if gala_modified else 0.4
    disp_vz = 0.5
    rx  = np.random.normal(size=2*N) * disp_x + mean_x
    rz  = np.random.normal(size=2*N) * disp_z * rj
    rvy =(np.random.normal(size=2*N) * disp_vy + mean_vy) * vj * (rx if gala_modified else 1)
    rvz = np.random.normal(size=2*N) * disp_vz * vj
    rx *= rj
    offset_pos = np.column_stack([rx,  rx*0, rz ])  # position and velocity of particles in the reference frame
    offset_vel = np.column_stack([rx*0, rvy, rvz])  # centered on the progenitor and aligned with its orbit
    ic_stream = np.tile(orbit_sat, 2).reshape(2*N, 6)   # same but in the host-centered frame
    ic_stream[:,0:3] += np.einsum('ni,nij->nj', offset_pos, R)
    ic_stream[:,3:6] += np.einsum('ni,nij->nj', offset_vel, R)
    return ic_stream


def create_stream_particle_spray(time_total, num_particles, pot_host, posvel_sat, mass_sat, gala_modified=True):
    """
    Construct a stream using the particle-spray method.
    Arguments:
        time_total:  duration of time for stream generation 
            (positive; orbit of the progenitor integrated from present day (t=0) back to time -time_total).
        num_particles:  number of points in the stream (even; divided equally between leading and trailing arms).
        pot_host:    an instance of agama.Potential for the host galaxy.
        posvel_sat:  present-day position and velocity of the satellite (array of length 6).
        mass_sat:    the satellite mass (a single number or an array of length num_particles//2).
        gala_modified:  if True, use modified parameters as in Gala, otherwise the ones from the original paper.
    Return:
        xv_stream: position and velocity of stream particles at present time (shape: num_particles, 6).
    """
    # number of points on the orbit: each point produces two stream particles (leading and trailing arms)
    N = num_particles//2

    # integrate the orbit of the progenitor from its present-day posvel (at time t=0)
    # back in time for an interval time_total, storing the trajectory at N points
    time_sat, orbit_sat = agama.orbit(
        potential=pot_host, ic=posvel_sat, time=-time_total, trajsize=N+1)
    # remove the 0th point (the present-day posvel) and reverse the arrays to make them increasing in time
    time_sat  = time_sat [1:][::-1]
    orbit_sat = orbit_sat[1:][::-1]

    # at each point on the trajectory, create a pair of seed initial conditions
    # for particles released at both Lagrange points
    rj, vj, R = get_rj_vj_R(pot_host, orbit_sat, mass_sat)
    ic_stream = create_ic_particle_spray(orbit_sat, rj, vj, R, gala_modified)
    time_seed = np.repeat(time_sat, 2)
    result = agama.orbit(
        potential=pot_host, ic=ic_stream,
        timestart=time_seed,  # starting time for each orbit (negative)
        time=-time_seed,      # integration time for each orbit: time_start + time = 0 (end time, i.e. now)
        trajsize=1)           # each orbit produces just one point (at the end of the integration)
    # 0th column is the array of times (here, just one element for each orbit, i.e. its end time),
    # 1st column is the trajectory (here, one end point per orbit)
    xv_stream = np.vstack(result[:,1])
    
    return xv_stream

def create_stream_particle_spray_with_progenitor_and_perturber(
    time_total, num_particles, pot_host, posvel_sat, mass_sat, radius_sat, gala_modified=True, add_perturber=False):
    """
    Construct a stream using the particle-spray method.
    Arguments:
        time_total:  duration of time for stream generation 
            (positive; orbit of the progenitor integrated from present day (t=0) back to time -time_total).
        num_particles:  number of points in the stream (even; divided equally between leading and trailing arms).
        pot_host:    an instance of agama.Potential for the host galaxy.
        posvel_sat:  present-day position and velocity of the satellite (array of length 6).
        mass_sat:    the satellite mass (a single number).
        radius_sat:  the scale radius of the satellite (assuming a Plummer profile).
        gala_modified:  if True, use modified parameters as in Gala, otherwise the ones from the original paper.
        add_perturber:  if True, add a perturbation from a fiducial flyby of a dark subhalo
        (no tunable parameters are available - this code should be adapted to specific scenarios).
    Return:
        xv_stream: position and velocity of stream particles at present time, evolved in the host potential only
        (shape: num_particles, 6), THIS WAS REMOVED
        xv_perturbed: same but including the potential of the progenitor and the perturber.
    """
    # number of points on the orbit: each point produces two stream particles (leading and trailing arms)
    N = num_particles//2

    # integrate the orbit of the progenitor from its present-day posvel (at time t=0)
    # back in time for an interval time_total, storing the trajectory at N points
    time_init = -time_total
    time_sat, orbit_sat = agama.orbit(
        potential=pot_host, ic=posvel_sat, time=time_init, trajsize=N+1, accuracy=1e-10)
    # remove the 0th point (the present-day posvel) and reverse the arrays to make them increasing in time
    time_sat  = time_sat [1:][::-1]
    orbit_sat = orbit_sat[1:][::-1]

    # at each point on the trajectory, create a pair of seed initial conditions
    # for particles released at both Lagrange points
    rj, vj, R = get_rj_vj_R(pot_host, orbit_sat, mass_sat)
    ic_stream = create_ic_particle_spray(orbit_sat, rj, vj, R, gala_modified)
    time_seed = np.repeat(time_sat, 2)
    # result = agama.orbit(
    #     potential=pot_host, ic=ic_stream,
    #     timestart=time_seed,  # starting time for each orbit (negative)
    #     time=-time_seed,      # integration time for each orbit: time_start + time = 0 (end time, i.e. now)
    #     trajsize=1)           # each orbit produces just one point (at the end of the integration)
    # # 0th column is the array of times (here, just one element for each orbit, i.e. its end time),
    # # 1st column is the trajectory (here, one end point per orbit)
    # xv_stream = np.vstack(result[:,1])

    # the gravitational potential of the progenitor moving on its orbit
    pot_sat = agama.Potential(
        type='Plummer', mass=mass_sat, scaleRadius=radius_sat, center=np.column_stack([time_sat, orbit_sat]))

    # the perturber: a dark subhalo flying through the stream track at some point denoted as
    # "xv_perturber_flyby", which coincides with the point on the actual orbit of the progenitor,
    # but has a different velocity, and the flyby time is slightly later than the time at which
    # the progenitor passes through the same point.
    xv_perturber_flyby = orbit_sat[len(orbit_sat)//2]  # crossing point at the middle of the progenitor orbit
    xv_perturber_flyby[3:6] += np.random.normal(size=3) * 50.0  # modify the velocity of the perturber
    time_flyby = -time_total * 0.4   # set the flyby at a different time than the progenitor

    # We could have approximated the perturber's motion as a straight line, but to make it more realistic,
    # we put it onto an actual orbit in the same Galactic potential.
    # The way to construct such an orbit is slightly convolved:
    # First, we integrate the orbit from the flyby time and corresponding point back in time
    # to the initial time of the stream simulation.
    # The returned value is a tuple (timestamps, trajectory), of which we need only the trajectory,
    # and select the last (and only) point, i.e. the position/velocity of the perturber at time_init
    if add_perturber:
        xv_perturber_init = agama.orbit(
            potential=pot_host, ic=xv_perturber_flyby, time=time_init-time_flyby, timestart=time_flyby, trajsize=1
        )[1][0]
        # Then we integrate the perturber's orbit forward in time from this initial point up to present,
        # storing the output at every internal timestep of the integrator (requested by trajsize=0)
        traj_perturber = np.column_stack(agama.orbit(
            potential=pot_host, ic=xv_perturber_init, time=time_total, timestart=time_init, trajsize=0))
        # and finally, create the NFW potential of the perturber moving on its orbit
        pot_perturber = agama.Potential(type='nfw', mass=2e7, scaleRadius=0.05, center=traj_perturber)

    # The total potential is now composed of three parts: the host galaxy, the progenitor, and the perturber
        pot_total = agama.Potential(pot_host, pot_sat, pot_perturber)
    else:
        pot_total = agama.Potential(pot_host, pot_sat)  # less dramatic

    # create a version of the stream in the new potential
    # xv_stream_perturbed = np.vstack(agama.orbit(
    #     potential=pot_total, ic=ic_stream, timestart=time_seed, time=-time_seed, trajsize=1)[:,1])

    # return xv_stream_perturbed

    # --- FIX: filter out NaN ICs caused by invalid rj (Omega^2 - d2Phi_dr2 < 0) ---
    valid_mask = np.isfinite(ic_stream).all(axis=1)
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        import warnings
        warnings.warn(
            f"{n_invalid}/{len(ic_stream)} stream particles have NaN ICs "
            f"(rj undefined where Omega^2 - d2Phi_dr2 < 0); filling with NaN.",
            RuntimeWarning
        )

    ic_valid    = ic_stream[valid_mask]
    time_valid  = time_seed[valid_mask]

    result = agama.orbit(
        potential=pot_total, ic=ic_valid, timestart=time_valid,
        time=-time_valid, trajsize=1, accuracy=1e-10)
    xv_valid = np.vstack(result[:, 1])

    # Reinsert NaN rows so output shape is always (num_particles, 6)
    xv_stream_perturbed = np.full((len(ic_stream), 6), np.nan)
    xv_stream_perturbed[valid_mask] = xv_valid
    # -------------------------------------------------------------------------------

    return xv_stream_perturbed



def create_stream_restricted_nbody(
    time_total, num_particles, pot_host, posvel_sat, mass_sat, radius_sat):
    """
    Create a stream using the "restricted N-body method", which follow the orbits of particles 
    both inside the progenitor and after they become stripped, and updates the potential of the progenitor
    self-consistently (approximating it as spherical).
    Arguments:
        time_total:  duration of time for stream generation 
            (positive; orbit of the progenitor integrated from present day (t=0) back to time -time_total).
        num_particles:  number of particles in the simulation.
        pot_host:    an instance of agama.Potential for the host galaxy.
        posvel_sat:  present-day position and velocity of the satellite (array of length 6).
        mass_sat:    initial satellite mass.
        radius_sat:  initial scale radius of the Plummer profile of the satellite.
    Return:
        xv_stream: position and velocity of stream particles at present time (shape: num_particles, 6).

    """
    
    # integrate the orbit of the progenitor from its present-day posvel (at time t=0) back in time
    # for an interval time_total, storing the trajectory at every timestep of the ODE integrator
    # (as specified by trajsize=0) to ensure an accurate interpolation
    time_sat, orbit_sat = agama.orbit(potential=pot_host, ic=posvel_sat, time=-time_total, trajsize=0)

    # reverse the arrays to make them increasing in time
    time_sat  = time_sat [::-1]
    orbit_sat = orbit_sat[::-1]
    # create the spline interpolators for each coordinate of the progenitor trajectory
    traj_x = agama.Spline(time_sat, orbit_sat[:,0], der=orbit_sat[:,3])
    traj_y = agama.Spline(time_sat, orbit_sat[:,1], der=orbit_sat[:,4])
    traj_z = agama.Spline(time_sat, orbit_sat[:,2], der=orbit_sat[:,5])

    # initial potential of the satellite (a Plummer profile)
    pot_sat  = agama.Potential(type='Plummer', mass=mass_sat, scaleRadius=radius_sat)

    # create a spherical isotropic DF for the satellite and sample it with particles
    # (note: the sampling will be different each time you run this cell, it does not use numpy random seed)
    df_sat = agama.DistributionFunction(type='QuasiSpherical', potential=pot_sat)
    xv, mass = agama.GalaxyModel(pot_sat, df_sat).sample(num_particles)
    print('velocity dispersion in the progenitor: %g km/s' % np.mean(xv[:,3:6]**2 / 3)**0.5)

    # shift the particles to the initial position/velocity of the satellite
    xv += orbit_sat[0]

    time_begin = -time_total  # start of the simulation
    time_end   = 0.0
    time_index = 0
    # wallclock_begin = time.time()
    while time_begin < time_end:
        # update interval is set to 10 timesteps of the original orbit integrator,
        # this choice makes it possible to resolve the rapid evolution around pericentre passages
        # in sufficient detail (though we still approximate the satellite potential as spherical)
        time_index = min(time_index+10, len(time_sat)-1)

        # initialize the time-dependent total potential (host + moving sat) on this time interval
        pot_sat_moving = agama.Potential(potential=pot_sat, center=np.column_stack([time_sat, orbit_sat]))
        pot_total = agama.Potential(pot_host, pot_sat_moving)

        # compute the trajectories of all particles moving in the combined potential of
        # the host galaxy and the moving satellite.
        # to speed up computations, we use a less stringent accuracy parameter than the default value of 1e-8;
        # however, if one further loosens it to 1e-4, the satellite gets artificially disrupted too quickly!
        xv = np.vstack(agama.orbit(ic=xv, potential=pot_total, trajsize=1,
            time=time_sat[time_index]-time_begin, timestart=time_begin, verbose=False, accuracy=1e-12)[:,1])

        # update the potential of the satellite (using a spherical monopole approximation),
        # subtracting the position of the satellite centre at the current time from particle coords
        time_begin = time_sat[time_index]
        xv_rel = xv - orbit_sat[time_index]
        pot_sat = agama.Potential(type='multipole', particles=(xv_rel[:,0:3], mass), symmetry='s')
        bound = pot_sat.potential(xv_rel[:,0:3]) + 0.5 * np.sum(xv_rel[:,3:6]**2, axis=1) < 0
        print('t=%6.3f, Phi(0)=%6.2f, bound mass=%5.0f' %
            (time_begin, pot_sat.potential(0,0,0), np.sum(mass[bound])))
    
    # print('wallclock time: %.1f s' % (time.time() - wallclock_begin))
    return xv
    # return orbit_sat


def simulate_stream_agama(parameters_dict, config, code_units, random_seed:int):
    """
    Simulate a stream using the particle-spray method implemented in AGAMA.
    Arguments:
        parameters_dict: dictionary containing the parameters for the simulation, including:
            - 'pot_host': an instance of agama.Potential for the host galaxy.
            - 'posvel_sat': present-day position and velocity of the satellite (array of length 6).
            - 'mass_sat': the satellite mass (a single number or an array of length num_particles//2).
        config: dictionary containing the configuration for the simulation, including:
            - 'time_total': duration of time for stream generation (positive; orbit of the progenitor integrated from present day (t=0) back to time -time_total).
            - ' num_particles': number of points in the stream (even; divided equally between leading and trailing arms).
            - 'gala_modified': if True, use modified parameters as in Gala, otherwise the ones from the original paper.         
    """
    #external potential with the opitized initialization
    pot_params = [
        dict(type='Spheroid', 
            scaleRadius=75/1e3, 
            densityNorm=9.6e10,
            gamma=0, 
            alpha=1, 
            beta=1.8, 
            cutoffStrength=2,
            outerCutoffRadius=2.1,
            axisRatioY=1.0,
            axisRatioZ=0.5), #bulge
        dict(type='Spheroid', 
            scaleRadius=parameters_dict['a_TwoPowerTriaxial_halo'][0], 
            densityNorm=parameters_dict['rho_TwoPowerTriaxial_halo'][0],
            gamma=parameters_dict['gamma_TwoPowerTriaxial_halo'][0], 
            alpha=1, 
            beta=parameters_dict['beta_TwoPowerTriaxial_halo'][0], #fixed 
            cutoffStrength=2, #fixed
            outerCutoffRadius=np.inf, #fixed
            axisRatioY=1.0, #fixed
            axisRatioZ=parameters_dict['q_TwoPowerTriaxial_halo'][0]), #halo
        dict(type='Disk', 
             scaleRadius=parameters_dict['r_Disk'][0], 
             scaleHeight=parameters_dict['z_Disk'][0], 
             surfaceDensity=parameters_dict['Sigma_Disk'][0],
             sersicIndex=1, #fixed
             innerCutoffRadius = 0, #fixed
             ) #Disk 
        
    ]
    pot_host = agama.Potential(*pot_params) 

    ra0, dec0, dist0, pmra0, pmdec0, vlos0 = parameters_dict['ra'][0], parameters_dict['dec'][0], parameters_dict['r'][0], parameters_dict['mu_ra_cosdec'][0], parameters_dict['mu_dec'][0], parameters_dict['vr'][0]
    l0, b0, pml0, pmb0 = agama.transformCelestialCoords(agama.fromICRStoGalactic, ra0*np.pi/180, dec0*np.pi/180, pmra0, pmdec0)
    posvel_sat = agama.getGalactocentricFromGalactic(l0, b0, dist0, pml0*4.74, pmb0*4.74, vlos0)

    mass_sat   = parameters_dict['m_progenitor'][0]  # in Msun
    radius_sat = parameters_dict['a_progenitor'][0] / 1e3  # in kpc
    time_total = parameters_dict['t_end'][0] / 0.978    # in time units (0.978 Gyr)
    print('time total:', time_total)
    num_particles = int(config.N_particles)  # number of particles in the stream

    xv_stream_perturbed = create_stream_particle_spray_with_progenitor_and_perturber(time_total, 
                                                                                     num_particles, 
                                                                                     pot_host, 
                                                                                     posvel_sat, 
                                                                                     mass_sat, 
                                                                                     radius_sat, 
                                                                                     add_perturber=False)
    # xv_stream_perturbed = create_stream_restricted_nbody(time_total, 
    #                                                      num_particles, 
    #                                                      pot_host, 
    #                                                      posvel_sat, 
    #                                                      mass_sat, 
    #                                                      radius_sat)  # make the radius larger to facilitate the tidal disruption
    
    # print('Stream simulated with AGAMA for parameters:', parameters_dict)
    # print('Stream shape:', xv_stream_perturbed.shape)
    return xv_stream_perturbed