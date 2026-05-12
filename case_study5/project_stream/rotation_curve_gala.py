
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# import gala.coordinates as gc
# import gala.dynamics as gd
# import gala.potential as gp
# from gala.units import galactic
# from gala.dynamics import mockstream as ms
# from gala.potential import scf
# from astropy import units as u  




# data = dict(np.load('./data/streams/data_gala/training_data_300000.npz'))
# N_sample = 60_000
# data = {k: data[k][:N_sample] for k in data.keys()}
# print(f"loaded {N_sample} files")
# params = ['m_Triaxial_halo', 
#           'r_Triaxial_halo', 
#           'q2_Triaxial_halo', 
#           'rho_thin_disk', 
#           'hr_thin_disk', 
#           'hz_thin_disk', 
#           'rho_thick_disk', 
#           'hr_thick_disk', 
#           'hz_thick_disk', 
#           'm_bulge', 
#           'r_bulge', 
#           'alpha_bulge']
# parameters_dict = {k: data[k] for k in params}
# for k in parameters_dict.keys():
#     print(f"{k}: {parameters_dict[k].shape}")
# r_array = np.linspace(1, 30, 100) * u.kpc
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111)
# colors = plt.cm.viridis(np.linspace(0, 1, N_sample))
# for i in tqdm(range(N_sample)):
#     pot = gp.CCompositePotential()
#     pot['halo'] = gp.NFWPotential(m     = parameters_dict['m_Triaxial_halo'][i][0],
#                                 r_s   = parameters_dict['r_Triaxial_halo'][i][0],
#                                 a     = 1,
#                                 b     = 1,
#                                 c     = parameters_dict['q2_Triaxial_halo'][i][0],
#                                 units =galactic)

#     pot['thin_disk'] = gp.MN3ExponentialDiskPotential(m = 4 * np.pi * parameters_dict['rho_thin_disk'][i][0]*parameters_dict['hr_thin_disk'][i][0]**2 * parameters_dict['hz_thin_disk'][i][0],
#                                                     h_R=parameters_dict['hr_thin_disk'][i][0],
#                                                     h_z=parameters_dict['hz_thin_disk'][i][0],
#                                                     units=galactic,
#                                                     positive_density=True)
#     pot['thick_disk'] = gp.MN3ExponentialDiskPotential(m = 4 * np.pi * parameters_dict['rho_thick_disk'][i][0] *parameters_dict['hr_thick_disk'][i][0]**2 * parameters_dict['hz_thick_disk'][i][0],
#                                                     h_R=parameters_dict['hr_thick_disk'][i][0],
#                                                     h_z=parameters_dict['hz_thick_disk'][i][0],
#                                                     units=galactic,
#                                                     positive_density=True)
#     pot['bulge'] = gp.PowerLawCutoffPotential(m=parameters_dict['m_bulge'][i][0],
#                                             r_c=parameters_dict['r_bulge'][i][0],
#                                             alpha=parameters_dict['alpha_bulge'][i][0],
#                                             units=galactic)
#     v_circ = pot.circular_velocity(R=r_array, z=np.zeros_like(r_array))
#     ax.plot(r_array, v_circ, color=colors[i], alpha=0.05, )
# ax.set_xlabel('Radius (kpc)')
# ax.set_ylabel('Circular Velocity (km/s)')
# ax.axhline(190, color='red', lw=0.5, ls='--')
# ax.axhline(260, color='red', lw=0.5, ls='--')

# fig.savefig('./data/plots/gala_rotcurv/rotation_curves.png', dpi=300)
# print('Saved at ./data/plots/gala_rotcurv/rotation_curves.png')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import corner

import gala.potential as gp
from gala.units import galactic
from astropy import units as u

# ── Observed rotation curve (Eilers et al. or similar) ───────────────────────
obs_R   = np.array([5.24,5.74,6.25,6.77,7.23,7.83,8.21,8.78,9.26,9.75,
                    10.25,10.75,11.25,11.75,12.24,12.74,13.25,13.74,14.23,14.74,
                    15.23,15.74,16.24,16.74,17.23,17.74,18.35,18.90,19.50,20.41,
                    21.28,22.39,23.16,24.00])          # kpc
obs_Vc  = np.array([225.10,233.53,234.30,233.17,236.19,236.00,233.19,233.15,232.15,231.24,
                    230.34,230.54,229.11,227.48,226.69,225.56,224.90,223.57,221.10,220.19,
                    219.59,217.36,216.61,217.28,216.25,213.81,217.53,212.10,210.46,206.69,
                    207.71,203.72,205.20,200.64])       # km/s
obs_sVc = np.array([0.69,0.68,0.62,0.60,0.45,0.29,0.26,0.22,0.17,0.16,
                    0.17,0.18,0.19,0.20,0.25,0.27,0.27,0.31,0.40,0.43,
                    0.50,0.68,0.74,0.87,1.02,1.15,1.45,1.58,1.32,1.71,
                    1.69,2.01,2.50,4.94])              # km/s  (1σ)

N_obs   = len(obs_R)
r_array = obs_R * u.kpc

# ── Load data ─────────────────────────────────────────────────────────────────
data     = dict(np.load('./data/streams/data_gala/training_data_300000.npz'))
N_sample = 300_000
data     = {k: data[k][:N_sample] for k in data.keys()}
print(f"Loaded {N_sample} samples")

params = ['m_Triaxial_halo','r_Triaxial_halo','q2_Triaxial_halo',
          'rho_thin_disk','hr_thin_disk','hz_thin_disk',
          'rho_thick_disk','hr_thick_disk','hz_thick_disk',
          'm_bulge','r_bulge','alpha_bulge']
parameters_dict = {k: data[k] for k in params}

# ── Main loop ─────────────────────────────────────────────────────────────────
all_vcirc  = np.zeros((N_sample, N_obs))   # km/s, stored for reuse
accepted   = np.zeros(N_sample, dtype=bool)

for i in tqdm(range(N_sample)):
    p = {k: parameters_dict[k][i][0] for k in params}

    pot = gp.CCompositePotential()
    pot['halo']       = gp.NFWPotential(
                            m=p['m_Triaxial_halo'], r_s=p['r_Triaxial_halo'],
                            a=1, b=1, c=p['q2_Triaxial_halo'], units=galactic)
    pot['thin_disk']  = gp.MN3ExponentialDiskPotential(
                            m=4*np.pi*p['rho_thin_disk']*p['hr_thin_disk']**2*p['hz_thin_disk'],
                            h_R=p['hr_thin_disk'], h_z=p['hz_thin_disk'],
                            units=galactic, positive_density=True)
    pot['thick_disk'] = gp.MN3ExponentialDiskPotential(
                            m=4*np.pi*p['rho_thick_disk']*p['hr_thick_disk']**2*p['hz_thick_disk'],
                            h_R=p['hr_thick_disk'], h_z=p['hz_thick_disk'],
                            units=galactic, positive_density=True)
    pot['bulge']      = gp.PowerLawCutoffPotential(
                            m=p['m_bulge'], r_c=p['r_bulge'],
                            alpha=p['alpha_bulge'], units=galactic)

    v_circ = pot.circular_velocity(R=r_array, z=np.zeros_like(r_array))
    v_circ_kms = v_circ.to(u.km/u.s).value
    all_vcirc[i] = v_circ_kms

    # Accept if every point lies within 3σ of the observed curve
    accepted[i] = np.all(np.abs(v_circ_kms - obs_Vc) <= 3 * obs_sVc)

n_accepted = accepted.sum()
print(f"Accepted {n_accepted} / {N_sample}  ({100*n_accepted/N_sample:.2f} %)")

# ── Save rotation curves + acceptance mask ────────────────────────────────────
np.savez('./data/plots/gala_rotcurv/rotation_curves.npz',
         r_kpc=obs_R, vcirc_kms=all_vcirc, accepted=accepted,
         obs_Vc=obs_Vc, obs_sVc=obs_sVc)
print("Saved rotation curves → ./data/plots/gala_rotcurv/rotation_curves.npz")

# ── Rotation-curve plot ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
colors_all = plt.cm.Greys(np.linspace(0.3, 0.6, N_sample))
colors_acc = plt.cm.viridis(np.linspace(0, 1, n_accepted))

for i in range(N_sample):
    if not accepted[i]:
        ax.plot(obs_R, all_vcirc[i], color='grey', alpha=0.02, lw=0.4)

for j, i in enumerate(np.where(accepted)[0]):
    ax.plot(obs_R, all_vcirc[i], color=colors_acc[j], alpha=0.15, lw=0.6)

ax.errorbar(obs_R, obs_Vc, yerr=3*obs_sVc, fmt='o', color='red',
            ms=3, lw=1, capsize=2, label='Observed ±3σ', zorder=5)
ax.set_xlabel('Radius (kpc)')
ax.set_ylabel('Circular Velocity (km/s)')
ax.legend()
fig.savefig('./data/plots/gala_rotcurv/rotation_curves.png', dpi=300)
plt.close(fig)
print("Saved rotation curve plot → ./data/plots/gala_rotcurv/rotation_curves.png")

# ── Corner plot: prior vs accepted ───────────────────────────────────────────
param_labels = [r'$m_\mathrm{halo}$', r'$r_\mathrm{halo}$', r'$q_2$',
                r'$\rho_\mathrm{thin}$', r'$h_{R,\mathrm{thin}}$', r'$h_{z,\mathrm{thin}}$',
                r'$\rho_\mathrm{thick}$', r'$h_{R,\mathrm{thick}}$', r'$h_{z,\mathrm{thick}}$',
                r'$m_\mathrm{bulge}$', r'$r_\mathrm{bulge}$', r'$\alpha_\mathrm{bulge}$']

# Build (N_sample, n_params) array
prior_samples    = np.column_stack([parameters_dict[k][:, 0] for k in params])
accepted_samples = prior_samples[accepted]

fig = corner.corner(
    prior_samples,
    labels=param_labels,
    color='steelblue',
    hist_kwargs={'density': True, 'alpha': 0.5},
    plot_datapoints=False,
    plot_contours=False,
    fill_contours=False,
)
corner.corner(
    accepted_samples,
    labels=param_labels,
    color='darkorange',
    hist_kwargs={'density': True, 'alpha': 0.8},
    plot_datapoints=False,
    plot_contours=True,
    fill_contours=True,
    fig=fig,
)

# Manual legend
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color='steelblue', alpha=0.5, label='Prior (all)'),
                    Patch(color='darkorange', alpha=0.8, label=f'Accepted (3σ, n={n_accepted})')],
           loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)

fig.savefig('./data/plots/gala_rotcurv/corner_prior_vs_accepted.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
print("Saved corner plot → ./data/plots/gala_rotcurv/corner_prior_vs_accepted.png")