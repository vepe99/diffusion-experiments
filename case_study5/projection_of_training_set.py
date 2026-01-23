from autocvd import autocvd
autocvd(num_gpus = 1)

import numpy as np
import jax.numpy as jnp
from jax import vmap


from observational_realism import galc_to_icrs_spherical


# data = np.load("./case_study5/training_set_odisseo_triaxial.npz")
data = np.load("./case_study5/test_set_multistream_odisseo_triaxial.npz")

sim_data = data["sim_data"]  # shape (N, 6) with columns [x, y, z, vx, vy, vz]
print(sim_data.shape)
# icrs_spherical = vmap(galc_to_icrs_spherical)(jnp.array(sim_data))
icrs_spherical = vmap(vmap(galc_to_icrs_spherical))(jnp.array(sim_data))
mean_sim_data = np.mean(sim_data, axis=(0, 1))  
std_sim_data = np.std(sim_data, axis=(0, 1))  

# np.savez('./case_study5/projection_training_set_odisseo_triaxial.npz',
#     j=data['j'],
#     sim_data=icrs_spherical,
#     mean_sim_data=mean_sim_data,
#     std_sim_data=std_sim_data,
#     m_nfw=data['m_nfw'],
#     mean_m_nfw=data['mean_m_nfw'],
#     std_m_nfw=data['std_m_nfw'],
#     r_s=data['r_s'],
#     mean_r_s=data['mean_r_s'],
#     std_r_s=data['std_r_s'],
#     q1=data['q1'],
#     mean_q1=data['mean_q1'],
#     std_q1=data['std_q1'],
#     q2=data['q2'],
#     mean_q2=data['mean_q2'],
#     std_q2=data['std_q2'],
#     prog_mass=data['prog_mass'],
#     mean_prog_mass=data['mean_prog_mass'],
#     std_prog_mass=data['std_prog_mass'],
#     t_end=data['t_end'],
#     mean_t_end=data['mean_t_end'],
#     std_t_end=data['std_t_end'],
#     x_c=data['x_c'],
#     mean_x_c=data['mean_x_c'],
#     std_x_c=data['std_x_c'],
#     y_c=data['y_c'],
#     mean_y_c=data['mean_y_c'],
#     std_y_c=data['std_y_c'],
#     z_c=data['z_c'],
#     mean_z_c=data['mean_z_c'],
#     std_z_c=data['std_z_c'],
#     v_xc=data['v_xc'],
#     mean_v_xc=data['mean_v_xc'],
#     std_v_xc=data['std_v_xc'],
#     v_yc=data['v_yc'],
#     mean_v_yc=data['mean_v_yc'],
#     std_v_yc=data['std_v_yc'],
#     v_zc=data['v_zc'],
#     mean_v_zc=data['mean_v_zc'],
#     std_v_zc=data['std_v_zc'],
# )

np.savez('./case_study5/projection_test_set_multistream_odisseo_triaxial.npz',
         j=data['j'],
         sim_data=icrs_spherical,
        m_nfw=data['m_nfw'],
        r_s=data['r_s'],
        q1=data['q1'],
        q2=data['q2'],
        prog_mass=data['prog_mass'],
        t_end=data['t_end'],
        x_c=data['x_c'],
        y_c=data['y_c'],
        z_c=data['z_c'],
        v_xc=data['v_xc'],
        v_yc=data['v_yc'],
        v_zc=data['v_zc'],
)