import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 6))
data_path = '../data/streams/data_agama/'
color = ['red', 'blue', 'green']
for i in range(100):
    data = dict(np.load(data_path + f'simulation_{i}.npz'))
    sim = data['sim_data_projected']
    j = data['j']
    plt.scatter(sim[:, 0], sim[:, 1], s=1, c=color[j[0]])
plt.xlabel('ra')
plt.ylabel('dec')
# plt.legend()
plt.savefig('./stream_agama_sim.png')
print('Fig saved in ' + './stream_agama_sim.png')

