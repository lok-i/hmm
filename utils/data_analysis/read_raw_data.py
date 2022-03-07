import numpy as np
import matplotlib.pyplot as plt

assets_path = './gym_hmm_ec/envs/assets/'

npz_filepath = assets_path + 'our_data/marker_data/processed_data/AB1_Session1_Right6_Left6_from_1200_to_1500.npz'
mocap_data = np.load(npz_filepath)

cop_data = mocap_data['cops']


timesteps = np.arange(cop_data.shape[0])


nrows = 2
ncols = 3
fig,axs = plt.subplots(nrows,ncols)
fig.set_size_inches(18.5, 10.5)

k = 0
labels = ['lx','ly','lz','rx','ry','rz']
for row in range(nrows):
    for col in range(ncols):
        axs[row,col].plot(timesteps,cop_data[:,k])
        axs[row,col].set_title(labels[k])
        axs[row,col].grid()
        k+=1
plt.show()