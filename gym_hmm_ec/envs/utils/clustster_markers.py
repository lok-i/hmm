
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
assets_path = './gym_hmm_ec/envs/assets/'

frame = np.zeros(1)


all_frames=[]
for n_frame in range(19):
    print("Frame:",n_frame)
    # prev_frame=frame.copy()
    frame = np.load(assets_path+"our_data/mocap_data/frame_rand"+str(n_frame)+".npy")
    all_frames.append(frame.tolist())

all_frames = np.array(all_frames) 
distance_matrix = np.zeros((all_frames[0].shape[0],all_frames[0].shape[0]))
distance_diff_matrix = np.zeros((all_frames[0].shape[0],all_frames[0].shape[0]))

print(all_frames.shape)   

k = 0
distance_sum = 0.
prev_sum = 0
for i in range(distance_matrix.shape[0]):
    for j in range(distance_matrix.shape[1]):
        
        # distance = all_frames[k+1][i] - all_frames[k+1][i]   
        # print(i,j)
        distance_sum = 0.
        for k in range(all_frames.shape[0]):
            distance_sum += np.linalg.norm(all_frames[k,i]-all_frames[k,j])
        distance_sum = distance_sum / all_frames.shape[0]
        distance_matrix[i,j] = distance_sum
        distance_diff_matrix[i,j] = distance_sum - prev_sum
        prev_sum = distance_sum.copy()

        # distance_matrix[i][j] = 

# plt.imshow(distance_matrix)
fig, axs = plt.subplots(1,2)
axs[0].set_title("distance_diff_matrix")
im = axs[0].imshow(distance_diff_matrix)
axs[0].set_xlabel("Markers")
axs[0].set_ylabel("Markers")


divider = make_axes_locatable(axs[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im,cax=cax)

axs[1].set_title("distance_matrix")

im = axs[1].imshow(distance_matrix)
axs[1].set_xlabel("Markers")
axs[1].set_ylabel("Markers")

divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im,cax=cax)

fig.suptitle('Distance Corelation')

plt.show()
