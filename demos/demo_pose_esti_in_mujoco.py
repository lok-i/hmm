import c3d
import numpy as np
from tqdm import tqdm
import mujoco_py


assets_path = './gym_hmm_ec/envs/assets/'
model_name = 'marker_set'
marker_conf_file_name = 'marker_config.yaml'
c3d_file_name = 'mocap_data/Trial_1.c3d'
#print(np.max(data[:,:,2]))

marker_positions = np.load('./gym_hmm_ec/envs/assets/' + 'our_data/pose_esti/trial_1/3d_pose_coordinates_1.npy')
print("Marker Pos. Traj. Shape:", marker_positions.shape)


# simulaion starts
model = mujoco_py.load_model_from_path(assets_path+"models/"+model_name+".xml")
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

sim.reset()

viewer._paused = True
viewer.cam.distance = 6
cam_pos = [0.0, 0.0, 0.0]

for i in range(3):        
    viewer.cam.lookat[i]= cam_pos[i] 
viewer.cam.elevation = -15
viewer.cam.azimuth = 220


# play simulation 
for frame in marker_positions:
    
    for i in range(40):#frame.shape[0]):
        marker_name = 'm'+str(i)
        sim.data.set_mocap_pos(marker_name, frame[i,:] )

    sim.step()
    viewer.render()












