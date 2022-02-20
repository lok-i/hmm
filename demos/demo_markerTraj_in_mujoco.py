import c3d
import numpy as np
from tqdm import tqdm
import mujoco_py

VIEW_MARKER_BASED_DATA = True

assets_path = './gym_hmm_ec/envs/assets/'
model_name = 'marker_set'
marker_conf_file_name = 'marker_config.yaml'



if VIEW_MARKER_BASED_DATA:
    c3d_file_name = 'mocap_data/Trial_1.c3d'
    # load data from c3d of  mocap data of 40 marker set
    marker_positions = []
    
    with open( assets_path+"our_data/"+c3d_file_name , 'rb') as handle:
        manager = c3d.Manager(handle)
        # print(manager.last_frame )
        reader = c3d.Reader(handle)

        pbar = tqdm(total=reader.last_frame) 
        pbar.set_description("Loading mocap data")   
        for data in reader.read_frames():
            pbar.update(1)
            # print('Frame {}'.format(data[0],data[1].shape,data[2][0]))
            all_marker = []
            
            for pt in data[1]:
                all_marker.append(pt[0:3].tolist())
        
            marker_positions.append(all_marker)

    marker_positions = 0.001*np.array(marker_positions)

else:
    point_trajfile = '3d_pose_coordinates_2.npy'
    marker_positions = np.load('./gym_hmm_ec/envs/assets/our_data/pose_esti/trial_1/'+point_trajfile)

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
    for i in range(frame.shape[0]):
        marker_name = 'm'+str(i)
        sim.data.set_mocap_pos(marker_name, frame[i,:] )

    sim.step()
    viewer.render()












