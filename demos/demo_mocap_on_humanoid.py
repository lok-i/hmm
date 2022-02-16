
from gym_hmm_ec.envs.bipedal_env import BipedEnv 
import c3d
import numpy as np
from tqdm import tqdm
import yaml
from mujoco_py import functions

from dm_control import mujoco
# from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from gym_hmm_ec.envs.utils.ik_solver import qpos_from_site_pose



assets_path = './gym_hmm_ec/envs/assets/'
marker_conf_file_name = 'marker_config.yaml'
c3d_file_name = 'mocap_data/Trial_1.c3d'

config_file = open(assets_path+"our_data/"+ marker_conf_file_name,'r+')
marker_conf = yaml.load(config_file, Loader=yaml.FullLoader)

link_marker_info = marker_conf['index2marker_info']
marker_name_in_order = [0]*40
for link_id in link_marker_info:
    marker_name_in_order[link_id] = link_marker_info[link_id]['marker_name']
'''
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
print("Marker Pos. Traj. Shape:", marker_positions.shape)

_ = 0
for f_id in np.linspace(start=0,stop=marker_positions.shape[0],num=20):
    print(f_id)
    np.save(assets_path+"our_data/frame_rand"+str(_)+".npy",marker_positions[int(f_id)])
    _+=1


'''
# environment config and setup
env_conf = {
            'set_on_rack': False,
            'render': True,
            'model_name':'humanoid_no_hands_mocap',
            'mocap':False
            }

env = BipedEnv(**env_conf)
env.model.opt.gravity[2] = 0

# initialse the env,reset simualtion
env.reset()



physics = mujoco.Physics.from_xml_path(assets_path+"models/"+env.env_params['model_name']+".xml")


link_wise_markers = [   
                        # ['C7','LSHO','RSHO','CLAV'],
                        ['RASI','LASI','RPSI','LPSI'],

                        [ "RKNL", "RKNM"],

                        ["LKNL","LKNM"],


                        # [ "RANL","RANM"],

                        # [ "LANL","LANM"],

                    
                    ]

relavent_doFs = [
                    ['root'],
                    ['right_hip_x','right_hip_y','right_hip_z'],
                    ['left_hip_x','left_hip_y','left_hip_z'],

                    # ['right_knee'],
                    # ['left_knee'],

                ]





# keep the similation in pause until activated manually
if env.env_params['render']:
    env.viewer._paused = True
env.viewer.cam.distance = 3
cam_pos = [0.0, 1.0, 0.75]

for i in range(3):        
    env.viewer.cam.lookat[i]= cam_pos[i] 
env.viewer.cam.elevation = -15
env.viewer.cam.azimuth = 180



for n_frame in range(19):
    print("Frame:",n_frame)
    frame = np.load(assets_path+"our_data/frame_rand"+str(n_frame)+".npy")

    target_qpos = np.zeros(len(env.sim.data.qpos))
    for step in range(500):
        # for i in range(frame.shape[0]):
        for tm in link_wise_markers:
            for target_site in tm:
            
                marker_name = 'm'+str(marker_name_in_order.index(target_site))
                env.sim.data.set_mocap_pos(marker_name, frame[marker_name_in_order.index(target_site),:] )

        # syncornize the mujoco_py stata with ik (dm_control's) stat
        for i in range(len(env.sim.data.qpos)):
            physics.data.qpos[i] = env.sim.data.qpos[i] 

        # solve for pelvis
        for target_i,target_site in enumerate(link_wise_markers[0]):
            ik_result = qpos_from_site_pose(physics,
                                            site_name=target_site,
                                            target_pos=frame[ marker_name_in_order.index(target_site)  ,:],
                                            # joint_names = relavent_doFs[0],
                                            max_steps=100,
                                            inplace=True
                                            )
            for i in range(7):
                
                if target_i != 0: 
                    alpha = target_i / (target_i +1)
                    target_qpos[i] = alpha*target_qpos[i] + physics.data.qpos[i] / (target_i +1)
                else:
                    target_qpos[i] = physics.data.qpos[i].copy()
                
                # target_qpos[i] = physics.data.qpos[i]
            
        
        # solve for r hip
        for target_j,target_site in enumerate(link_wise_markers[1]):
            ik_result = qpos_from_site_pose(physics,
                                            site_name=target_site,
                                            target_pos=frame[ marker_name_in_order.index(target_site)  ,:],
                                            joint_names = relavent_doFs[1],
                                            max_steps=100,
                                            inplace=True

                                            )
            for j in range(len(relavent_doFs[1])):
                
                if target_j != 0: 
                    alpha = target_j / (target_j +1)
                    target_qpos[10+j] = alpha*target_qpos[10+j] + physics.data.qpos[10+j] / (target_j +1)
                else:
                    target_qpos[10+j] = physics.data.qpos[10+j].copy()
                
                # target_qpos[10+j] = physics.data.qpos[10+j]
        
        # solve for l hip
        for target_k,target_site in enumerate(link_wise_markers[2]):
            ik_result = qpos_from_site_pose(physics,
                                            site_name=target_site,
                                            target_pos=frame[ marker_name_in_order.index(target_site)  ,:],
                                            joint_names = relavent_doFs[2],
                                            max_steps=100,
                                            inplace=True

                                            )
            for k in range(len(relavent_doFs[2])):
                
                if target_k != 0: 
                    alpha = target_k / (target_k +1)
                    target_qpos[10+6+k] = alpha*target_qpos[10+6+k] + physics.data.qpos[10+6+k] / (target_k +1)
                else:
                    target_qpos[10+6+k] = physics.data.qpos[10+6+k].copy()
                
                # target_qpos[10+6+k] = physics.data.qpos[10+6+k]


        

        for i in range(len(env.sim.data.qpos)):
            env.sim.data.qpos[i] = target_qpos[i]#physics.data.qpos[i]

        obs,reward,done,info = env.step(action = np.zeros(shape=env.n_act_joints))



