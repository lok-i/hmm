
from gym_hmm_ec.envs.bipedal_env import BipedEnv 
import c3d
import numpy as np
from tqdm import tqdm
import yaml
from mujoco_py import functions

from dm_control import mujoco
from gym_hmm_ec.envs.utils.ik_solver import qpos_from_site_pose


LOAD_COMPLETE_MOCAP = True

assets_path = './gym_hmm_ec/envs/assets/'
marker_conf_file_name = 'marker_config.yaml'
c3d_file_name = 'mocap_data/c3ds/Trial_1.c3d'

config_file = open(assets_path+"our_data/"+ marker_conf_file_name,'r+')
marker_conf = yaml.load(config_file, Loader=yaml.FullLoader)

link_marker_info = marker_conf['index2marker_info']
marker_name_in_order = [0]*40
for link_id in link_marker_info:
    marker_name_in_order[link_id] = link_marker_info[link_id]['marker_name']


if LOAD_COMPLETE_MOCAP:
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


# environment config and setup
env_conf = {
            'set_on_rack': False,
            'render': True,
            'model_name':'humanoid_no_hands_mocap_generated',
            'mocap':False
            }

env = BipedEnv(**env_conf)
env.model.opt.gravity[2] = 0

# initialse the env,reset simualtion
env.reset()

physics = mujoco.Physics.from_xml_path(assets_path+"models/"+env.env_params['model_name']+".xml")


marker_names = ['RASI','LASI','RPSI','LPSI',
                "right_leg/RKNL", "right_leg/RKNM","right_leg/RANL","right_leg/RANM","right_leg/RHEE","right_leg/RM1","right_leg/RM5",
                "left_leg/LKNL", "left_leg/LKNM","left_leg/LANL","left_leg/LANM","left_leg/LHEE","left_leg/LM1","left_leg/LM5"]
relavent_doFs = ['root',
                 'right_leg/hip_x','right_leg/hip_y','right_leg/hip_z','right_leg/knee','right_leg/ankle_y','right_leg/ankle_x',
                 'left_leg/hip_x','left_leg/hip_y','left_leg/hip_z','left_leg/knee','left_leg/ankle_y','left_leg/ankle_x',]




# keep the similation in pause until activated manually
if env.env_params['render']:
    env.viewer._paused = True
env.viewer.cam.distance = 3
cam_pos = [0.0, 1.0, 0.75]

for i in range(3):        
    env.viewer.cam.lookat[i]= cam_pos[i] 
env.viewer.cam.elevation = -15
env.viewer.cam.azimuth = 180


if not LOAD_COMPLETE_MOCAP:
    marker_positions = []

    for n_frame in range(19):
        print("Frame:",n_frame)
        frame = np.load(assets_path+"our_data/mocap_data/test_frames/frame_rand"+str(n_frame)+".npy")
        marker_positions.append(frame)

else:
    marker_positions = marker_positions[1400:,:,:]



for frame in marker_positions:

    target_qpos = np.zeros(len(env.sim.data.qpos))

    for target_site in  marker_names:
        temp_target_site = target_site 
        if 'right_leg'in target_site:
            temp_target_site = target_site.replace('right_leg/','')
        if 'left_leg'in target_site:
            temp_target_site = target_site.replace('left_leg/','')             
        marker_name = 'm'+str(marker_name_in_order.index(temp_target_site))
        env.sim.data.set_mocap_pos(marker_name, frame[marker_name_in_order.index(temp_target_site),:] )

    # syncornize the mujoco_py stata with ik (dm_control's) stat
    for i in range(len(env.sim.data.qpos)):
        physics.data.qpos[i] = env.sim.data.qpos[i] 
        target_qpos[i] = env.sim.data.qpos[i] 
    
    # solve for all
    target_posses = []
    for target_site in marker_names:
        temp_target_site = target_site 

        if 'right_leg'in target_site:
            temp_target_site = target_site.replace('right_leg/','')
        if 'left_leg'in target_site:
            temp_target_site = target_site.replace('left_leg/','')  

        target_posses.append( frame[ marker_name_in_order.index(temp_target_site)  ,:].tolist() )
    target_posses = np.array(target_posses)
    ik_result = qpos_from_site_pose(physics,
                                    site_names= marker_names,
                                    target_pos=target_posses,
                                    joint_names = relavent_doFs,
                                    max_steps=1000,
                                    inplace=False
                                    )

            
    target_qpos = ik_result.qpos.copy() 
    

    if LOAD_COMPLETE_MOCAP:
        n_steps_per_pose = 1
    else:
        n_steps_per_pose = 500
    
    
    for i in range(n_steps_per_pose):
        for i in range(len(env.sim.data.qpos)):
            env.sim.data.qpos[i] = target_qpos[i]#physics.data.qpos[i]

        obs,reward,done,info = env.step(action = np.zeros(shape=env.n_act_joints))

    #np.save(assets_path+"our_data/mocap_data/ik_solns_test_frames/ik_soln_rand_"+str(n_frame)+".npy",env.sim.data.qpos)

