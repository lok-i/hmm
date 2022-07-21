
from cv2 import threshold
from gym_hmm_ec.envs.bipedal_env import BipedEnv 
from gym_hmm_ec.controllers.pd_controller import PDController 
import matplotlib.pyplot as plt
from utils.misc_functions import quat2euler
import numpy as np
import yaml
import gym


env_conf = {
            'set_on_rack': False,
            'render': True,
            'model_name': 'AB3_Session1_pm_mll_2D',
            'mocap':False,
            'observations':
            {
                'current_model_state': None
            },
            'actions':
            {   'joint_torques':
                    {
                        'dim': 15,
                        'torque_max': 5
                    }                
            },
            'initalisation':
            {
                'step_position_2D': 0.001,      
            },
            'rewards':
            {
                'energy':
                { 'alpha_1': 0.04 }
            },
            'terminations':
            {
                'indefinite':None
            }                
            }

env = BipedEnv(**env_conf)

# initialse the env,reset simualtion
env.reset()

# keep the similation in pause until activated manually
if env.env_params['render']:
    env.viewer._paused = True
    env.viewer.cam.distance = 3
    cam_pos = [0.0, 0.0, 0.75]

    for i in range(3):        
        env.viewer.cam.lookat[i]= cam_pos[i] 
    env.viewer.cam.elevation = -10
    env.viewer.cam.azimuth = 90

    # env.viewer.cam.azimuth = 180


# print(env.model.actuator_names,'\n',env.model.joint_names)

# for j_n in env.model.joint_names:
#     print(j_n,env.model.joint_name2id(j_n))

# for a_n in env.model.actuator_names:
#     print(a_n,env.model.actuator_name2id(a_n))

# print(env.model.opt.enableflags)
# exit()
for i in range(env.model.nbody):
    
    mass = env.model.body_mass[i]
    if mass != 0:
        print( i,env.model.body_id2name(i),mass )

# # print(functions.mj_getTotalmass(env.model) )
print("Total Mass:",sum(env.model.body_mass) )

env.sim.data.qvel[0] = 1

while True:

    control_actions = np.array([0,0,0,0,0,0])  
    obs,reward,done,info = env.step(action = control_actions )

    # print("act:",control_actions)
    # print("obs:",obs)
    # print("rew:",reward)
    # print("done:",done)
    # print('Applied: ',env.sim.data.qfrc_applied[:])
    # print('Actuator: ',env.sim.data.qfrc_actuator[:].shape)
    # print('Control: ',env.sim.data.ctrl[:])
    # print('Applied: ',env.sim.data.qfrc_applied[:])
    qpos = env.sim.data.qpos[:].copy()
    qvel = env.sim.data.qvel[:].copy()

    # print('Vel: ',qpos.shape, 'Actuator: ',env.sim.data.qfrc_actuator[:].shape)
    # x,y,z = euler_from_quaternion(qpos[3],qpos[4],qpos[5],qpos[6])
    # print(x*114.64,y*114.64,z*114.64)
    ll = env.sim.data.body_xpos[env.model.body_name2id("left_leg/foot")]
    rl = env.sim.data.body_xpos[env.model.body_name2id("right_leg/foot")]
    # print(quat2euler([qpos[3],qpos[4],qpos[5],qpos[6]])*180/np.pi)
    print(qpos)

    if done:
        break

env.close()

'''
#to load arbitary xml and view

import mujoco_py

fullpath = 'gym_hmm_ec/envs/assets/models/default_red_model.xml'
model = mujoco_py.load_model_from_path(fullpath)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)


sim.reset()
viewer._paused = True
while True:
    sim.step()
    viewer.render()

'''