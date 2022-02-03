
from gym_hmm_ec.envs.bipedal_env import BipedEnv 
from gym_hmm_ec.controllers.pd_controller import PDController 
import matplotlib.pyplot as plt
import numpy as np
from mujoco_py import functions

from dm_control import mujoco
from dm_control.utils.inverse_kinematics import qpos_from_site_pose

# environment config and setup
env_conf = {
            'set_on_rack': False,
            'render': True,
            'model_name':'humanoid_no_hands_mocap',
            'mocap':False
            }

env = BipedEnv(**env_conf)

# initialse the env,reset simualtion
env.reset()

# keep the similation in pause until activated manually
if env.env_params['render']:
    env.viewer._paused = True

assets_path = './gym_hmm_ec/envs/assets/'
physics = mujoco.Physics.from_xml_path(assets_path+"models/"+env.env_params['model_name']+".xml")
env.model.opt.gravity[2] = 0

# for _ in range(1):
while True:
    ik_result = qpos_from_site_pose(physics,
                                    site_name='right_foot',
                                    target_pos=[0.1,-0.5,0.5],
                                    joint_names = ['right_hip_x','right_hip_z','right_hip_y','right_knee','right_ankle_y','right_ankle_x']
                                    
                                    )
    print(ik_result)
    ik_soln =  ik_result.qpos
    for i in range(len(env.sim.data.qpos)):
        env.sim.data.qpos[i] = ik_soln[i]

    obs,reward,done,info = env.step(action = np.zeros(shape=env.n_act_joints))
    for i in range(len(env.sim.data.qpos)):
        physics.data.qpos[i] = env.sim.data.qpos[i] 

env.close()