from gym_hmm_ec.envs.bipedal_env import BipedEnv 
from mujoco_py import functions
from gym_hmm_ec.envs.utils.parse_amc import euler2quat 
from gym_hmm_ec.controllers.pd_controller import PDController 
import matplotlib.pyplot as plt
import numpy as np
from mujoco_py import functions

# environment config and setup
env_conf = {
            'set_on_rack': False,
            'render': True,
            'model_name':'walker2D_mocap',
            'mocap':True
            }

env = BipedEnv(**env_conf)


env.reset()

# keep the similation in pause until activated manually
if env.env_params['render']:
    env.viewer._paused = True

for _ in range(5000):
    
    for body_name in env.model.body_names:
        if 'mocap_' in body_name:
        #     env.sim.data.set_mocap_pos(body_name, env.sim.data.get_mocap_pos(body_name) - np.array([0.001,0.,0.]) )

            euler_in_deg = np.zeros(3)
            if 'right_thigh' in body_name:
                euler_in_deg[1] = 25*np.sin(-0.01*_)

            if 'right_leg' in body_name:
                euler_in_deg[1] = -25*np.sin(-0.01*_)

            if 'leftt_thigh' in body_name:
                euler_in_deg[1] = -25*np.sin(-0.01*_)

            if 'left_leg' in body_name:
                euler_in_deg[1] = 25*np.sin(-0.01*_)

            quat_from_euler = euler2quat(euler_in_deg[0],euler_in_deg[1],euler_in_deg[2])
            env.sim.data.set_mocap_quat(body_name, quat_from_euler )

    obs,reward,done,info = env.step(action = np.zeros(shape=env.n_act_joints))
    functions.mj_inverse(env.model,env.sim.data)
    
    print('IK:',env.sim.data.qpos[:])
    print('ID:',env.sim.data.qfrc_inverse)
    

env.close()