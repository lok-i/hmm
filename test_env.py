from gym_hmm_ec.envs.bipedal_env import BipedEnv 
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

for _ in range(2000):
    
    for body_name in env.model.body_names:
        if 'mocap_' in body_name:
            env.sim.data.set_mocap_pos(body_name, env.sim.data.get_mocap_pos(body_name) - np.array([0.001,0.,0.]) )
        #   env.sim.data.set_mocap_quat(body_name, np.array([0.7071068, 0, 0.7071068, 0,]) )

    # print(env.model.body_mocapid)
    # env.sim.data.mocap_pos[0] = env.sim.data.mocap_pos[0] + np.array([0.001,0.,0.])
    obs,reward,done,info = env.step(action = np.zeros(shape=env.n_act_joints))
    # print(env.sim.data.body_xpos)
    # exit()

    

env.close()