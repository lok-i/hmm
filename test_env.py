
from gym_hmm_ec.envs.bipedal_env import BipedEnv 
from gym_hmm_ec.controllers.pd_controller import PDController 
import matplotlib.pyplot as plt
import numpy as np
from mujoco_py import functions

# environment config and setup
env_conf = {
            'set_on_rack': False,
            'render': True,
            'model_name':'humanoid_no_hands',
            'mocap':False
            }

env = BipedEnv(**env_conf)

# initialse the env,reset simualtion
env.reset()

# keep the similation in pause until activated manually
if env.env_params['render']:
    env.viewer._paused = True

for _ in range(2000):
    
    obs,reward,done,info = env.step(action = np.zeros(shape=env.n_act_joints))


env.close()