
from gym_hmm_ec.envs.bipedal_env import BipedEnv 
from gym_hmm_ec.controllers.pd_controller import PDController 
import matplotlib.pyplot as plt
import numpy as np
from mujoco_py.generated import const

# TODO: Fix the mocap integration of env with the generated model
# environment config and setup
env_conf = {
            'set_on_rack': False,
            'render': True,
            'model_name': 'default_humanoid_mocap_generated',
            'mocap':False # problem when true
            }

env = BipedEnv(**env_conf)

# initialse the env,reset simualtion
env.reset()

# keep the similation in pause until activated manually
if env.env_params['render']:
    env.viewer._paused = True


while True:

    control_actions = np.zeros(shape=env.n_act_joints)
    obs,reward,done,info = env.step(action = control_actions )

env.close()

