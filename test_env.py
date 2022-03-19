
from gym_hmm_ec.envs.bipedal_env import BipedEnv 
from gym_hmm_ec.controllers.pd_controller import PDController 
import matplotlib.pyplot as plt
import numpy as np
from mujoco_py import functions
from utils import misc_functions

# TODO: Fix the mocap integration of env with the generated model
# environment config and setup
env_conf = {
            'set_on_rack': False,
            'render': True,
            'model_name': 'ankle_yaw_updated',#'rand_1_updated',
            'mocap':False # problem when true
            }

env = BipedEnv(**env_conf)

# initialse the env,reset simualtion
env.reset()

# keep the similation in pause until activated manually
if env.env_params['render']:
    env.viewer._paused = True


# print(env.model.actuator_names,'\n',env.model.joint_names)

# for j_n in env.model.joint_names:
#     print(j_n,env.model.joint_name2id(j_n))

# for a_n in env.model.actuator_names:
#     print(a_n,env.model.actuator_name2id(a_n))


# for i in range(env.model.nbody):
    
#     mass = env.model.body_mass[i]
#     if mass != 0:
#         print( i,env.model.body_id2name(i),mass )

# # print(functions.mj_getTotalmass(env.model) )
# # print(sum(env.model.body_mass) )


while True:

    control_actions = np.zeros(shape=env.n_act_joints)    
    obs,reward,done,info = env.step(action = control_actions )

env.close()

