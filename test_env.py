
from gym_hmm_ec.envs.bipedal_env import BipedEnv 
from gym_hmm_ec.controllers.pd_controller import PDController 
import matplotlib.pyplot as plt
import numpy as np
import yaml
import gym


# TODO: Fix the mocap integration of env with the generated model
# # environment config and setup
# env_conf = {
#             'set_on_rack': False,
#             'render': True,
#             'model_name': 'hopper'#'AB1_Session1_upd',#'rand_1_updated',
#             }

config_file = open("./experiments/redu_exp/sub_exp/conf.yaml")
traning_config = yaml.load(config_file, Loader=yaml.FullLoader)
env_conf =  traning_config['env_kwargs'].copy()


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
    env.viewer.cam.elevation = -15
    env.viewer.cam.azimuth = 180

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


while True:

    control_actions = np.zeros(shape=env.action_dim)    
    obs,reward,done,info = env.step(action = control_actions )

    # print("act:",control_actions)
    # print("obs:",obs)
    # print("rew:",reward)
    # print("done:",done)

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