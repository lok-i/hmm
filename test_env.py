
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
            'model_name': 'rand_1_updated',
            'mocap':False # problem when true
            }

env = BipedEnv(**env_conf)

# initialse the env,reset simualtion
env.reset()

# keep the similation in pause until activated manually
if env.env_params['render']:
    env.viewer._paused = True


print(env.model.actuator_names,'\n',
env.model.joint_names)

for j_n in env.model.joint_names:
    print(j_n,env.model.joint_name2id(j_n))

for a_n in env.model.actuator_names:
    print(a_n,env.model.actuator_name2id(a_n))



ik_soln_filpath =  './data/our_data/ik_solns/AB1_Session1_Right6_Left6_from_1200_to_3200.npz'
id_soln_filpath =  './data/our_data/id_solns/AB1_Session1_Right6_Left6_from_1200_to_3200.npz'

ik_solns = np.load(ik_soln_filpath)['ik_solns']
id_solns = np.load(id_soln_filpath)['id_solns']


print(ik_solns.shape,id_solns.shape)

env.sim.data.qpos[:] = ik_solns[0,:]
env.sim.data.qvel[:] = np.zeros(21)

#exit()
# print(env.sim.data.qfrc_applied.shape)
# exit()
# while True:
print(env.model.opt.timestep)
env.model.opt.timestep = 0.01 # 1 / 100.
print(env.model.opt.timestep)

# exit()
for i in range( id_solns.shape[0]):
    control_actions = np.zeros(shape=env.n_act_joints)
    # print(id_solns[i,:])
    env.sim.data.qfrc_applied[:] = id_solns[i,:]
    obs,reward,done,info = env.step(action = control_actions )

env.close()

