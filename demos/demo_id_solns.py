
from gym_hmm_ec.envs.bipedal_env import BipedEnv 
from gym_hmm_ec.controllers.pd_controller import PDController 
import matplotlib.pyplot as plt
import numpy as np
from utils import misc_functions

# TODO: Fix the mocap integration of env with the generated model
# environment config and setup
env_conf = {
            'set_on_rack': True,
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
processed_filepath = './data/our_data/marker_data/processed_data/AB1_Session1_Right6_Left6_from_1200_to_3200.npz'

ik_solns = np.load(ik_soln_filpath)['ik_solns']
id_solns = np.load(id_soln_filpath)['id_solns']


print(ik_solns.shape,id_solns.shape)

# env.sim.data.qpos[:] = ik_solns[0,:]
# env.sim.data.qvel[:] = np.zeros(21)


# while True:

print('sim dt:',env.model.opt.timestep)

frame_rate = 100.
timestep = env.model.opt.timestep
mocap_marker_data = np.load(processed_filepath)
grf_data = misc_functions.interpolate_data(data=mocap_marker_data['grfs'],old_dt= 1./frame_rate,new_dt=timestep)
ik_solns = misc_functions.interpolate_data(data=ik_solns,old_dt= 1./frame_rate,new_dt=timestep)


# exit()
control_actions = np.zeros(shape=env.n_act_joints)

for i in range( id_solns.shape[0]):

    # qpos = ik_solns[i]
    # env.sim.data.qpos[:] = qpos
    # if i == 0:
    #     prev_qpos = qpos.copy()
    #     prev_qvel = np.zeros(env.model.nv)

    # env.sim.data.qvel[:] = np.concatenate([
    #     (qpos[:3]-prev_qpos[:3]) / timestep,
    #     misc_functions.mj_quat2vel(
    #         misc_functions.mj_quatdiff(prev_qpos[3:7], qpos[3:7]), timestep),
    #     (qpos[7:]-prev_qpos[7:]) / timestep
    # ]
    # ).ravel()


    # body_name = 'right_leg/foot'
    # body_id = env.sim.model.body_name2id(body_name)
    # wrench_prtb = -1. * grf_data[i,0:6]

    # frc_pos = env.sim.data.get_body_xpos(body_name)
    # env.view_vector_arrows(
    #     vec=wrench_prtb[0:3],
    #     vec_point=frc_pos,
    #     vec_mag_max=500,
    # )
    # env.sim.data.xfrc_applied[body_id][:] = wrench_prtb

    # body_name = 'left_leg/foot'
    # body_id = env.sim.model.body_name2id(body_name)
    # wrench_prtb = -1. * grf_data[i,6:12]
    # frc_pos = env.sim.data.get_body_xpos(body_name)
    # env.view_vector_arrows(
    #     vec=wrench_prtb[0:3],
    #     vec_point=frc_pos,
    #     vec_mag_max=500,
    # )
    # env.sim.data.xfrc_applied[body_id][:] = wrench_prtb



    env.sim.data.qfrc_applied[6:] = 0.005*id_solns[i,6:]
    # env.sim.data.qfrc_actuator[6:] = 0.005*id_solns[i,6:]
    # control_actions = 0.0005*id_solns[i,6:] #np.zeros(shape=env.n_act_joints)
    # print(id_solns[i,:])
    
    
    obs,reward,done,info = env.step(action = control_actions )

env.close()

