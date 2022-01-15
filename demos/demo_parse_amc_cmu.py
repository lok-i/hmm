from gym_hmm_ec.envs.utils import parse_amc 
from mujoco_py import functions
from gym_hmm_ec.envs.bipedal_env import BipedEnv 
from gym_hmm_ec.controllers.pd_controller import PDController 
import matplotlib.pyplot as plt
import numpy as np
import time
from mujoco_py import functions

# index to joint relation of the humanoid CMU model 
index2joint = {
               0: 'root', 1: 'root', 2: 'root', 3: 'root', 4: 'root', 5: 'root', 6: 'root', 7: 'lfemurrz', 
               8: 'lfemurry', 9: 'lfemurrx', 10: 'ltibiarx', 11: 'lfootrz', 12: 'lfootrx', 13: 'ltoesrx', 
               14: 'rfemurrz', 15: 'rfemurry', 16: 'rfemurrx', 17: 'rtibiarx', 18: 'rfootrz', 19: 'rfootrx', 
               20: 'rtoesrx', 21: 'lowerbackrz', 22: 'lowerbackry', 23: 'lowerbackrx', 24: 'upperbackrz', 
               25: 'upperbackry', 26: 'upperbackrx', 27: 'thoraxrz', 28: 'thoraxry', 29: 'thoraxrx', 
               30: 'lowerneckrz', 31: 'lowerneckry', 32: 'lowerneckrx', 33: 'upperneckrz', 34: 'upperneckry', 
               35: 'upperneckrx', 36: 'headrz', 37: 'headry', 38: 'headrx', 39: 'lclaviclerz', 40: 'lclaviclery', 
               41: 'lhumerusrz', 42: 'lhumerusry', 43: 'lhumerusrx', 44: 'lradiusrx', 45: 'lwristry', 46: 'lhandrz',
               47: 'lhandrx', 48: 'lfingersrx', 49: 'lthumbrz', 50: 'lthumbrx', 51: 'rclaviclerz', 52: 'rclaviclery', 
               53: 'rhumerusrz', 54: 'rhumerusry', 55: 'rhumerusrx', 56: 'rradiusrx', 57: 'rwristry', 58: 'rhandrz', 
               59: 'rhandrx', 60: 'rfingersrx', 61: 'rthumbrz', 62: 'rthumbrx'
               }

traj_delta_t = 0.002
# path to the amc file
taskname = 'run'
filename = './gym_hmm_ec/envs/assets/cmu_mocap/'+ taskname +'.amc'
converted = parse_amc.convert(
                                filename,
                                index2joint, 
                                traj_delta_t
                            )
print('Trajctory Lengths:',converted.qpos.shape,converted.qvel.shape)

env_conf = {
            'set_on_rack': False,
            'render': True,
            'model_name':'humanoid_CMU',
            'mocap':False
            }

env = BipedEnv(**env_conf)


# keep the similation in pause until activated manually
if env.env_params['render']:
    env.viewer._paused = True

joints_of_intrest = [ 
                      'lfemurrz','lfemurry', 'lfemurrx', 'ltibiarx', 'lfootrz', 'lfootrx', 'ltoesrx', 
                    #   'rfemurrz','rfemurry', 'rfemurrx', 'rtibiarx', 'rfootrz', 'rfootrx', 'rtoesrx'
                    ]

 
torques_of_joints = [] # without contact
torques_of_joints_contact = [] # with contact
env.reset()
for n_epi in range(1):
    env.reset()
    for _ in range(converted.qvel.shape[1]):
        

        env.sim.data.qpos[:] =  converted.qpos[:,_]
        if n_epi == 1:
            env.sim.data.qpos[2] -= 0.1
        env.sim.data.qvel[:] =  converted.qvel[:,_]
        
        obs,reward,done,info = env.step(action = np.zeros(shape=env.n_act_joints))
        
        functions.mj_inverse(env.model,env.sim.data)
        torques = []
        for joint_name in joints_of_intrest:
            
            for j_id, j_name in index2joint.items():
                if j_name == joint_name:
                    joint_id = j_id
            torques.append( env.sim.data.qfrc_inverse[joint_id])
        if n_epi == 0:
            torques_of_joints.append(torques)
        else:
            torques_of_joints_contact.append(torques)

 
env.close()



torques_of_joints = np.array(torques_of_joints)
torques_of_joints_contact = np.array(torques_of_joints_contact)
print(torques_of_joints.shape)
nrows = 2
ncols = 4
fig,axs = plt.subplots(nrows,ncols)
fig.set_size_inches(18.5, 10.5)
timesteps = traj_delta_t * np.arange(torques_of_joints.shape[0])
for plot_id,joint_name in enumerate(joints_of_intrest):

    row = plot_id // ncols
    col = plot_id % ncols
    axs[row,col].plot(timesteps, torques_of_joints[:,plot_id],label=joint_name+' (no contact)')
    # axs[row,col].plot(timesteps, torques_of_joints_contact[:,plot_id],label=joint_name)

    axs[row,col].set_ylabel("torques (Nm)")
    axs[row,col].set_xlabel("time (s)")
    # axs[row,col].set_ylim([-20, 20])
    axs[row,col].legend(loc='upper right')
    axs[row,col].grid()

fig.suptitle('ID for input type 2 (CMU Dataset as .amc) of Task: '+taskname )
fig.tight_layout()
plt.savefig('./evaluation_plots/ip_type2_'+taskname+'.jpg')
plt.show()






