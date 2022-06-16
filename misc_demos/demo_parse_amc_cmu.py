from utils import parse_amc 
from mujoco_py import functions
from gym_hmm_ec.envs.bipedal_env import BipedEnv 
from gym_hmm_ec.controllers.pd_controller import PDController 
import matplotlib.pyplot as plt
import numpy as np
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
            'render':True,
            'model_name': 'humanoid_CMU',
            'mocap':False,
            'observations':
            {
                'current_model_state': None
            },
            'actions':
            {   'joint_torques':
                    {
                        'dim': 15,
                        'torque_max': 5
                    }                
            },
            'rewards':
            {
                'zero_reward':None
            },
            'terminations':
            {
                'indefinite':None
            }                
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
        env.sim.step()
        env.viewer.render()
        


 
env.close()



