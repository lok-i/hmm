from turtle import pos
import numpy as np
from utils.misc_functions import quat2euler

class observation_base():

    def __init__(self,params) -> None:
        self.params = params

    def step(self):
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError

class current_model_state(observation_base):

    def step(self,input_dict):
        qpos = input_dict['q']
        qvel = input_dict['dq']

        if isinstance(self.params, dict): 
            
            if '-bpx' in self.params['obs_type']:
                qpos[0] = 0.0
            if '-bpy' in self.params['obs_type']:
                qpos[1] = 0.0
            if '-bpz' in self.params['obs_type']:
                qpos[2] = 0.0

            if '-bpvx' in self.params['obs_type']:
                qvel[0] = 0.0
            if '-bpvy' in self.params['obs_type']:
                qvel[1] = 0.0
            if '-bpvz' in self.params['obs_type']:
                qvel[2] = 0.0

        return np.concatenate([qpos,qvel]).ravel()
    
    def reset(self):
        pass

class reduced_current_model_state(observation_base):

    def step(self,input_dict):
        qpos = input_dict['q']
        qvel = input_dict['dq']
        euler = quat2euler([qpos[3],qpos[4],qpos[5],qpos[6]])

        return np.concatenate([qpos[7:],qvel[6:],euler[:2]]).ravel()
    
    def reset(self):
        pass

class reduced_current_model_state_2D(observation_base):

    def step(self,input_dict):
        qpos = input_dict['q']
        qvel = input_dict['dq']

        return np.concatenate([qpos[1:3],qvel[:3]]).ravel()
    
    def reset(self):
        pass

class history_model_state(observation_base):

    def step(self,input_dict):
        qpos = input_dict['q']
        qvel = input_dict['dq']

        if isinstance(self.params, dict): 
            
            if '-bpx' in self.params['obs_type']:
                qpos[0] = 0.0
            if '-bpy' in self.params['obs_type']:
                qpos[1] = 0.0
            if '-bpz' in self.params['obs_type']:
                qpos[2] = 0.0

            if '-bpvx' in self.params['obs_type']:
                qvel[0] = 0.0
            if '-bpvy' in self.params['obs_type']:
                qvel[1] = 0.0
            if '-bpvz' in self.params['obs_type']:
                qvel[2] = 0.0

        self.history[0] = self.history[1]
        self.history[1] = self.history[2]
        self.history[2] = self.history[3]
        self.history[3] = self.history[4]
        self.history[4] = np.concatenate([qpos,qvel]).ravel()
        return self.history.ravel()
    
    def reset(self):
        self.history = np.zeros((5,self.params['obs_dim']))
        pass


