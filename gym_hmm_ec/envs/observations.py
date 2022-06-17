import numpy as np

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


