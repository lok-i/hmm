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

        return np.concatenate([qpos,qvel]).ravel()
    def reset(self):
        pass

