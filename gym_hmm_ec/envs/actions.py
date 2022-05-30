import numpy as np

class action_base():

    def __init__(self,params) -> None:
        self.params = params
    
    def unnormalize_action(self):
        raise NotImplementedError
    def step(self):
        # from actions -> torques
        # output should always be the shape of self.sim.data.ctrl
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError

class joint_torques(action_base):

    def step(self,policy_output):
        actions = self.unnormalize_action(policy_output)

        return actions
    def unnormalize_action(self,norm_action):
        
        return self.params['torque_max']*norm_action
    
    def reset(self):
        pass

