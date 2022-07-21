from utils.misc_functions import quat2euler
import numpy as np

class termination_base():

    def __init__(self,params) -> None:
        self.params = params
    def step(self):
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError


class indefinite(termination_base):

    def step(self,input_dict):
        return False
    def reset(self):
        pass

class min_base_height(termination_base):

    def step(self,input_dict):
        if (not self.mn_flag) : 
            self.height = input_dict['q'][2]
            self.mn_flag = 1

        if input_dict['q'][2] < ((1-self.params['threshold']) * self.height): ## changed 3 -> 2
            return True
        else:
            return False
    
    def reset(self):
        self.mn_flag = 0
        pass

class max_base_height(termination_base):

    def step(self,input_dict):
        if (not self.mx_flag) : 
            self.height = input_dict['q'][2]
            self.mx_flag = 1

        if input_dict['q'][2] > ((1+self.params['threshold']) * self.height): 
            return True
        else:
            
            return False

    def reset(self):
        self.mx_flag = 0
        pass

class max_foot_height(termination_base):

    def step(self,input_dict):
        if input_dict['left_foot'][2] > self.params['threshold']: 
            return True
        elif input_dict['right_foot'][2] > self.params['threshold']: 
            return True
        else:
            return False

    def reset(self):
        pass

# class min_imitation_threshold(termination_base):

#     def step(self,input_dict):
#         if input_dict['motion_imitation'] < self.params['threshold']:
#             return True
#         else:
#             return False
#     def reset(self):
#         pass

class catwalk(termination_base):
    
    def step(self,input_dict):
        if (input_dict['left_foot'][1] < 0) or (input_dict['right_foot'][1] > 0): 
            return True
        else:
            return False

    def reset(self):
        pass

class tilt_twist_bend(termination_base):
    
    def step(self,input_dict):
        q = input_dict['q']
        euler = quat2euler([q[3],q[4],q[5],q[6]])

        if euler[0] > self.params['roll'] or euler[1] > self.params['pitch']: 
            return True
        else:
            return False

    def reset(self):
        pass

class leg_spread(termination_base):

    def step(self,input_dict):
        if abs(input_dict['left_foot'][1] - input_dict['right_foot'][1]) < self.params['threshold']: 
            return True
        else:
            return False

    def reset(self):
        pass

class y_bound(termination_base):

    def step(self,input_dict):
        if abs(input_dict['q'][1]) > self.params['threshold']: 
            return True
        else:
            return False

    def reset(self):
        pass

class mocap_epi_len(termination_base):

    def step(self,input_dict):
        if input_dict['mocap_n_step'] >= input_dict['mocap_len']:
            return True
        else:
            return False
    def reset(self):
        pass

class min_base_height_2D(termination_base):

    def step(self,input_dict):
        if input_dict['q'][1] < self.params['threshold']: 
            return True
        else:
            return False
    
    def reset(self):
        pass

class max_base_height_2D(termination_base):

    def step(self,input_dict):
        if input_dict['q'][1] > self.params['threshold']: 
            return True
        else:
            return False

    def reset(self):
        pass

class max_min_vel(termination_base):

    def step(self,input_dict):
        if (input_dict['dq'][0] > self.params['max']) or (input_dict['dq'][0] < self.params['min']): 
            return True
        else:
            
            return False

    def reset(self):
        pass

class healthy_angle_range(termination_base):

    def step(self,input_dict):
        if (input_dict['q'][2] > self.params['max']) or (input_dict['q'][2] < self.params['min']): 
            return True
        else:
            return False

    def reset(self):
        pass

class max_ep_len(termination_base):

    def step(self,input_dict):
        self.n_steps += 1
        if (self.n_steps >= 1500): 
            return True
        else:
            return False

    def reset(self):
        self.n_steps = 0
        pass