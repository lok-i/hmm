import numpy as np

class reward_base():

    def __init__(self,params) -> None:
        self.params = params
        
    def step(self):
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError


class forward_x_base_pos(reward_base):

        
    def step(self,input_dict):
        px = input_dict['q'][0]
        return self.params['k']*np.power(px,self.params['pow'])


    def reset(self):
        pass

class penalise_effort(reward_base):

        
    def step(self,input_dict):
        torques = input_dict['ctrl']
        return self.params['k']*np.power(torques,self.params['pow']).sum()
    def reset(self):
        pass


class forward_x_base_vel(reward_base):

        
    def step(self,input_dict):
        vx = input_dict['dq'][0]
        return self.params['k']*np.power(vx,self.params['pow'])


    def reset(self):
        pass


class motion_imitation(reward_base):


    def step(self,input_dict):

        qpos = input_dict['q']
        qvel = input_dict['dq']

        q_des = input_dict['ik_solns']
        
        target_base_pos = q_des[self._n_step][0:3]
        target_base_ori = q_des[self._n_step][3:7]
        target_joint_pos = q_des[self._n_step][7:]

        # TBD        
        # target_base_linVel = input_dict['dq_des'][0:3]
        # target_base_angVel = input_dict['dq_des'][3:6]
        # target_joint_vel = input_dict['qd_des'][6:]

        ith_norm = int(self.params['kernel'].partition('norm_l')[2])
        kernel_errors =np.array(
                                [
                                np.linalg.norm(np.subtract(target_base_pos,qpos[0:3]),ord=ith_norm),
                                np.linalg.norm(np.subtract(target_base_ori,qpos[3:7]),ord=ith_norm),
                                np.linalg.norm(np.subtract(target_joint_pos,qpos[7:]),ord=ith_norm),

                                # np.linalg.norm(np.subtract(target_base_linVel,qvel[0:3]),ord=ith_norm),
                                # np.linalg.norm(np.subtract(target_base_angVel,qvel[3:6]),ord=ith_norm),
                                # np.linalg.norm(np.subtract(target_joint_vel,qvel[6:3]),ord=ith_norm)
                                ]
                                )

        weighted_error_terms = np.multiply(self.params['exp_const_list'],kernel_errors)
        reward_exponentials = np.exp(weighted_error_terms)
        
        total_reward = np.sum(np.multiply(self.params['weight_list'],reward_exponentials))
        self._n_step += 1
        return total_reward

    def reset(self):
        self._n_step = 0