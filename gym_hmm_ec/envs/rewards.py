import numpy as np
import utils.misc_functions as misc_functions


class reward_base():

    def __init__(self,params) -> None:
        self.params = params
        
    def step(self):
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError

class zero_reward(reward_base):
 
    def step(self,input_dict):
        return 0

    def reset(self):
        pass

class forward_x_base_pos(reward_base):

        
    def step(self,input_dict):
        p = input_dict['sim.data'].qpos[:]
        return self.params['k']*np.power(p[0],self.params['pow'])

    def reset(self):
        pass

class penalise_effort(reward_base):
   
    def step(self,input_dict):
        torques = input_dict['sim.data'].ctrl[:]
        return self.params['k']*(np.power(torques,self.params['pow']).sum())
    def reset(self):
        pass


class forward_x_base_vel(reward_base):

    def step(self,input_dict):
        v = input_dict['sim.data'].qvel[:]
        return self.params['k']*np.power(v[0],self.params['pow'])

    def reset(self):
        pass

class target_x_base_vel_2D(reward_base):

    def step(self,input_dict):
        v = input_dict['sim.data'].qvel[:].copy()
        error = abs(v[0] -self.params['target_vel'])
        return self.params['k']*error

    def reset(self):
        pass

class forward(reward_base):

    def step(self,input_dict):
        dq = input_dict['sim.data'].qvel[:].copy()
        vel_error = abs(dq[0] - self.params['target_vel'])
        y_vel = dq[1]
        twist = dq[5]
        reward = -1*self.params['alpha_2']*vel_error - np.power(y_vel,2) - np.power(twist,2)
        return reward

    def reset(self):
        pass

class alive(reward_base):

    def step(self,input_dict):
        r_alive = self.params['alive_bonus']*self.params['target_vel']
        return r_alive

    def reset(self):
        pass

class energy(reward_base):

    def step(self,input_dict):
        tau = input_dict['sim.data'].qfrc_actuator[:].copy().T
        dq = input_dict['sim.data'].qvel[:].copy()
        return -1*self.params['alpha_1']*(np.dot(tau,dq))

    def reset(self):
        pass

class smoothen_action(reward_base):

    def step(self,input_dict):
        curr_action = input_dict['sim.data'].qfrc_actuator[:]
        reward = self.params['k']*np.linalg.norm(curr_action - self.prev_action)
        self.prev_action = curr_action
        return reward

    def reset(self):
        self.prev_action = np.zeros(self.params['action_dim'])
        pass

class optimal_height(reward_base):

    def step(self,input_dict):
        q = input_dict['sim.data'].qpos[:]
        z = q[1]
        return self.params['k']*np.power((z+0.05),self.params['pow'])

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

        # frame_rate = input_dict['mocap_data']['frame_rate']
        # timestep = input_dict['sim.model'].opt.timestep if input_dict['sim.model'].opt.timestep < (1. / frame_rate) else (1. / frame_rate)

        # dq_des = np.concatenate([
        #             (q_des[:3]-self.q_des_prev[:3]) / timestep,
        #             misc_functions.mj_quat2vel(
        #                 misc_functions.mj_quatdiff(self.q_des_prev[3:7], q_des[3:7]), timestep),
        #             (q_des[7:]-self.q_des_prev[7:]) / timestep
        #         ]
        #         ).ravel()

        # self.q_des_prev = q_des

        # TBD        
        # target_base_linVel = dq_des[0:3]
        # target_base_angVel = dq_des[3:6]
        # target_joint_vel = dq_des[6:]

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
        # self.q_des_prev = np.zeros(13)
        self._n_step = 0


class motion_imitation_2D(reward_base):


    def step(self,input_dict):

        qpos = input_dict['sim.data'].qpos[:]

        q_des = input_dict['ik_solns'][self._n_step]
        q_des_2D = np.zeros(9)

        q_des_2D[0] = q_des[0]
        q_des_2D[1] = q_des[2]
        q_des_2D[2] = misc_functions.quat2euler([q_des[3],q_des[4],q_des[5],q_des[6]])[1]
        q_des_2D[3:] = q_des[7:]
        
        target_base_pos = q_des[0:2]
        target_base_ori = q_des[2]
        target_joint_pos = q_des[3:]

        # frame_rate = input_dict['mocap_data']['frame_rate']
        # timestep = input_dict['sim.model'].opt.timestep if input_dict['sim.model'].opt.timestep < (1. / frame_rate) else (1. / frame_rate)

        # dq_des = np.concatenate([
        #             (q_des[:3]-self.q_des_prev[:3]) / timestep,
        #             misc_functions.mj_quat2vel(
        #                 misc_functions.mj_quatdiff(self.q_des_prev[3:7], q_des[3:7]), timestep),
        #             (q_des[7:]-self.q_des_prev[7:]) / timestep
        #         ]
        #         ).ravel()

        # self.q_des_prev = q_des

        # TBD        
        # target_base_linVel = dq_des[0:3]
        # target_base_angVel = dq_des[3:6]
        # target_joint_vel = dq_des[6:]

        ith_norm = int(self.params['kernel'].partition('norm_l')[2])
        kernel_errors =np.array(
                                [
                                np.linalg.norm(np.subtract(target_base_pos,qpos[0:2]),ord=ith_norm),
                                np.linalg.norm(np.subtract(target_base_ori,qpos[2]),ord=ith_norm),
                                np.linalg.norm(np.subtract(target_joint_pos,qpos[3:]),ord=ith_norm),

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
        # self.q_des_prev = np.zeros(13)
        self._n_step = 0