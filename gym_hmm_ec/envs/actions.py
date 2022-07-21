import numpy as np
from gym_hmm_ec.controllers.pd_controller import PDController 
from utils import misc_functions


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

    def step(self,policy_output,input_dict):
        actions = self.unnormalize_action(policy_output)

        return actions
    def unnormalize_action(self,norm_action):
        
        return self.params['torque_max']*norm_action
    
    def reset(self):
        pass

class check_pd_control(action_base):

    def step(self,policy_output,input_dict):
        actions = self.unnormalize_action(policy_output) + self.get_pd_torques(input_dict)
        return actions

    def unnormalize_action(self,norm_action):
        
        return self.params['torque_max']*norm_action

    def get_pd_torques(self,input_dict):

        env = input_dict['env']
        timestep = env.model.opt.timestep if env.model.opt.timestep < (1. / self.frame_rate) else (1. / self.frame_rate)
        pd_controller = PDController(
                             kps=np.full(env.n_act_joints,100),
                             kds=np.full(env.n_act_joints,1),
                             )

        q_act_des = np.zeros(env.n_act_joints)
        dq_act_des = np.zeros(env.n_act_joints)
        base_dof = env.sim.data.qpos.shape[0] - env.n_act_joints

        qpos = env.sim.data.qpos[:].copy()
        qvel = env.sim.data.qvel[:].copy()

        qpos_ik = self.ik_solns[self._n_step+1 + env.mocap_n_step][7:]
        prev_qpos_ik = self.ik_solns[self._n_step + env.mocap_n_step][7:]
        q_act_des = qpos_ik
        dq_act_des = (qpos_ik - prev_qpos_ik) / timestep

        torque = pd_controller.get_torque(
                                        q_des = q_act_des,
                                        dq_des= dq_act_des,
                                        q = qpos[base_dof :].copy(),
                                        dq= qvel[qvel.shape[0] - env.n_act_joints :].copy()
                                        )

        self._n_step += 1

        return torque
    
    def reset(self):
        self.mocap_data = np.load('data/mitmcl_data/marker_data/processed_data/AB3_Session1_Right10_Left10_from_2000_to_2500.npz')
        self.ik_solns = np.load('data/mitmcl_data/ik_solns/AB3_Session1_Right10_Left10_from_2000_to_2500.npz')['ik_solns']
        self.mocap_len = self.ik_solns.shape[0]
        self.frame_rate = self.mocap_data['frame_rate']
        self._n_step = 0
        pass

