import numpy as np
import time
import gym
import math
from gym import utils
from gym import spaces
import gym_hmm_ec.envs.mujoco_env as mujoco_env

class BipedEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self,**kwargs):
        
        self.env_params = kwargs
        necessary_args = ['render','model_name','set_on_rack']
        for key in necessary_args:
            if key not in self.env_params.keys():
                raise Exception('necessary arguments are absent. Check:'+str(necessary_args))        
        
        
        
        # base env config
        self.obs_dim = 12 
        self.action_dim = 5
        
        mujoco_env.MujocoEnv.__init__(
                                      self, 
                                      model_name = self.env_params['model_name']+'.xml',
                                      frame_skip= 1,
                                      )
        utils.EzPickle.__init__(self)

        self.n_act_joints = len(self.sim.data.ctrl)
        print("No. of actuated joints:",self.n_act_joints)

    def step(self,action):
        applied_motor_torque = action
        n_step_same_target = 1
        self.do_simulation(applied_motor_torque, n_step_same_target)
        
        obs = self.get_observation()
        reward = self.get_reward()
        done = self.check_termination()
        
        if self.env_params['render']:
            self.render()

        return obs, reward, done, {}

    def reset_model(self):
        # TBI
        initial_obs = self.get_observation()
        print("Initial state:",initial_obs)
        return initial_obs
    
    def get_observation(self):
        qpos,qvel = self.get_state()
        
        # TBI
        # base_pos = qpos[0: qpos.shape[0] - self.n_act_joints ]
        # base_vel = qpos[0: qvel.shape[0] - self.n_act_joints ]
        # joint_pos = qpos[qpos.shape[0] - self.n_act_joints + 1  :]
        # joint_vel = qpos[qvel.shape[0] - self.n_act_joints : ]
        # print(qpos.shape[0], qvel.shape[0])        

        return np.concatenate((qpos,qvel)).ravel()

    def get_state(self):
        qpos = self.sim.data.qpos.copy()
        qvel = self.sim.data.qvel.copy()
        return qpos,qvel 
        
    def get_reward(self):
        # TBI
        return 0 

    def check_termination(self):
        # TBI
        return False