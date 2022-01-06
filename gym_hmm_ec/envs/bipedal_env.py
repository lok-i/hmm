import numpy as np
import time
import gym
import math
from gym import utils
from gym import spaces
import gym_hmm_ec.envs.mujoco_env as mujoco_env

class BipedEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,model_name = 'walker'):
        self.is_render = True
        # base env config
        self.obs_dim = 12
        self.action_dim = 5
        mujoco_env.MujocoEnv.__init__(self, model_name+'.xml', 1)
        utils.EzPickle.__init__(self)

        print(len(self.sim.data.ctrl))


    def step(self,action):
        applied_motor_torque = np.zeros(len(self.sim.data.ctrl))
        # applied_motor_torque = np.random.random(len(self.sim.data.ctrl))

        n_step_same_target = 1
        self.do_simulation(applied_motor_torque, n_step_same_target)
        obs = self.get_observation()
        reward = self.get_reward()
        done = self.check_termination()
        return obs, reward, done, {}

    def reset_model(self):
        initial_obs = self.get_observation()
    def get_observation(self):
        return None

    def get_reward(self):
        return 0 

    def check_termination(self):
        return False