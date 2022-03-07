import numpy as np
import time
import gym
import math
from gym import utils
from gym import spaces
from mujoco_py.generated import const
from utils import misc_functions
import gym_hmm_ec.envs.mujoco_env as mujoco_env

class BipedEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self,**kwargs):
        
        self.env_params = kwargs
        necessary_env_args = ['model_name']
        
        default_env_args = {
                          'render':True,
                          'set_on_rack': False,
                          'mocap':False
                          }
        
        for key in necessary_env_args:
            if key not in self.env_params.keys():
                raise Exception('necessary arguments are absent. Check:'+str(necessary_env_args))        
        
        for key in default_env_args.keys():
            if key not in self.env_params.keys():
                self.env_params[key] = default_env_args[key]
        
        
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
        if self.env_params['mocap']:
            applied_motor_torque = np.zeros(self.n_act_joints)
        else:
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
        self.sim.forward()
        if self.env_params['mocap']:
            self.attach_mocap_objects()
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
    
    def attach_mocap_objects(self):

        for body_name in self.model.body_names:
            if body_name not in ['world', 'floor'] and 'mocap_' not in body_name:

                body_pos = self.sim.data.get_geom_xpos(body_name)

                self.sim.data.set_mocap_pos('mocap_'+body_name, body_pos )
        
    def view_vector_arrows(self,vec,vec_point,vec_mag_max=20,vec_txt=''):

        vec_mag = np.round(np.linalg.norm(vec),1) 
        arrow_scale = vec_mag/vec_mag_max
        if self.env_params['render']:
            self.viewer.add_marker(
                        pos=vec_point , #position of the arrow
                        size= arrow_scale*np.array([0.03,0.03,1]), #size of the arrow
                        mat= misc_functions.calc_rotation_vec_a2b(vec), # orientation as a matrix
                        rgba=np.array([0.,0.,1.,1.]),#color of the arrow
                        type=const.GEOM_ARROW,
                        label= vec_txt,
                        )        



