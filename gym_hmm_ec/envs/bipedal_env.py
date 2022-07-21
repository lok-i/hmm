from tracemalloc import start
import numpy as np
import os
import gym
from gym import  spaces
from gym import utils
from mujoco_py.generated import const
from utils import misc_functions
import gym_hmm_ec.envs.mujoco_env as mujoco_env

import gym_hmm_ec.envs.observations as obs
import gym_hmm_ec.envs.actions as act
import gym_hmm_ec.envs.rewards as rew
import gym_hmm_ec.envs.terminations as trm

import random

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



        self.init_observations()
        self.init_actions()

        self.init_rewards()
        self.init_terminations()
        self.init_mocap_data()

        mujoco_env.MujocoEnv.__init__(
                                      self, 
                                      model_name = self.env_params['model_name']+'.xml',
                                      frame_skip= 1,
                                      )
        utils.EzPickle.__init__(self)

        self.reset_model()
        
        # TODO: obs - action deicsion
        dummy_obs = self.get_observation()
        self.obs_dim = dummy_obs.shape[0]
        self.action_dim = self.get_action(inital=True)

        high = np.full(self.action_dim,np.inf)
        low =  np.full(self.action_dim,-np.inf)
        self.action_space = spaces.Box(low=low, high=high)
        high = np.full(self.obs_dim,1)
        low =  np.full(self.obs_dim,-1)
        self.observation_space = spaces.Box(low, high)


        self.n_act_joints = len(self.sim.data.ctrl)

    def init_observations(self):
        self.observation_list = []
        self.observations = []
        for obs_name in self.env_params['observations']:
            self.observation_list.append(obs_name)
            observation_params = self.env_params['observations'][obs_name]
            observation_class = getattr(obs,obs_name)
            self.observations.append(observation_class(observation_params))

    def init_actions(self):
        self.action_list = []
        self.actions = []
        for act_name in self.env_params['actions']:
            self.action_list.append(act_name)
            action_params = self.env_params['actions'][act_name]
            action_class = getattr(act,act_name)
            self.actions.append(action_class(action_params))

    def init_rewards(self):
        self.reward_list = []
        self.rewards = []
        for rew_name in self.env_params['rewards']:
            self.reward_list.append(rew_name)
            reward_params = self.env_params['rewards'][rew_name]
            reward_class = getattr(rew,rew_name)
            self.rewards.append(reward_class(reward_params))
        
    def init_terminations(self):
        self.termination_list = []
        self.terminations = []
        for trm_name in self.env_params['terminations']:
            self.termination_list.append(trm_name)
            termination_params = self.env_params['terminations'][trm_name]
            termination_class = getattr(trm,trm_name)
            self.terminations.append(termination_class(termination_params))

    def init_mocap_data(self):
        if 'mocap_data' in self.env_params.keys():
            self.mocap_data = np.load(self.env_params['mocap_data']['processed_data_path'])

            if os.path.isfile(self.env_params['mocap_data']['ik_solns_path']):
                self.ik_solns = np.load(self.env_params['mocap_data']['ik_solns_path'])['ik_solns']
            else:
                print("IK solns missing, run ./utils/compute_ik.py for the above mocap trial to generate the ik solns")
                exit()

    def step(self,action):

        applied_actuator_torque = self.get_action(policy_output=action)
        
        n_step_same_target = 1
        self.do_simulation(applied_actuator_torque, n_step_same_target)
        
        obs = self.get_observation()
        reward = self.get_reward()
        done = self.check_termination()
        
        if self.env_params['render']:
            self.render()
        
        # print('ctrl:', applied_actuator_torque)#, reward, done)
        # print('q :',  self.sim.data.qpos[:])#, reward, done)
        # if self.env_n_step == 0:
        #     print('rew :', reward)#, reward, done)
        # print('done:', done)#, reward, done)

        self.env_n_step += 1
        self.mocap_n_step += 1
        return obs, reward, done, {}

    def reset_model(self):

        self.env_n_step = 0
        self.mocap_n_step = 0

        for obs_condn in self.observations:
            obs_condn.reset()
        for act_condn in self.actions:
            act_condn.reset()
        for rew_condn in self.rewards:
            rew_condn.reset()
        for trm_condn in self.terminations:
            trm_condn.reset()


        # set any cutom initialisation here
        if 'initalisation' in self.env_params.keys():

            if 'random_on_mocap_trajectory' in self.env_params['initalisation'].keys():
                mocap_len = self.ik_solns.shape[0]
                self.mocap_n_step = np.random.randint(low=0,high=mocap_len)

                frame_rate = self.mocap_data['frame_rate']
                timestep = self.model.opt.timestep if self.model.opt.timestep < (1. / frame_rate) else (1. / frame_rate)

                qpos = self.ik_solns[self.mocap_n_step]
                prev_qpos =self.ik_solns [self.mocap_n_step-1]
                
                self.sim.data.qpos[:] = qpos
                self.sim.data.qvel[:] = np.concatenate([
                    (qpos[:3]-prev_qpos[:3]) / timestep,
                    misc_functions.mj_quat2vel(
                        misc_functions.mj_quatdiff(prev_qpos[3:7], qpos[3:7]), timestep),
                    (qpos[7:]-prev_qpos[7:]) / timestep
                ]
                ).ravel()

            if 'initial_pose' in self.env_params['initalisation'].keys():
                mocap_len = self.ik_solns.shape[0]
                self.mocap_n_step = np.random.randint(low=0,high=mocap_len)

                frame_rate = self.mocap_data['frame_rate']
                timestep = self.model.opt.timestep if self.model.opt.timestep < (1. / frame_rate) else (1. / frame_rate)

                qpos = self.ik_solns[0]
                qpos[2] = 0.82
                # prev_qpos =self.ik_solns[0]
                
                self.sim.data.qpos[:] = qpos
                # self.sim.data.qvel[:] = np.concatenate([
                #     (qpos[:3]-prev_qpos[:3]) / timestep,
                #     misc_functions.mj_quat2vel(
                #         misc_functions.mj_quatdiff(prev_qpos[3:7], qpos[3:7]), timestep),
                #     (qpos[7:]-prev_qpos[7:]) / timestep
                # ]
                # ).ravel()

            if 'initial_pose_2D' in self.env_params['initalisation'].keys():
                mocap_len = self.ik_solns.shape[0]
                self.mocap_n_step = 1

                frame_rate = self.mocap_data['frame_rate']
                timestep = self.model.opt.timestep if self.model.opt.timestep < (1. / frame_rate) else (1. / frame_rate)
                
                qpos = self.ik_solns[self.mocap_n_step]
                prev_qpos =self.ik_solns[self.mocap_n_step-1]
                
                qpos_2D = np.zeros(9)
                prev_qpos_2D = np.zeros(9)

                qpos_2D[0] = qpos[0]
                qpos_2D[1] = qpos[2]
                qpos_2D[2] = misc_functions.quat2euler(qpos[3:7])[1]
                qpos_2D[3:] = qpos[7:]

                prev_qpos_2D[0] = prev_qpos[0]
                prev_qpos_2D[1] = prev_qpos[2]
                prev_qpos_2D[2] = misc_functions.quat2euler(prev_qpos[3:7])[1]
                prev_qpos_2D[3:] = prev_qpos[7:]

                self.sim.data.qpos[:] = qpos_2D
                self.sim.data.qpos[1] = -0.04
                self.sim.data.qvel[:] = (qpos_2D-prev_qpos_2D) / timestep

            if 'random_on_mocap_trajectory_2D' in self.env_params['initalisation'].keys():
                mocap_len = self.ik_solns.shape[0]
                # self.mocap_n_step = np.random.randint(low=0,high=mocap_len)
                self.mocap_n_step = np.random.randint(low=0,high=200)

                frame_rate = self.mocap_data['frame_rate']
                timestep = self.model.opt.timestep if self.model.opt.timestep < (1. / frame_rate) else (1. / frame_rate)

                qpos = self.ik_solns[self.mocap_n_step]
                prev_qpos =self.ik_solns[self.mocap_n_step-1]
                
                qpos_2D = np.zeros(9)
                prev_qpos_2D = np.zeros(9)

                qpos_2D[0] = qpos[0]
                qpos_2D[1] = qpos[2]
                qpos_2D[2] = misc_functions.quat2euler(qpos[3:7])[1]
                qpos_2D[3:] = qpos[7:]

                prev_qpos_2D[0] = prev_qpos[0]
                prev_qpos_2D[1] = prev_qpos[2]
                prev_qpos_2D[2] = misc_functions.quat2euler(prev_qpos[3:7])[1]
                prev_qpos_2D[3:] = prev_qpos[7:]

                self.sim.data.qpos[:] = qpos_2D
                self.sim.data.qpos[1] = -0.04

                self.sim.data.qvel[:] = (qpos_2D-prev_qpos_2D) / timestep
                    


            if 'base_velocity' in self.env_params['initalisation'].keys():
                self.sim.data.qvel[0] = self.env_params['initalisation']['base_velocity'] 

            if 'initial_base_height' in self.env_params['initalisation'].keys():
                self.sim.data.qpos[2] = self.env_params['initalisation']['initial_base_height']

            if 'initial_base_height_2D' in self.env_params['initalisation'].keys():
                self.sim.data.qpos[1] = self.env_params['initalisation']['initial_base_height_2D']

            if 'step_position'in self.env_params['initalisation'].keys():
                # self.sim.data.qpos[2] = self.env_params['initalisation']['step_position']
                # self.sim.data.qpos[3] = 1 + random.random()
                # self.sim.data.qpos[4] = 0 + self.env_params['initalisation']['random']*random.random()
                # self.sim.data.qpos[5] = 0.1 + self.env_params['initalisation']['random']*random.random()
                # self.sim.data.qpos[6] = 2 + self.env_params['initalisation']['step_position']*random.random()
                # self.sim.data.qpos[7] = 0.2 + self.env_params['initalisation']['random']*random.random()
                self.sim.data.qpos[8] = -0.3 + self.env_params['initalisation']['step_position']*random.random()
                self.sim.data.qpos[9] = -0.14 + self.env_params['initalisation']['step_position']*random.random()
                # self.sim.data.qpos[10] = 1 + self.env_params['initalisation']['random']*random.random()
                # self.sim.data.qpos[11] = -0.5 + self.env_params['initalisation']['random']*random.random()
                self.sim.data.qpos[12] = -0.5 + self.env_params['initalisation']['step_position']*random.random()

            if 'step_position_2D'in self.env_params['initalisation'].keys():
                self.sim.data.qpos[1] = -0.03
                self.sim.data.qpos[3] = 0 + self.env_params['initalisation']['step_position_2D']*random.uniform(-1,1)
                self.sim.data.qpos[4] = 0.1 + self.env_params['initalisation']['step_position_2D']*random.uniform(-1,1)
                self.sim.data.qpos[5] = -0.2 + self.env_params['initalisation']['step_position_2D']*random.uniform(-1,1)
                self.sim.data.qpos[6] = 0 + self.env_params['initalisation']['step_position_2D']*random.uniform(-1,1)
                self.sim.data.qpos[7] = -0.2 + self.env_params['initalisation']['step_position_2D']*random.uniform(-1,1)
                self.sim.data.qpos[8] = -0.1 + self.env_params['initalisation']['step_position_2D']*random.uniform(-1,1)
                self.sim.data.qvel[7] = 0.8 + self.env_params['initalisation']['step_position_2D']*random.uniform(-1,1)


            if 'randomness'in self.env_params['initalisation'].keys():
                self.sim.data.qpos[:] += self.env_params['initalisation']['randomness']*random.uniform(-1,1)
                

        self.sim.forward()
        
        # depreciated
        # if self.env_params['mocap']:
        #     self.attach_mocap_objects()
        
        initial_obs = self.get_observation()
        
        # print("Initial state:",initial_obs.shape)
        return initial_obs
    
    def get_observation(self):

        data_dict = {}
        data_dict['q'] = self.sim.data.qpos[:].copy()
        data_dict['dq'] = self.sim.data.qvel[:].copy()
        
        obs_vector = []
        for observation in self.observations:
            obs_vector.append(observation.step(input_dict = data_dict) )

        obs_vector = np.concatenate(obs_vector).ravel()
            

        return obs_vector

    def get_action(self,inital=False,policy_output=None):
        
        data_dict = {}
        data_dict['env'] = self

        if inital:
            act_dim = 0
            for action in self.actions:
                act_dim += action.params['dim']
            return act_dim 

        else:
            act_vector = []
            start_id = 0
            for action in self.actions:
                act_vector.append(action.step(policy_output = policy_output[start_id: start_id+action.params['dim']],input_dict = data_dict) )
                # act_vector.append(action.step(policy_output = policy_output[start_id: start_id+action.params['dim']]))

                start_id += action.params['dim']
            act_vector = np.array(act_vector)
            
            # sum of all torques finally
            act_vector = np.sum(act_vector,axis=0)

            return act_vector
     
    def get_reward(self):

        data_dict = {}
        data_dict['sim.data'] = self.sim.data
        data_dict['sim.model'] = self.sim.model
        

        if 'mocap_data' in self.env_params.keys(): 
            data_dict['ik_solns'] = self.ik_solns
            data_dict['mocap_data'] = self.mocap_data

        total_reward = 0
        self.reward_value_list = {}
        
        for reward,rew_name in zip(self.rewards,self.reward_list):
            reward_val = reward.step(input_dict = data_dict)         
            # self.reward_value_list = {rew_name:reward_val}
            self.reward_value_list[rew_name] = reward_val 
            total_reward += reward_val

        return total_reward

    def check_termination(self):
        data_dict = {}
        if 'mocap_data' in self.env_params.keys():
            data_dict['mocap_len'] = self.ik_solns.shape[0]
            data_dict['mocap_n_step'] = self.mocap_n_step
            # data_dict['motion_imitation'] = self.reward_value_list['motion_imitation']

        data_dict['q'] = self.sim.data.qpos[:].copy()
        data_dict['dq'] = self.sim.data.qvel[:].copy() 

        data_dict['left_foot'] = self.sim.data.body_xpos[self.model.body_name2id("left_leg/foot")]
        data_dict['right_foot'] = self.sim.data.body_xpos[self.model.body_name2id("right_leg/foot")]

        dones = []
        for termination in self.terminations:
            dones.append(termination.step(input_dict = data_dict))
        
        if True in dones:
            return True
        else:
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



