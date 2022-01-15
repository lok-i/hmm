import os
import glfw
import copy
from mujoco_py.utils import rec_copy, rec_assign

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import six

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

equality_constraint_id2type = {
                                'connect':0,
                                'weld':1,
                                'joint':2,
                                'tendon':3,
                                'distance':4
                                }

class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_name, frame_skip):
        if model_name.startswith("/"):
            fullpath = model_name
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets/models", model_name)
        if not path.exists(fullpath):    
            raise IOError("File %s does not exist" % fullpath)

        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)

        
        self.sim = mujoco_py.MjSim(self.model)

        # to deactivate gravity
        # self.model.opt.gravity[2] = 0
        # print(self.model.opt.gravity)

        if self.env_params['render']:
            self.viewer = mujoco_py.MjViewer(self.sim)

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        high = np.full(self.action_dim,1)
        low =  np.full(self.action_dim,-1)
        self.action_space = spaces.Box(low=low, high=high)
        high = np.full(self.obs_dim,1)
        low =  np.full(self.obs_dim,-1)
        self.observation_space = spaces.Box(low, high)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(self): # -1 is random
        self.sim.reset()

        self.remove_all_weld_contraints()
        ob = self.reset_model()
        self.set_required_weld_contraints()
        if self.env_params['render'] and self.viewer is not None:
            self.viewer_setup()
        return ob

    def remove_all_weld_contraints(self):

        for eq_id,eq_type in enumerate(self.model.eq_type):
            if eq_type ==  equality_constraint_id2type['weld']:
                self.model.eq_active[eq_id] = 0

    def set_required_weld_contraints(self):

        for eq_id,(eq_obj1id,eq_obj2id,eq_type) in enumerate(zip(self.model.eq_obj1id,self.model.eq_obj2id,self.model.eq_type)):
            
            if eq_type ==  equality_constraint_id2type['weld']:
                
                if self.model.body_id2name(eq_obj1id) == 'floor':
                    if self.env_params['set_on_rack']:
                        self.model.eq_active[eq_id] = 1
                if 'mocap_' in self.model.body_id2name(eq_obj1id) or 'mocap_' in self.model.body_id2name(eq_obj2id):
                    if self.env_params['mocap']:
                        self.model.eq_active[eq_id] = 1 
    
    def set_state(self, qpos, qvel):
        
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def get_state(self):
        raise NotImplementedError
    
    def assign_desired_vel(self, desired_vel):
        raise NotImplementedError

    def just_simulate(self,n_frames):
        for _ in range(n_frames):
            self.sim.step()

    def get_sensor_data(self,sensor_name):
        return self.sim.data.get_sensor(sensor_name)    

    @property
    def dt(self):
        # print(self.model.opt.timestep)
        return self.model.opt.timestep * self.frame_skip 

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        # print("ctrl:",self.sim.data.ctrl)
        for _ in range(n_frames):
            self.sim.step()
    

    def render(self, mode='human'):
        if mode == 'rgb_array':
            self._get_viewer().render()
            # window size used for old mujoco-py:
            width, height = 500, 500
            # data = self._get_viewer()read_pixels(width, height, depth=False)
            data =  self._read_pixels_as_in_window()
            # cv2.imshow("test:",data)
            # exit()
            # original image is upside-down, so flip it
            # return data#[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer_setup()
        return self.viewer

    def get_body_com(self, body_name):
        return self.sim.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])
