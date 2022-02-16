
from gym_hmm_ec.envs.bipedal_env import BipedEnv 
from gym_hmm_ec.controllers.pd_controller import PDController 
import matplotlib.pyplot as plt
import numpy as np
import mujoco_py

assets_path = './gym_hmm_ec/envs/assets/'
model_name = 'testWrite'
model = mujoco_py.load_model_from_path(assets_path+"models/"+model_name+".xml")
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

sim.reset()

viewer._paused = True
viewer.cam.distance = 6
cam_pos = [0.0, 0.0, 0.0]

for i in range(3):        
    viewer.cam.lookat[i]= cam_pos[i] 
viewer.cam.elevation = -15
viewer.cam.azimuth = 220


# play simulation 
while True:
    sim.step()
    viewer.render()



'''
# environment config and setup
env_conf = {
            'set_on_rack': True,
            'render': True,
            'model_name':'humanoid_no_hands',
            'mocap':False
            }

env = BipedEnv(**env_conf)

# initialse the env,reset simualtion
env.reset()

# keep the similation in pause until activated manually
if env.env_params['render']:
    env.viewer._paused = True

for _ in range(2000):

    # for sensors
    sensor_name = 'torso_gyro'
    sensor_val = env.sim.data.get_sensor(sensor_name)
    print(_,sensor_name,sensor_val)
    control_actions = np.zeros(shape=env.n_act_joints)
    obs,reward,done,info = env.step(action = control_actions )

env.close()
'''
