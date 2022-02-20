from gym_hmm_ec.envs.bipedal_env import BipedEnv 
from gym_hmm_ec.controllers.pd_controller import PDController 
import matplotlib.pyplot as plt
import numpy as np
from gym_hmm_ec.envs.utils import misc_functions
from mujoco_py.generated import const




# TODO: Fix the mocap integration of env with the generated model
# environment config and setup
env_conf = {
            'set_on_rack': True,
            'render': True,
            'model_name': 'humanoid_no_hands_mocap_generated',
            'mocap':False # problem when true
            }

env = BipedEnv(**env_conf)

# initialse the env,reset simualtion
env.reset()

# keep the similation in pause until activated manually
if env.env_params['render']:
    env.viewer._paused = True


frc_max = 50
while True:

    # print(frc_mag)
    # env.sim.data.qfrc_applied[0] = frc_ptb[0]
    # env.sim.data.qfrc_applied[1] = frc_ptb[1]
    # env.sim.data.qfrc_applied[2] = frc_ptb[2]
    body_names = ['right_leg/foot','right_leg/foot_dummy1','right_leg/foot_dummy2']
    frc_ptbs = [ np.array([10,0,10]),np.array([-5,0,20]),np.array([2,0,20])]
    for frc_ptb, body_name in zip(frc_ptbs,body_names):
        body_id = env.sim.model.body_name2id(body_name)
        # frc_ptb = np.array([100,0,0])
        frc_mag = np.round(np.linalg.norm(frc_ptb),1) 
        frc_pos = env.sim.data.get_body_xpos(body_name)

        for i in range(3):
            env.sim.data.xfrc_applied[body_id][i] = frc_ptb[i]
        # print( env.sim.data.xfrc_applied.shape)
        arrow_scale = frc_mag/frc_max
        env.viewer.add_marker(
                    pos=frc_pos , #position of the arrow
                    size= arrow_scale*np.array([0.03,0.03,1]), #size of the arrow
                    mat= misc_functions.calc_rotation_vec_a2b(frc_ptb), # orientation as a matrix
                    rgba=np.array([0.,0.,1.,1.]),#color of the arrow
                    type=const.GEOM_ARROW,
                    label= '',#str(frc_mag)+' N'
                    )
    control_actions = np.zeros(shape=env.n_act_joints)
    obs,reward,done,info = env.step(action = control_actions )