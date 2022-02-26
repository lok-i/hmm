from gym_hmm_ec.envs.bipedal_env import BipedEnv 
import numpy as np
from tqdm import tqdm
import argparse
from mujoco_py import functions
from gym_hmm_ec.envs.utils import misc_functions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ik_soln_filename',help='name of the precomputed ik soln file',default='AB1_Session1_Right6_Left6',type=str)
    parser.add_argument('--export_solns',help='whether to export the ik solns',default=False, action='store_true')
    parser.add_argument('--render',help='whether to render while solving for ik',default=False, action='store_true')
    args = parser.parse_args()

    assets_path = './gym_hmm_ec/envs/assets/'
    c3d_file_name = 'mocap_data/c3ds/Trial_1'

    # environment config and setup
    env_conf = {
                'set_on_rack': False,
                'render': args.render,
                'model_name':'default_humanoid_mocap_generated',
                'mocap':False
                }

    env = BipedEnv(**env_conf)
    env.model.opt.gravity[2] = 0

    # initialse the env,reset simualtion
    env.reset()


    # keep the similation in pause until activated manually
    if env.env_params['render']:
        env.viewer._paused = True
        env.viewer.cam.distance = 3
        cam_pos = [0.0, 0.5, 0.75]

        for i in range(3):        
            env.viewer.cam.lookat[i]= cam_pos[i] 
        env.viewer.cam.elevation = -15
        env.viewer.cam.azimuth = 180

    ik_soln = np.load(assets_path+"our_data/ik_solns/"+args.ik_soln_filename+'.npz')['ik_solns']
    # ext_frc = np.load(assets_path+"our_data/"+c3d_file_name+'_ext_frc.npy')
    id_solns = []
    frame_rate = 100.
    timestep = 1. / frame_rate
    prev_qpos = np.zeros(env.sim.data.qpos.shape)
    for qpos in tqdm(ik_soln):

            env.sim.data.qpos[:] =  qpos

            env.sim.data.qvel[:] = np.concatenate(       [
                    (qpos[:3]-prev_qpos[:3])/ timestep,
                    misc_functions.mj_quat2vel(
                        misc_functions.mj_quatdiff(prev_qpos[3:7], qpos[3:7]), timestep),
                    (qpos[7:]-prev_qpos[7:])/ timestep
                    ]
                ).ravel() 
            obs,reward,done,info = env.step(action = np.zeros(shape=env.n_act_joints))
            
            functions.mj_inverse(env.model,env.sim.data)
            id_solns.append( env.sim.data.qfrc_inverse.tolist())
            prev_qpos = qpos.copy()

    print(np.array(id_solns).shape)
