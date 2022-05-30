from ntpath import join
from gym_hmm_ec.envs.bipedal_env import BipedEnv
import numpy as np
from tqdm import tqdm
import argparse
import os
import yaml
from mujoco_py import functions
from utils import misc_functions
import matplotlib.pyplot as plt
from gym_hmm_ec.controllers.pd_controller import PDController 


def qpos_quat2rpy(qpos):
    return np.concatenate([
            qpos[:3],
            misc_functions.quat2euler(qpos[3:7]),
            qpos[7:]    
                    ]).ravel()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--processed_filepath',help='name of the preprocessed mocap file',default='AB1_Session1_Right6_Left6',type=str)

    parser.add_argument('--plot_solns', help='whether to plot the id solns',
                        default=False, action='store_true')
    parser.add_argument('--model_filename',help='name of the model file',
    default='default_humanoid_mocap',type=str)

    parser.add_argument('--render', help='whether to render while solving for id',
                        default=False, action='store_true')
    args = parser.parse_args()

    assets_path = './gym_hmm_ec/envs/assets/'
    c3d_file_name = 'mocap_data/c3ds/Trial_1'

    # environment config and setup
    env_conf = {
        'set_on_rack': False,
        'render': args.render,
        'model_name': args.model_filename if '.xml' not in args.model_filename else args.model_filename.replace('.xml',''),
        'mocap': False
    }
    # marker config
    marker_confpath = args.processed_filepath.split('processed_data/')[0]+'confs/' \
                        + args.processed_filepath.split('processed_data/')[-1].split('_from')[0]+'.yaml' 

    marker_config_file = open(marker_confpath, 'r+')
    marker_conf = yaml.load(marker_config_file, Loader=yaml.FullLoader)

    env = BipedEnv(**env_conf)

    pd_controller = PDController(
                                kps=np.full(env.n_act_joints,10.),
                                kds=np.full(env.n_act_joints,0.1),
                                )


    # initialse the env,reset simualtion
    env.reset()

    # keep the similation in pause until activated manually
    if env.env_params['render']:
        env.viewer._paused = True
        env.viewer.cam.distance = 3
        cam_pos = [0.0, 0.5, 0.75]

        for i in range(3):
            env.viewer.cam.lookat[i] = cam_pos[i]
        env.viewer.cam.elevation = -15
        env.viewer.cam.azimuth = 180


    mocap_data = np.load(args.processed_filepath)

    ik_soln_filpath =  args.processed_filepath.split('marker_data/processed_data/')[0]+'ik_solns/' \
                        + args.processed_filepath.split('marker_data/processed_data/')[-1] 

    if os.path.isfile(ik_soln_filpath):
        ik_solns = np.load(ik_soln_filpath)['ik_solns']
    else:
        print("Missing: IK solutions absent, either not computed or not saved.")
        exit()
    id_solns = []
    frame_rate = mocap_data['frame_rate']
    prev_qpos = np.zeros(env.sim.data.qpos.shape)

    timestep = env.model.opt.timestep if env.model.opt.timestep < (1. / frame_rate) else (1. / frame_rate)

    grf_data = mocap_data['grfs'] 
    cop_data = mocap_data['cops'] 
    print("After:",ik_solns.shape, grf_data.shape,cop_data.shape)
    print("env timestep:",timestep)

    step = 0


    joint_actuator_to_chk = "right_leg/ankle_x"
    actuator_id_being_chkd = env.model.actuator_name2id(joint_actuator_to_chk) 
    base_dof = env.sim.data.qpos.shape[0] - env.n_act_joints

    dq_act_des = np.zeros(env.n_act_joints)

    for qpos,grf,cop in tqdm(zip(ik_solns,grf_data,cop_data),total=ik_solns.shape[0]):
        
        # env.sim.data.qpos[:] = qpos
        # if step == 0:
        #     prev_qpos = qpos.copy()
        #     prev_qvel = np.zeros(env.model.nv)


        # env.sim.data.qvel[:] = np.concatenate([
        #     (qpos[:3]-prev_qpos[:3]) / timestep,
        #     misc_functions.mj_quat2vel(
        #         misc_functions.mj_quatdiff(prev_qpos[3:7], qpos[3:7]), timestep),
        #     (qpos[7:]-prev_qpos[7:]) / timestep
        # ]
        # ).ravel()

        # with FD to obtain qacc
        # obs, reward, done, info = env.step(action=np.zeros(shape=env.n_act_joints))
        # env.sim.step()
        # functions.mj_forward(env.model, env.sim.data)


        q,dq = env.get_state()
        q_act_des = qpos
        torque = pd_controller.get_torque(
                                        q_des = q_act_des[base_dof :].copy(),
                                        dq_des= dq_act_des,
                                        q = q[base_dof :].copy(),
                                        dq= dq[dq.shape[0] - env.n_act_joints :].copy()
                                        )

        obs,reward,done,info = env.step(action = torque)

        env.sim.step()
        if env.env_params['render']:
            env.render()

        
        
        prev_qpos = qpos.copy()
        prev_qvel = env.sim.data.qvel.copy()

        # to rotate camera 360 around the model
        # env.viewer.cam.azimuth += 0.25 #180
        
        step += 1

    if args.plot_solns:
        timestep = 1. / 100.
        ik_solns = np.array(ik_solns)
        time_scale = timestep*np.arange(ik_solns.shape[0])

        nrows = 6
        ncols = 2
        fig, axs = plt.subplots(nrows, ncols)
        fig.set_size_inches(18.5, 10.5)

        joints_of_intrest = [
            # 0,1,2,3,4,5,6, # root 3D pos + 4D quat
            # 7,8,9, # abdomen joints

            10, 11, 12, 13, 14, 15,  # left_leg
            16, 17, 18, 19, 20, 21,  # right_leg

        ]
        joint_id2name = {}
        joint_id = 0
        root_dof = ['x','y','z','qw','qx','qy','qz']
        for joint_name in env.model.joint_names:
            if joint_name == 'root':
                for i in range(len(root_dof)):
                    joint_id2name[joint_id] = 'unactuated root_'+str(root_dof[i])
                    joint_id += 1
            else:
                joint_id2name[joint_id] = joint_name
                joint_id += 1

        plot_id = 0

        for joint_id in joints_of_intrest:

            # row major
            # row = plot_id // ncols
            # col = plot_id % ncols

            # col major
            row = plot_id % nrows
            col = plot_id // nrows

            axs[row, col].plot(
                time_scale,
                ik_solns[:, joint_id],
            )


            axs[row, col].set_title(joint_id2name[joint_id])
                        
            # axs[row,col].plot(timesteps, torques_of_joints_contact[:,plot_id],label=joint_name)

            axs[row, col].set_ylabel("joint angles (rads)")
            axs[row, col].set_xlabel("time (s)")
            # axs[row,col].legend(loc='upper right')
            axs[row, col].grid()
            plot_id += 1

        fig.suptitle('IK Output: ')
        fig.tight_layout()
        # plt.savefig('./evaluation_plots/ip_type2_'+taskname+'.jpg')
        plt.show()