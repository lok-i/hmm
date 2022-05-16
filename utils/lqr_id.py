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
from control.matlab import lqr



def f(env,qpos,qvel,u,grf):
    env.sim.data.qpos[:] = qpos 
    env.sim.data.qvel[:] = qvel 
    env.sim.data.ctrl[:] = u

    # body_name = 'right_leg/foot'
    # body_id = env.sim.model.body_name2id(body_name)
    # wrench_prtb = grf[0:6]
    # frc_pos = env.sim.data.get_body_xpos(body_name)
    # env.view_vector_arrows(
    #     vec=wrench_prtb[0:3],
    #     vec_point=frc_pos,
    #     vec_mag_max=500,
    # )
    # env.sim.data.xfrc_applied[body_id][:] = wrench_prtb

    # body_name = 'left_leg/foot'
    # body_id = env.sim.model.body_name2id(body_name)
    # wrench_prtb = grf[6:12]
    # frc_pos = env.sim.data.get_body_xpos(body_name)
    # env.view_vector_arrows(
    #     vec=wrench_prtb[0:3],
    #     vec_point=frc_pos,
    #     vec_mag_max=500)
    # env.sim.data.xfrc_applied[body_id][:] = wrench_prtb

    # print("Bfore:",env.sim.data.qacc)
    functions.mj_forward(env.model, env.sim.data)
    # print("After:",env.sim.data.qacc)
    # exit()
    #functions.mj_fullM(env.model, env.sim.data)
    return np.concatenate([env.sim.data.qvel,env.sim.data.qacc])


def qpos_quat2rpy(qpos):
    return np.concatenate([
            qpos[:3],
            misc_functions.quat2euler(qpos[3:7]),
            qpos[7:]    
                    ]).ravel()

def qpos_rpy2quat(qpos):
    return np.concatenate([
            qpos[:3],
            misc_functions.euler2quat(qpos[3],qpos[4],qpos[5]),
            qpos[6:]    
                    ]).ravel()


    
if __name__ == '__main__':    
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--processed_filepath',help='name of the preprocessed mocap file',default='AB1_Session1_Right6_Left6',type=str)
    parser.add_argument('--export_solns', help='whether to export the id solns',
                        default=False, action='store_true')
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
            env.viewer.cam.lookat[i] = cam_pos[i]
        env.viewer.cam.elevation = -15
        env.viewer.cam.azimuth = 180


    mocap_data = np.load(args.processed_filepath)

    ik_soln_filpath =  args.processed_filepath.split('marker_data/processed_data/')[0]+'ik_solns/' \
                        + args.processed_filepath.split('marker_data/processed_data/')[-1] 
    id_soln_filpath =  args.processed_filepath.split('marker_data/processed_data/')[0]+'id_solns/' \
                        + args.processed_filepath.split('marker_data/processed_data/')[-1] 

    ik_solns = np.load(ik_soln_filpath)['ik_solns']
    id_solns = np.load(id_soln_filpath)['id_solns']

    frame_rate = mocap_data['frame_rate']
    prev_qpos = np.zeros(env.sim.data.qpos.shape)

    timestep = env.model.opt.timestep if env.model.opt.timestep < (1. / frame_rate) else (1. / frame_rate)

    grf_data = mocap_data['grfs'] 
    cop_data = mocap_data['cops'] 
    print("After:",ik_solns.shape, grf_data.shape,cop_data.shape)
    print("env timestep:",timestep)


    step = 0
    qvels = []
    qaccs = []
    id_solns_lqr = []

    epsilon = 1e-4
    u0 = np.zeros(env.sim.data.ctrl.shape)

    for qpos,grf,cop in tqdm(zip(ik_solns,grf_data,cop_data),total=ik_solns.shape[0]):
        
        
        qvel = np.concatenate([
            (qpos[:3]-prev_qpos[:3]) / timestep,
            misc_functions.mj_quat2vel(
                misc_functions.mj_quatdiff(prev_qpos[3:7], qpos[3:7]), timestep),
            (qpos[7:]-prev_qpos[7:]) / timestep
        ]
        ).ravel()
        qpos_rpy = qpos_quat2rpy(qpos)
        state = np.concatenate([qpos_rpy,qvel]).ravel()


        f0 = f(env,qpos,qvel,u0,grf)

        A = np.zeros(shape=(state.shape[0],state.shape[0]))
        
        for j in range(A.shape[1]):
            
            state_epsilon = np.zeros(state.shape[0])
            state_epsilon[j] = epsilon
            
            f_perturb = f(
                            env,
                            qpos_rpy2quat(state_epsilon[0:qpos_rpy.shape[0]]),
                            state_epsilon[qpos_rpy.shape[0]:qpos_rpy.shape[0]+qvel.shape[0]],
                            u0,
                            grf
                        )
            # for i in range(A.shape[0]):
            #     A[i][j] = ( f_perturb[i] - f0[i] ) / epsilon 
            A[:,j] = ( f_perturb - f0 ) / epsilon
        
        
        B = np.zeros(shape=(state.shape[0],env.sim.data.ctrl.shape[0]))
        
        for j in range(B.shape[1]):
            
            u_epsilon = np.zeros(u0.shape[0])
            u_epsilon[j] = epsilon
            f_perturb = f(
                            env,
                            qpos,
                            qvel,
                            u_epsilon,
                            grf

                        )

            B[:,j] = ( f_perturb - f0 ) / epsilon 

        
        Q = np.eye(N=state.shape[0])
        R = np.eye(N=u0.shape[0])
        K,S,E = lqr(A,B,Q,R)
        
        u = -np.dot(K,state)
        u = np.asarray(u).ravel()
        # print()

        # exit()
        id_solns_lqr.append(u)
        if env.env_params['render']:
            env.render()
        if step > 500:
            break
        step +=1

    time_scale = timestep*np.arange(id_solns.shape[0])
    id_solns_lqr = np.array(id_solns_lqr)
    nrows = 6
    ncols = 3
    fig, axs = plt.subplots(nrows, ncols)
    fig.set_size_inches(18.5, 10.5)
    # plottinf should be reordered
    joints_of_intrest = [
        0,1,2,3,4,5, # root 6D
        # 6,7,8, # abdomen joints

        9, 10, 11, 12, 13, 14,  # left_leg
        15, 16, 17, 18, 19, 20,  # right_leg

    ]

    joint_id2name = {}
    joint_id = 0
    root_dof = ['x','y','z','roll','pitch','yaw']
    for joint_name in env.model.joint_names:
        if joint_name == 'root':
            for i in range(6):
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

        # axs[row, col].plot(
        #     time_scale,
        #     id_solns[:, joint_id],
        #     label='id_muj'
        # )
        if joint_id >6:
            axs[row, col].plot(
                time_scale[0:id_solns_lqr.shape[0]],
                id_solns_lqr[:, joint_id-6],
                label='id_lqr'
            )


        # axs[row, col].plot(
        #     time_scale,
        #     qaccs[:, joint_id],
        #     label='qaccs'
        # )

        # axs[row, col].plot(
        #     time_scale,
        #     qvels[:, joint_id],
        #     label='qvels'

        # )

        axs[row, col].set_title(joint_id2name[joint_id])

        
        # axs[row,col].plot(timesteps, torques_of_joints_contact[:,plot_id],label=joint_name)

        if col < 1 and row <3:
            axs[row, col].set_ylabel("forces (N)")
        else:
            axs[row, col].set_ylabel("torques (Nm)")
        axs[row, col].set_xlabel("time (s)")
        axs[row,col].legend(loc='upper right')
        axs[row, col].grid()
        plot_id += 1

    fig.suptitle('ID Output: ')
    fig.tight_layout()
    plt.show()


    # fig,ax = plt.subplots(1,2,gridspec_kw={'width_ratios': [2.8, 1]})
    # im1 = ax[0].imshow( A,cmap='seismic')
    # im2 = ax[1].imshow( B,cmap='seismic')
    # plt.show()
    exit()