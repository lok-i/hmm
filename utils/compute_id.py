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
    c3d_file_name = 'mocap_marker_data/c3ds/Trial_1'

    # environment config and setup
    env_conf = {
        'set_on_rack': False,
        'render': args.render,
        'model_name': args.model_filename,
        'mocap': False
    }
    # marker config
    marker_confpath = args.processed_filepath.split('processed_data/')[0]+'confs/' \
                        + args.processed_filepath.split('processed_data/')[-1].split('_from')[0]+'.yaml' 

    marker_config_file = open(marker_confpath, 'r+')
    marker_conf = yaml.load(marker_config_file, Loader=yaml.FullLoader)

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


    mocap_marker_data = np.load(args.processed_filepath)

    ik_soln_filpath =  args.processed_filepath.split('marker_data/processed_data/')[0]+'ik_solns/' \
                        + args.processed_filepath.split('marker_data/processed_data/')[-1] 

    if os.path.isfile(ik_soln_filpath):
        ik_solns = np.load(ik_soln_filpath)['ik_solns']
    else:
        print("Missing: IK solutions absent, either not computed or not saved.")
        exit()
    id_solns = []
    frame_rate = 100.
    timestep = 1. / frame_rate
    prev_qpos = np.zeros(env.sim.data.qpos.shape)

    # FORCE_PLATFORM.ORIGIN[0] = [-280.  -935.     2.5]
    # FORCE_PLATFORM.ORIGIN[1] = [ 275.  -935.     2.5]
    fp_origin = 0.001*np.array(
                                    [[-280.,  -935.,     2.5],
                                    [-275.,  -935.,     2.5]]
                                    )
    step = 0

    for qpos in tqdm(ik_solns):


        for i in range(2):

            marker_name = 'm'+str(i)

            
            env.sim.data.set_mocap_pos(
                marker_name, mocap_marker_data['cops'][step, 3*i:3*i+3] 
                # - fp_origin[i]
                )
            # print(i,mocap_marker_data['cops'][step,3*i:3*i+3])
        

        
        env.sim.data.qpos[:] = qpos
        if step == 0:
            prev_qpos = qpos.copy()

        env.sim.data.qvel[:] = np.concatenate([
            (qpos[:3]-prev_qpos[:3]) / timestep,
            misc_functions.mj_quat2vel(
                misc_functions.mj_quatdiff(prev_qpos[3:7], qpos[3:7]), timestep),
            (qpos[7:]-prev_qpos[7:]) / timestep
        ]
        ).ravel()

        obs, reward, done, info = env.step(
            action=np.zeros(shape=env.n_act_joints))

        body_name = 'right_leg/foot'
        body_id = env.sim.model.body_name2id(body_name)
        wrench_prtb = -1.*np.array([
            mocap_marker_data['grfs'][step,
                               marker_conf['forces_name2id']['Force.Fx1']],
            mocap_marker_data['grfs'][step,
                               marker_conf['forces_name2id']['Force.Fy1']],
            mocap_marker_data['grfs'][step,
                               marker_conf['forces_name2id']['Force.Fz1']],
            mocap_marker_data['grfs'][step,
                               marker_conf['forces_name2id']['Moment.Mx1']],
            mocap_marker_data['grfs'][step,
                               marker_conf['forces_name2id']['Moment.My1']],
            mocap_marker_data['grfs'][step,
                               marker_conf['forces_name2id']['Moment.Mz1']],

        ])  # in world frame

        frc_pos = env.sim.data.get_body_xpos(body_name)
        env.view_vector_arrows(
            vec=wrench_prtb[0:3],
            vec_point=frc_pos,
            vec_mag_max=500,
        )
        env.sim.data.xfrc_applied[body_id][:] = wrench_prtb

        body_name = 'left_leg/foot'
        body_id = env.sim.model.body_name2id(body_name)
        wrench_prtb = -1.*np.array([
            mocap_marker_data['grfs'][step,
                               marker_conf['forces_name2id']['Force.Fx2']],
            mocap_marker_data['grfs'][step,
                               marker_conf['forces_name2id']['Force.Fy2']],
            mocap_marker_data['grfs'][step,
                               marker_conf['forces_name2id']['Force.Fz2']],
            mocap_marker_data['grfs'][step,
                               marker_conf['forces_name2id']['Moment.Mx2']],
            mocap_marker_data['grfs'][step,
                               marker_conf['forces_name2id']['Moment.My2']],
            mocap_marker_data['grfs'][step,
                               marker_conf['forces_name2id']['Moment.Mz2']],

        ])  # in world frame

        frc_pos = env.sim.data.get_body_xpos(body_name)
        env.view_vector_arrows(
            vec=wrench_prtb[0:3],
            vec_point=frc_pos,
            vec_mag_max=500,
        )
        env.sim.data.xfrc_applied[body_id][:] = wrench_prtb

        functions.mj_inverse(env.model, env.sim.data)
        id_solns.append(env.sim.data.qfrc_inverse.tolist())
        prev_qpos = qpos.copy()

        # to rotate camera 360 around the model
        # env.viewer.cam.azimuth += 0.25 #180
        step += 1

    if args.export_solns:
        id_solns = np.array(id_solns)

        # nan check
        nan_chk = np.isnan(id_solns) 
        for i in range(nan_chk.shape[0]):
            for j in range(nan_chk.shape[1]):
                if nan_chk[i,j]:
                    print('nan value in ID soln @ ',i,j)


        print('ID Soln Shape', id_solns.shape)

        output_filepath =  args.processed_filepath.split('marker_data/processed_data/')[0]+'id_solns/' \
                        + args.processed_filepath.split('marker_data/processed_data/')[-1] 
        np.savez_compressed(output_filepath, id_solns=id_solns)
        print("ID Solution written to:", output_filepath)
    
    if args.plot_solns:
        id_solns = np.array(id_solns)
        time_scale = timestep*np.arange(step)

        nrows = 6
        ncols = 2
        fig, axs = plt.subplots(nrows, ncols)
        fig.set_size_inches(18.5, 10.5)
        # plottinf should be reordered
        joints_of_intrest = [
            # 0,1,2,3,4,5, # root 6D
            # 6,7,8, # abdomen joints

            9, 10, 11, 12, 13, 14,  # left_leg
            15, 16, 17, 18, 19, 20,  # left_leg

        ]

        joint_id2name = {}
        joint_id = 0
        for joint_name in env.model.joint_names:
            if joint_name == 'root':
                for i in range(6):
                    joint_id2name[joint_id] = 'root'
                    joint_id += 1
            else:
                joint_id2name[joint_id] = joint_name
                joint_id += 1

        plot_id = 0

        for joint_id in joints_of_intrest:

            # row = plot_id // ncols
            # col = plot_id % ncols

            row = plot_id % nrows
            col = plot_id // nrows

            axs[row, col].plot(
                time_scale,
                id_solns[:, joint_id],
            )
            axs[row, col].set_title(joint_id2name[joint_id])
            if col == 0:
                # left grf
                axs_twin = axs[row, col].twinx()
                axs_twin.plot(
                    time_scale,
                    mocap_marker_data['grfs'][:,
                                       marker_conf['forces_name2id']['Force.Fz2']],
                    alpha=0.6,
                    color='red',
                    linestyle='-.'
                )
                axs_twin.set_ylabel('left_leg/Fz')
            else:
                # right grf
                axs_twin = axs[row, col].twinx()
                axs_twin.plot(
                    time_scale,
                    mocap_marker_data['grfs'][:,
                                       marker_conf['forces_name2id']['Force.Fz1']],
                    alpha=0.6,
                    color='red',
                    linestyle='-.'
                )
                axs_twin.set_ylabel('right_leg/Fz')

            # axs[row,col].plot(timesteps, torques_of_joints_contact[:,plot_id],label=joint_name)

            axs[row, col].set_ylabel("torques (Nm)")
            axs[row, col].set_xlabel("time (s)")
            # axs[row,col].legend(loc='upper right')
            axs[row, col].grid()
            plot_id += 1

        fig.suptitle('ID Output: ')
        fig.tight_layout()
        # plt.savefig('./evaluation_plots/ip_type2_'+taskname+'.jpg')
        plt.show()
