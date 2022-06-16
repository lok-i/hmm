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
                'model_name': args.model_filename if '.xml' not in args.model_filename else args.model_filename.replace('.xml','') ,
                'mocap':False,
                'observations':
                {
                    'current_model_state': None
                },
                'actions':
                {   'joint_torques':
                        {
                            'dim': 15,
                            'torque_max': 5
                        }                
                },
                'rewards':
                {
                    'zero_reward':None
                },
                'terminations':
                {
                    'indefinite':None
                }                
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
    qvels = []
    qaccs = []
    for qpos,grf,cop in tqdm(zip(ik_solns,grf_data,cop_data),total=ik_solns.shape[0]):

        # # COP points
        # for i in range(2):

        #     marker_name = 'm'+str(i)
        #     env.sim.data.set_mocap_pos(
        #         marker_name, cop[3*i:3*i+3] 
        #         )
        
        env.sim.data.qpos[:] = qpos
        if step == 0:
            prev_qpos = qpos.copy()
            prev_qvel = np.zeros(env.model.nv)


        body_name = 'right_leg/foot'
        body_id = env.sim.model.body_name2id(body_name)
        wrench_prtb = grf[0:6]
        frc_pos = env.sim.data.get_body_xpos(body_name)
        env.view_vector_arrows(
            vec=wrench_prtb[0:3],
            vec_point=frc_pos,
            vec_mag_max=500,
        )
        env.sim.data.xfrc_applied[body_id][:] = wrench_prtb

        body_name = 'left_leg/foot'
        body_id = env.sim.model.body_name2id(body_name)
        wrench_prtb = grf[6:12]
        frc_pos = env.sim.data.get_body_xpos(body_name)
        env.view_vector_arrows(
            vec=wrench_prtb[0:3],
            vec_point=frc_pos,
            vec_mag_max=500)
        env.sim.data.xfrc_applied[body_id][:] = wrench_prtb




        # qpos_0 = qpos_quat2rpy(qpos)
        # qpos_1 = qpos_quat2rpy(ik_solns[step-1,:])
        # qpos_2 = qpos_quat2rpy(ik_solns[step-2,:])
        # qpos_3 = qpos_quat2rpy(ik_solns[step-3,:])


        # env.sim.data.qvel[:] = ( 3*qpos_0 - 4*qpos_1 + qpos_2 ) / 2*timestep 
        # env.sim.data.qacc[:] = ( 2*qpos_0 - 5*qpos_1 + 4*qpos_2 - qpos_3 ) / (timestep**2) 

        # without FD to obtain qacc, finite differences
        env.sim.data.qvel[:] = np.concatenate([
            (qpos[:3]-prev_qpos[:3]) / timestep,
            misc_functions.mj_quat2vel(
                misc_functions.mj_quatdiff(prev_qpos[3:7], qpos[3:7]), timestep),
            (qpos[7:]-prev_qpos[7:]) / timestep
        ]
        ).ravel()

        # with FD to obtain qacc
        # obs, reward, done, info = env.step(action=np.zeros(shape=env.n_act_joints))
        # env.sim.step()
        functions.mj_forward(env.model, env.sim.data)
        
        qvels.append( env.sim.data.qvel.tolist())
        qaccs.append( env.sim.data.qacc.tolist())


        if env.env_params['render']:
            env.render()

        functions.mj_inverse(env.model, env.sim.data)
        
        # print("qfrc_inverse:",env.sim.data.qfrc_inverse)
        # print("qfrc_applied:",env.sim.data.qfrc_applied.shape)
        # print("qfrc_actuator:",env.sim.data.qfrc_actuator.shape)
        #print("xfrc_applied:",env.sim.data.xfrc_applied)

        
        # print(env.sim.data.qvel)
        id_solns.append(env.sim.data.qfrc_inverse.tolist())
        
        prev_qpos = qpos.copy()
        prev_qvel = env.sim.data.qvel.copy()

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
        qaccs =np.array(qaccs)
        qvels =np.array(qvels)

        time_scale = timestep*np.arange(id_solns.shape[0])

        joint_id2name = {}
        joint_id = 0
        root_dof = ['fx','fy','fz','taux','tauy','tauz'] # for generalised forces
        for joint_name in env.model.joint_names:
            if joint_name == 'root':
                for i in range(len(root_dof)):
                    joint_id2name[joint_id] = 'unactuated root_'+str(root_dof[i])
                    joint_id += 1
            else:
                joint_id2name[joint_id] = joint_name
                joint_id += 1

        
        total_plots_tbp = 0
        for joint_id,joint_name in joint_id2name.items():
            if 'leg' in joint_name or 'root' in joint_name:
                total_plots_tbp +=1


        ncols = 3 if 'humanoid' in args.model_filename else 4
        nrows = int(total_plots_tbp / ncols )
        fig, axs = plt.subplots(nrows, ncols)
        fig.set_size_inches(18.5, 10.5)


        plot_id = 0

        for joint_id,joint_name in joint_id2name.items():
            
            if 'leg' in joint_name or 'root' in joint_name:
                # row major
                # row = plot_id // ncols
                # col = plot_id % ncols

                # col major
                row = plot_id % nrows
                col = plot_id // nrows

                axs[row, col].plot(
                    time_scale,
                    id_solns[:, joint_id],
                    # label='id_solns'
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
                
                if 'left' in joint_name:
                    # left grf
                    axs_twin = axs[row, col].twinx()
                    axs_twin.plot(
                        time_scale,
                        grf_data[:,
                                        marker_conf['forces_name2id']['Force.Fz2']],
                        alpha=0.6,
                        color='red',
                        linestyle='-.'
                    )
                    axs_twin.set_ylabel('left_leg/Fz')
                
                elif 'right' in joint_name:
                    # right grf
                    axs_twin = axs[row, col].twinx()
                    axs_twin.plot(
                        time_scale,
                        grf_data[:,
                                        marker_conf['forces_name2id']['Force.Fz1']],
                        alpha=0.6,
                        color='red',
                        linestyle='-.'
                    )
                    axs_twin.set_ylabel('right_leg/Fz')
                
                # axs[row,col].plot(timesteps, torques_of_joints_contact[:,plot_id],label=joint_name)

                if col < 1 and row <3:
                    axs[row, col].set_ylabel("forces (N)")
                else:
                    axs[row, col].set_ylabel("torques (Nm)")
                axs[row, col].set_xlabel("time (s)")
                # axs[row,col].legend(loc='upper right')
                axs[row, col].grid()
                plot_id += 1

        fig.suptitle('ID Output: ')
        fig.tight_layout()
        plt.show()