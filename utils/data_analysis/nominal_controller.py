from ntpath import join
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import yaml
import argparse
from utils import misc_functions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ik_soln',help='path to ik soln file',default='AB1_Session1_Right6_Left6',type=str)
    parser.add_argument('--id_soln',help='path to id soln file',default='AB1_Session1_Right6_Left6',type=str)
    parser.add_argument('--plot_id',help='id selecting the req. plot',default=0,type=int)

    parser.add_argument('--export_solns',help='whether to export the ik solns',default=False, action='store_true')
    parser.add_argument('--render',help='whether to render while solving for ik',default=False, action='store_true')
    args = parser.parse_args()

    if os.path.isfile(args.ik_soln):
        kinematic_solns = np.load(args.ik_soln)
    else:
        print("Missing: IK solutions absent, either not computed or not saved.")
        exit()        

    if os.path.isfile(args.id_soln):
        id_solns = np.load(args.id_soln)['id_solns']
    else:
        print("Missing: ID solutions absent, either not computed or not saved.")
        exit()     
    
    # phase portrait
    if args.plot_id == 0:
        frame_rate = 100.
        timestep = 1. / frame_rate
        step = 0
        qvels  = []
        qposs = []
        
        
        for step,qpos in tqdm(enumerate(kinematic_solns['ik_solns'])):


            if step == 0:
                prev_qpos = qpos.copy()


            qvel = np.concatenate(       [
                    (qpos[:3]-prev_qpos[:3])/ timestep,
                    misc_functions.mj_quat2vel(
                        misc_functions.mj_quatdiff(prev_qpos[3:7], qpos[3:7]), timestep),
                    (qpos[7:]-prev_qpos[7:])/ timestep
                    ]
                ).ravel()
            qvels.append(qvel.tolist())
            qposs.append(qpos.tolist())
            prev_qpos = qpos.copy()

        # id_solns = np.array(id_solns)
        qvels = np.array(qvels)
        qposs = np.array(qpos)

        # time_scale = timestep*np.arange(step)

        nrows = 6
        ncols = 2
        fig,axs = plt.subplots(nrows,ncols)
        fig.set_size_inches(18.5, 10.5)

        joint_id = 0
        # for genarlized force
        joints_of_intrest = [ 
                                # 0,1,2,3,4,5, # root 6D
                                # 6,7,8, # abdomen joints        
                                9,10,11,12,13,14, # left_leg
                                15,16,17,18,19,20, # left_leg

                            ]
        joints_names = [ 
                        'left_leg/hip_x','left_leg/hip_y','left_leg/hip_z','left_leg/knee','left_leg/ankle_y','left_leg/ankle_x',
                        'right_leg/hip_x','right_leg/hip_y','right_leg/hip_z','right_leg/knee','right_leg/ankle_y','right_leg/ankle_x',

                        ]
        plot_id = 0
        for joint_id in joints_of_intrest:

            # row = plot_id // ncols
            # col = plot_id % ncols

            row = plot_id % nrows
            col = plot_id // nrows

            axs[row,col].plot(
                                kinematic_solns['ik_solns'][:,joint_id+1], 
                                qvels[:,joint_id],

                                )
            axs[row,col].set_title(joints_names[plot_id])
            axs[row,col].set_ylabel("dq")
            axs[row,col].set_xlabel("q")
            # axs[row,col].legend(loc='upper right')
            axs[row,col].grid()
            plot_id +=1 

        fig.suptitle('Phase Portriat of leg angles:')
        fig.tight_layout()

    # Mean Gait Cycle 
    elif args.plot_id == 1:
        
        frame_rate = 100.
        timestep = 1. / frame_rate
        step = 0
        qvels  = []
        qposs = []
        taus = []


        ####################################

        start_qpos = None
        qpos_residues = []

        for step,qpos in tqdm(enumerate(kinematic_solns['ik_solns'])):

            if step == 0:
                prev_qpos = qpos.copy()
                start_qpos = qpos.copy()
            
            else:
                qpos_residues.append(np.linalg.norm( (start_qpos - qpos) ) )


            qvel = np.concatenate(       [
                    (qpos[:3]-prev_qpos[:3])/ timestep,
                    misc_functions.mj_quat2vel(
                        misc_functions.mj_quatdiff(prev_qpos[3:7], qpos[3:7]), timestep),
                    (qpos[7:]-prev_qpos[7:])/ timestep
                    ]
                ).ravel()
            qvels.append(qvel.tolist())
            qposs.append(qpos.tolist())
            prev_qpos = qpos.copy()


        # id_solns = np.array(id_solns)
        qvels = np.array(qvels)
        qposs = np.array(qposs)        
        qpos_residues = np.array(qpos_residues)

        dt = 0.01
        T = 1.2
        n_cycles = int( dt*qposs.shape[0]/T )
        steps_per_cycle = int(T/dt)

        print(steps_per_cycle,n_cycles)
        joint_ids =  [
            # 0,1,2,3,4,5,6, # root 6D
            
            # 7,8,9, # abdomen joints

            10, 11, 12, 13, 14, 15,  # left_leg
            16, 17, 18, 19, 20, 21  # left_leg

        ]
        q_mean_cycle = np.zeros(( len(joint_ids),steps_per_cycle))
        dq_mean_cycle = np.zeros(( len(joint_ids),steps_per_cycle))

        for joint_id in joint_ids:
            for step in range(steps_per_cycle):

                for cycle in range(n_cycles):            
                    q_mean_cycle[joint_id - joint_ids[0], step] += qposs[cycle*steps_per_cycle + step,joint_id]

                    dq_mean_cycle[joint_id - joint_ids[0], step] += qvels[cycle*steps_per_cycle + step, joint_id-1 ]
                
                q_mean_cycle[joint_id - joint_ids[0], step] /= n_cycles
                dq_mean_cycle[joint_id - joint_ids[0], step] /= n_cycles


        q_diff = np.empty_like(qposs)
        dq_diff = np.empty_like(qvels)
        for joint_id in joint_ids:
            for step in range(steps_per_cycle):
                for cycle in range(n_cycles):
                    q_diff[cycle*steps_per_cycle + step,joint_id] = qposs[cycle*steps_per_cycle + step,joint_id] - q_mean_cycle[joint_id - joint_ids[0], step]
                    dq_diff[cycle*steps_per_cycle + step,joint_id-1] = qvels[cycle*steps_per_cycle + step, joint_id-1 ] - dq_mean_cycle[joint_id - joint_ids[0], step]

        # print(qposs.shape, qvels.shape)        
        # print(q_diff.shape, dq_diff.shape)

        np.save('data/q_diff.npy',q_diff)
        np.save('data/dq_diff.npy',dq_diff)

        nrows = 6
        ncols = 4
        fig, axs = plt.subplots(nrows, ncols)
        fig.set_size_inches(18.5, 10.5)        
        
        
        plot_id = 0
        for joint_id in joint_ids:

            row = plot_id % nrows
            col = plot_id // nrows

            # q_mean
            axs[row, col].plot(
                0.01*np.arange(q_mean_cycle.shape[1]),
                q_mean_cycle[joint_id- joint_ids[0],:],
            )
            axsT = axs[row, col].twinx()
            axsT.plot( 0.01*np.arange(qposs.shape[0]),
                        qposs[:,joint_id],
                        alpha=0.5,
                        color='red',
                        linestyle='-.'
                    ) 

            # dq_mean
            axs[row, col+2].plot(
                0.01*np.arange(dq_mean_cycle.shape[1]),
                dq_mean_cycle[joint_id- joint_ids[0],:],
            )
            axsT = axs[row, col+2].twinx()
            axsT.plot( 0.01*np.arange(qvels.shape[0]),
                        qvels[:,joint_id-1],
                        alpha=0.5,
                        color='red',
                        linestyle='-.'
                    )   
           
            axs[row, col].set_title("Joint ID: "+str(joint_id))
            axs[row, col].set_ylabel("ang pos (rad)")
            axs[row, col].set_xlabel("phase / time, t-nT (s)")
            axs[row, col].grid()

            axs[row, col+2].set_title("Joint ID: "+str(joint_id))
            axs[row, col+2].set_ylabel("ang vel (rad/s)")
            axs[row, col+2].set_xlabel("phase / time, t-nT (s)")
            axs[row, col+2].grid()


            plot_id += 1

        fig.suptitle('Mean Gait Cycle - States')
        fig.tight_layout()

        plt.grid()
        np.save("data/q_mean_cycle.npy",q_mean_cycle)
        np.save("data/dq_mean_cycle.npy",dq_mean_cycle)



    # Mean Torque Cycle 
    elif args.plot_id == 2:
            
        dt = 0.01
        T = 1.2
        n_cycles = int( dt*id_solns.shape[0]/T )
        steps_per_cycle = int(T/dt)

        joint_ids = [
            # 0,1,2,3,4,5, # root 6D
            # 6,7,8, # abdomen joints

            9, 10, 11, 12, 13, 14,  # left_leg
            15, 16, 17, 18, 19, 20,  # left_leg

        ]
        tau_mean_cycle = np.zeros(( len(joint_ids),steps_per_cycle))

        for joint_id in joint_ids:
            for step in range(steps_per_cycle):

                for cycle in range(n_cycles):            

                    tau_mean_cycle[joint_id - joint_ids[0], step] += id_solns[cycle*steps_per_cycle + step, joint_id ]
                
                tau_mean_cycle[joint_id - joint_ids[0], step] /= n_cycles    

        tau_mean_cycle = np.array(tau_mean_cycle)

        tau_diff = np.empty_like(id_solns)
        for joint_id in joint_ids:
            for step in range(steps_per_cycle):
                for cycle in range(n_cycles):
                    tau_diff[cycle*steps_per_cycle + step,joint_id] = id_solns[cycle*steps_per_cycle + step,joint_id] - tau_mean_cycle[joint_id - joint_ids[0], step]

        # print(qposs.shape, qvels.shape)        
        # print(q_diff.shape, dq_diff.shape)

        np.save('data/tau_diff.npy',tau_diff)

        
        nrows = 6
        ncols = 2
        fig, axs = plt.subplots(nrows, ncols)
        fig.set_size_inches(18.5, 10.5)        
        
        
        plot_id = 0
        for joint_id in joint_ids:

            row = plot_id % nrows
            col = plot_id // nrows

            # q_mean
            axs[row, col].plot(
                0.01*np.arange(tau_mean_cycle.shape[1]),
                tau_mean_cycle[joint_id - joint_ids[0],:],
            )
            axsT = axs[row, col].twinx()
            axsT.plot( 0.01*np.arange(
                        id_solns.shape[0]),
                        
                        id_solns[:,joint_id],
                        
                        alpha=0.5,
                        color='red',
                        linestyle='-.'
                    ) 

           
            axs[row, col].set_title("Joint ID: "+str(joint_id))
            axs[row, col].set_ylabel("torque (Nm)")
            axs[row, col].set_xlabel("phase / time, t-nT (s)")
            axs[row, col].grid()

            plot_id += 1

        fig.suptitle('Mean Gait Cycle - Torques')
        fig.tight_layout()

        plt.grid()
        np.save("data/tau_mean_cycle.npy",tau_mean_cycle)

        # plt.show()
        
    # phase variable
    elif args.plot_id == 3:
        
        frame_rate = 100.
        timestep = 1. / frame_rate
        step = 0
        qvels  = []
        qposs = []
        taus = []


        ####################################
        D_stride = 0.5

        yPelvis_yRl = kinematic_solns['pelvis_xpos'][:,1] - kinematic_solns['rfoot_xpos'][:,1]
        yPelvis_yLl = kinematic_solns['pelvis_xpos'][:,1] - kinematic_solns['lfoot_xpos'][:,1]

        yPelvis_ySt = [ ]
        for i in range(kinematic_solns['pelvis_xpos'].shape[0]):
            if  kinematic_solns['rfoot_xpos'][i,2] < kinematic_solns['lfoot_xpos'][i,2]:
                yPelvis_ySt.append( kinematic_solns['pelvis_xpos'][i,1] - kinematic_solns['rfoot_xpos'][i,1])
            else:
                yPelvis_ySt.append( kinematic_solns['pelvis_xpos'][i,1] - kinematic_solns['lfoot_xpos'][i,1])
        yPelvis_ySt = np.array(yPelvis_ySt)
        
        phase_var_Rl = yPelvis_yRl / D_stride # y_pelvis - D_Stride
        phase_var_Ll = yPelvis_yLl / D_stride # y_pelvis - D_Stride
        phase_var_St = yPelvis_ySt / D_stride # y_pelvis - D_Stride

        # plt.plot(np.arange(phase_var_Rl.shape[0]),phase_var_Rl, label='yPelvis-yRl')
        # plt.plot(np.arange(phase_var_Ll.shape[0]),phase_var_Ll, label='yPelvis-yLl')
        plt.plot(np.arange(phase_var_St.shape[0]),phase_var_St, label='yPelvis-ySt')

        plt.grid()
        plt.legend()
        plt.title("Phase Variable")
        
    # plt.savefig('./evaluation_plots/ip_type2_'+taskname+'.jpg')
    plt.show()
