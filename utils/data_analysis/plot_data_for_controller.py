from ntpath import join
import statistics
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from utils import misc_functions

def resize_to_m_window(data,m_in_past):
    
    n_data = []
    print("old",m_in_past,data.shape)

    stop =   data.shape[0] - m_in_past            
    
    if data.ndim != 1:
        for i in range(0,stop):
            
            n_data.append( data[i:i+m_in_past,:].flatten())
    else:
        for i in range(0,stop):
            n_data.append( data[i:i+m_in_past].flatten())        
    
    n_data = np.array(n_data)
    print("new",n_data.shape) 
    return n_data

def sample_once_m(data,m):
    print("old",data.shape)            
    n_data = []
    if data.ndim != 1:
        for i in range(m,data.shape[0]):
            n_data.append( data[i-1,:])
    else:
        for i in range(m,data.shape[0]):
            n_data.append( data[i-1])        
    n_data = np.array(n_data)
    if n_data.ndim ==1:
        n_data = np.atleast_2d(n_data).T
    print("new",n_data.shape) 
    return n_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ik_soln',help='path to ik soln file',default='AB1_Session1_Right6_Left6',type=str)
    parser.add_argument('--id_soln',help='path to id soln file',default='AB1_Session1_Right6_Left6',type=str)
    parser.add_argument('--plot_id',help='id selecting the req. plot',default=0,type=int)
    parser.add_argument('--subplot_id',help='id selecting the req. sub plot',default=0,type=int)
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
        
        
        for step,qpos in enumerate(kinematic_solns['ik_solns']):


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

        for step,qpos in enumerate(kinematic_solns['ik_solns']):

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



        tau_diff = np.zeros_like(id_solns)
        print('spc:',steps_per_cycle,'nc:', n_cycles )
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

    # R**2 Test, only current state
    elif args.plot_id == 4:
        q_diff = np.load('./data/q_diff.npy')
        
        # convert quat to rpy in qpos
        q_diff = np.array([ base_pos.tolist() + misc_functions.quat2euler(base_quat).tolist() + joint_pos.tolist() \
                    for base_pos, base_quat, joint_pos in 
                    zip( q_diff[:,0:3], q_diff[:,3:7], q_diff[:,7:] ) ] )        

        dq_diff = np.load('./data/dq_diff.npy')
        tau_diff = np.load('./data/tau_diff.npy')
        
        
        tau_and_state_diff = [ q.tolist() + dq.tolist() + tau.tolist() for q,dq,tau in zip(q_diff,dq_diff,tau_diff)] 
        tau_and_state_diff = np.array(tau_and_state_diff)
        torque_rows = [q_diff.shape[1] + dq_diff.shape[1], q_diff.shape[1]+ dq_diff.shape[1] + tau_diff.shape[1] ]
        state_cols = [0,q_diff.shape[1] + dq_diff.shape[1]]

        R = np.corrcoef(x=tau_and_state_diff,rowvar=False)[torque_rows[0]:torque_rows[1],state_cols[0]:state_cols[1] ]
        print( "Corr. Coef matrix's shape:", R.shape )
        
        nan_chk = np.isnan(R) 
        for i in range(nan_chk.shape[0]):
            for j in range(nan_chk.shape[1]):
                if nan_chk[i,j]:
                    R[i,j] = 0.
                    # print(i,j)


        fig,ax = plt.subplots(1,1)

        # print(R[25,25])
        
        im = ax.imshow(R**2)
        fig.colorbar(im, ax=ax)

        tau_labels = [
                        'base_x','base_y','base_z',
                        'base_ro','base_pi','base_yw',

                        'abdmn_x','abdmn_y','abdmn_z',
                        'lhip_x','lhip_z','lhip_y', 'lknee','lankle_y','lankle_x',
                        'rhip_x','rhip_z','rhip_y', 'rknee','rankle_y','rankle_x',
                        ''
        ]
        
        
        
        ax.set_yticks(np.arange(-0.5, R.shape[0]))
        ax.set_yticklabels(tau_labels,verticalalignment="top")
        ax.set_ylabel("Generalised Forces")

        state_labels = [
                        
                        'base_x','base_y','base_z',
                        'base_ro','base_pi','base_yw',
                        'abdmn_x','abdmn_y','abdmn_z',
                        'lhip_x','lhip_z','lhip_y', 'lknee','lankle_y','lankle_x',
                        'rhip_x','rhip_z','rhip_y', 'rknee','rankle_y','rankle_x',

                        'v_base_x','v_base_y','v_base_z',
                        'v_base_ro','v_base_pi','v_base_yw',
                        'v_abdmn_x','v_abdmn_y','v_abdmn_z',
                        'v_lhip_x','v_lhip_z','v_lhip_y', 'v_lknee','v_lankle_y','v_lankle_x',
                        'v_rhip_x','v_rhip_z','v_rhip_y', 'v_rknee','v_rankle_y','v_rankle_x',
                        
                        ''
                        ]

        ax.set_xticks(np.arange(-0.5, R.shape[1]))
        ax.set_xticklabels(state_labels,horizontalalignment="left",rotation=90)
        ax.set_xlabel("Generalised States")
        ax.set_title('R^2 test over a single trial, current action vs state')
        ax.grid()        
        plt.show()        
        pass        
    # plt.savefig('./evaluation_plots/ip_type2_'+taskname+'.jpg')

    # R**2 Test for m-length state window
    elif args.plot_id == 5:

        q_diff = np.load('./data/q_diff.npy')
        dq_diff = np.load('./data/dq_diff.npy')
        tau_diff = np.load('./data/tau_diff.npy')
       
        # convert quat to rpy in qpos
        q_diff = np.array([ base_pos.tolist() + misc_functions.quat2euler(base_quat).tolist() + joint_pos.tolist() \
                    for base_pos, base_quat, joint_pos in 
                    zip( q_diff[:,0:3], q_diff[:,3:7], q_diff[:,7:] ) ] )        


        m_in_past = 20
        torque_start = 9 # from front side
        torque_stop = 2 # from back side
        dt = round(1./100,2)


        state_labels = [
                        
                        'base_x','base_y','base_z',
                        'base_ro','base_pi','base_yw',
                        'abdmn_x','abdmn_y','abdmn_z',
                        'lhip_x','lhip_z','lhip_y', 'lknee','lankle_y','lankle_x',
                        'rhip_x','rhip_z','rhip_y', 'rknee','rankle_y','rankle_x',

                        'v_base_x','v_base_y','v_base_z',
                        'v_base_ro','v_base_pi','v_base_yw',
                        'v_abdmn_x','v_abdmn_y','v_abdmn_z',
                        'v_lhip_x','v_lhip_z','v_lhip_y', 'v_lknee','v_lankle_y','v_lankle_x',
                        'v_rhip_x','v_rhip_z','v_rhip_y', 'v_rknee','v_rankle_y','v_rankle_x',
                        
                        ]

        states_to_ignore = [
                        'base_x','base_y','base_z',
                        'base_ro','base_pi','base_yw',
                        # 'abdmn_x','abdmn_y','abdmn_z',
                        # 'lhip_x','lhip_z','lhip_y', 'lknee','lankle_y','lankle_x',
                        # 'rhip_x','rhip_z','rhip_y', 'rknee','rankle_y','rankle_x',

                        'v_base_x','v_base_y','v_base_z',
                        'v_base_ro','v_base_pi','v_base_yw',
                        'v_abdmn_x','v_abdmn_y','v_abdmn_z',
                        # 'v_lhip_x','v_lhip_z','v_lhip_y', 'v_lknee','v_lankle_y','v_lankle_x',
                        # 'v_rhip_x','v_rhip_z','v_rhip_y', 'v_rknee','v_rankle_y','v_rankle_x',

                            ]
        
        states_id_to_keep = [state_labels.index(state) for state in state_labels if state not in states_to_ignore ]

        states_q_id = [q_id for q_id in states_id_to_keep if q_id < q_diff.shape[1]]
        states_dq_id = [dq_id -  q_diff.shape[1] for dq_id in states_id_to_keep if dq_id < q_diff.shape[1]+dq_diff.shape[1] and dq_id >= q_diff.shape[1]]

        q_diff = q_diff[:,states_q_id]
        dq_diff = dq_diff[:,states_dq_id]
        
        
        q_diff = resize_to_m_window(q_diff,m_in_past)
        dq_diff = resize_to_m_window(dq_diff,m_in_past)
        tau_diff = sample_once_m(tau_diff,m_in_past)

    
        # print(q_diff.shape,dq_diff.shape,tau_diff.shape)

        tau_and_state_diff = [ np.concatenate([q,dq,tau]) for q,dq,tau in zip(q_diff,dq_diff,tau_diff)] 
        

        tau_and_state_diff = np.array(tau_and_state_diff)

        


        torque_rows = [
                        q_diff.shape[1] + dq_diff.shape[1] + torque_start, 
                        q_diff.shape[1]+ dq_diff.shape[1] + tau_diff.shape[1] - torque_stop 
                        ]
        
        state_cols = [0,q_diff.shape[1] + dq_diff.shape[1]]

        R = np.corrcoef(x=tau_and_state_diff,rowvar=False)[torque_rows[0]:torque_rows[1],state_cols[0]:state_cols[1] ]
        
        print( "Corr. Coef matrix's shape:", R.shape )
        
        nan_chk = np.isnan(R) 
        for i in range(nan_chk.shape[0]):
            for j in range(nan_chk.shape[1]):
                if nan_chk[i,j]:
                    R[i,j] = 0.
                    # print(i,j)


        fig,ax = plt.subplots(1,1)
        
        im = ax.imshow(R**2,
                        aspect='auto',
                        vmin=0, vmax=1,
                        # cmap='seismic'
                        )
        fig.colorbar(im, ax=ax)

        tau_labels = [
                        'base_x','base_y','base_z',
                        'base_ro','base_pi','base_yw',

                        'abdmn_x','abdmn_y','abdmn_z',
                        'lhip_x','lhip_z','lhip_y', 'lknee','lankle_y','lankle_x',
                        'rhip_x','rhip_z','rhip_y', 'rknee','rankle_y','rankle_x',
                        
                        ]
        
        ax.set_yticks(np.arange(-0.5, R.shape[0]))
        ax.set_yticklabels(tau_labels[torque_start:tau_diff.shape[1] - torque_stop] +[''],verticalalignment="top")
        ax.set_ylabel("Generalised Forces")
        state_vec_len = int((q_diff.shape[1] +dq_diff.shape[1])/m_in_past )        
        
        k = 0
        for i in np.arange(0, R.shape[1]):
            if i % state_vec_len == 0:
                ax.axvline(x=i,color='red',linestyle='--',linewidth=0.5)
                ax.annotate(
                            text='t:'+str( (k - R.shape[1]/state_vec_len)*dt ),
                            xy=(i,-0.5),
                            # color='white',
                            rotation=45,
                            # fontsize= 5
                            
                            
                        )
                k+=1
        
        if R.shape[1] <= 100:
            state_lables_to_keep = [state_labels[state_id] for state_id in states_id_to_keep ]

            state_labels_to_keep = state_lables_to_keep*m_in_past + ['']

            ax.set_xticks(np.arange(-0.5, R.shape[1]))
            ax.set_xticklabels(state_labels_to_keep,horizontalalignment="left",rotation=90)
        
        ax.set_xlabel("Generalised States")
        # ax.set_title('R^2 test', fontsize=5)
        ax.grid()        

    # R**2 Test for m-length state window, pair wise
    elif args.plot_id == 6:

        q_diff = np.load('./data/q_diff.npy')
        dq_diff = np.load('./data/dq_diff.npy')
        tau_diff = np.load('./data/tau_diff.npy')
       
        # convert quat to rpy in qpos
        q_diff = np.array([ base_pos.tolist() + misc_functions.quat2euler(base_quat).tolist() + joint_pos.tolist() \
                    for base_pos, base_quat, joint_pos in 
                    zip( q_diff[:,0:3], q_diff[:,3:7], q_diff[:,7:] ) ] )        


        m_in_past = 240
        dt = round(1./100,2)

        tau_labels = [
                        'base_x','base_y','base_z',
                        'base_ro','base_pi','base_yw',

                        'abdmn_x','abdmn_y','abdmn_z',
                        'lhip_x','lhip_z','lhip_y', 'lknee','lankle_y','lankle_x',
                        'rhip_x','rhip_z','rhip_y', 'rknee','rankle_y','rankle_x',
                        
                        ]

        state_labels = [
                        
                        'base_x','base_y','base_z',
                        'base_ro','base_pi','base_yw',
                        'abdmn_x','abdmn_y','abdmn_z',
                        'lhip_x','lhip_z','lhip_y', 'lknee','lankle_y','lankle_x',
                        'rhip_x','rhip_z','rhip_y', 'rknee','rankle_y','rankle_x',

                        'v_base_x','v_base_y','v_base_z',
                        'v_base_ro','v_base_pi','v_base_yw',
                        'v_abdmn_x','v_abdmn_y','v_abdmn_z',
                        'v_lhip_x','v_lhip_z','v_lhip_y', 'v_lknee','v_lankle_y','v_lankle_x',
                        'v_rhip_x','v_rhip_z','v_rhip_y', 'v_rknee','v_rankle_y','v_rankle_x',
                        
                        ]
        state_action_pair = ['lknee','lknee']
        


        if state_labels.index(state_action_pair[0]) < q_diff.shape[1]:
            state = q_diff[:, state_labels.index(state_action_pair[0]) ]
        else:
            state = dq_diff[:, state_labels.index(state_action_pair[0])- q_diff.shape[1] ]
        
        tau_diff = tau_diff[:, tau_labels.index(state_action_pair[1])  ]
        
        state = resize_to_m_window(state,m_in_past)
        # q_diff = resize_to_m_window(q_diff,m_in_past)
        # dq_diff = resize_to_m_window(dq_diff,m_in_past)
        tau_diff = sample_once_m(tau_diff,m_in_past)


        print(state.shape,tau_diff.shape)

        tau_and_state_diff = [ np.concatenate([x,tau]) for x,tau in zip(state,tau_diff)] 
        

        tau_and_state_diff = np.array(tau_and_state_diff)


        R = np.corrcoef(x=tau_and_state_diff,rowvar=False) #[torque_rows[0]:torque_rows[1],state_cols[0]:state_cols[1] ]
        
        print( "Corr. Coef matrix's shape:", R.shape )
        nan_chk = np.isnan(R) 
        for i in range(nan_chk.shape[0]):
            for j in range(nan_chk.shape[1]):
                if nan_chk[i,j]:
                    R[i,j] = 0.
                    # print(i,j)

        xs = dt*np.arange(-m_in_past,0)
        ys = R[-1,:R.shape[1]-1]**2 # R^2
        
        # print( R[-1,-1]**2 )
        
        fig,ax = plt.subplots(1,1)
        
        ax.plot(
                        xs,
                        ys,
                    )


        
        # ax.set_xticks(xs)
        # ax.set_xticklabels(xs)
        ax.set_ylabel('R^2 with torque at '+state_action_pair[1])
        ax.set_xlabel(state_action_pair[0]+' (along time, right most is current time)')
        # ax.set_title('R^2 test', fontsize=5)
        ax.grid()        


    # plt.savefig('./evaluation_plots/ip_type2_'+taskname+'.jpg')
    plt.show()

