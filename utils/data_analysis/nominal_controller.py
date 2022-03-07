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
        ik_solns = np.load(args.ik_soln)['ik_solns']
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
        for step,qpos in tqdm(enumerate(ik_solns)):

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
                                ik_solns[:,joint_id+1], 
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

    # Gait Cycle
    elif args.plot_id == 1:
        
        frame_rate = 100.
        timestep = 1. / frame_rate
        step = 0
        qvels  = []
        qposs = []
        for step,qpos in tqdm(enumerate(ik_solns)):

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
        qposs = np.array(qposs)        
        
        joint_id = 10
        sp = np.fft.fft(a=qposs[:,joint_id])
        freq = np.fft.fftfreq(qposs[:,joint_id].shape[-1])
        
        print(sp.shape,freq.shape)
        plt.plot(freq, sp.real, freq, sp.imag)
        plt.grid()
        plt.show()
        '''


        # time_scale = timestep*np.arange(step)

        nrows = 6
        ncols = 3
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
                                ik_solns[:,joint_id+1], 
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
        '''


    # plt.savefig('./evaluation_plots/ip_type2_'+taskname+'.jpg')
    plt.show()
