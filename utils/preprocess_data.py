from operator import le
from statistics import mean
import c3d
import yaml
import scipy.io
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import misc_functions


# TODO: 
# 1. find why the COP is not generated in 
#    c3d and fix it, remove .mat dependency once done.
# 2. npz file naming from index to time sytems ?

def clip_max_delta_filter(data,min_val,max_val,delta_threshold):
    data = np.clip(data,min_val,max_val)
    
    data_deltas = data - np.concatenate([[0], data[:-1] ])
    data_mean = np.mean(data) #np.mean(COPS_l[:,1])
    print('data mean:',data_mean)
    data_filtered = []
    for val,delta in zip (data,data_deltas):
        if abs(delta) > delta_threshold or val == min_val or val == max_val :
            data_filtered.append(data_mean)
        # elif :
        #     data_filtered.append(data_mean)
        else:
            data_filtered.append(val)
    return data_filtered

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def generate_conf_file(dataFile_path,confFile_path):

    conf_data = {
                    'marker_name2id': {},
                    'forces_name2id':{},
                }
    
    with open(dataFile_path, 'rb') as handle:
        
        reader = c3d.Reader(handle)

        for id,m_name in enumerate(reader.point_labels):
             
            conf_data['marker_name2id'][m_name.replace(' ','')] = id

        for id,a_name in enumerate(reader.analog_labels):    
            conf_data['forces_name2id'][a_name.replace(' ','')] = id

    config_file = open(confFile_path,'w')
    yaml.dump(conf_data,config_file)

def generate_npz_file(
                        dataFile_path,
                        npzFile_path,
                        confFile_path,
                        roi_start = 0.,
                        roi_stop = None,
                        plot_results = False,
                        ):


    
    config_file = open(confFile_path,'r+')
    marker_conf = yaml.load(config_file, Loader=yaml.FullLoader)
    forces_name2id = marker_conf['forces_name2id']
    marker_name2id = marker_conf['marker_name2id']

    with open( dataFile_path , 'rb') as handle:
        reader = c3d.Reader(handle)
        
        # force plate parameters
        force_plate_origins = reader.get('FORCE_PLATFORM.ORIGIN').float_array
        print("Force plate origin:\n", force_plate_origins)
        rot_w_fp = np.array(

                        [
                            [-1.,0.,0.],
                            [0.,1.,0.],
                            [0.,0.,-1.]
                        ]
        )
        stance_fz_mag_threshold = 10
        
        # from insung
        right_plate_analog_id = 0
        left_plate_analog_id = 1

        # from meta data
        if force_plate_origins[0][0] > force_plate_origins[1][0]:
            right_plate_origin_id = 0
            left_plate_origin_id = 1
        else:
            right_plate_origin_id = 1
            left_plate_origin_id = 0

        roi_stop = reader.last_frame if roi_stop == None else roi_stop
    
        pbar = tqdm(total=roi_stop)
        pbar.set_description("Loading mocap data")   
        
        marker_positions = []
        grf_data = []
        cop_data = []

            
        for data in reader.read_frames():
            pbar.update(1)        
            
            # TODO: Do nan chks
            if data[0] > roi_start:

                marker_positions.append(data[1][:,0:3].tolist())
                
                # just the first analog reading in the frame
                grf_data.append(data[2][:,0].tolist())
                
                    
                rFz = data[2][ forces_name2id['Force.Fz'+str(right_plate_analog_id+1)] , 0]
                lFz = data[2][ forces_name2id['Force.Fz'+str(left_plate_analog_id+1)] , 0]

                rMx = data[2][ forces_name2id['Moment.Mx'+str(right_plate_analog_id+1)] , 0]
                rMy = data[2][ forces_name2id['Moment.My'+str(right_plate_analog_id+1)] , 0]
                
                lMx = data[2][ forces_name2id['Moment.Mx'+str(left_plate_analog_id+1)] , 0]
                lMy = data[2][ forces_name2id['Moment.My'+str(left_plate_analog_id+1)] , 0]

                # NOTE: GRF should be force applied by ground on the foot. Here the raw GRF's are negative
                # meaning, the Force plate's +Z axis is opposite to the world / mujoco co ordinate frame
                x_cop_r = -rMy / rFz if abs(rFz) >= stance_fz_mag_threshold else 0
                y_cop_r = rMx / rFz if abs(rFz) >= stance_fz_mag_threshold else -force_plate_origins[right_plate_origin_id][1]                                   
                
                x_cop_l = -lMy / lFz if abs(lFz) >= stance_fz_mag_threshold else 0.
                y_cop_l = lMx / lFz if abs(lFz) >= stance_fz_mag_threshold else -force_plate_origins[left_plate_origin_id][1]

                # rotate and translate to world frame
                # print(rot_w_fp,np.array([x_cop_r,y_cop_r,0]) )
                pr = rot_w_fp@np.array([x_cop_r,y_cop_r,0]) 
                # print(pr)
                # exit()
                pr = pr + force_plate_origins[right_plate_origin_id]
                
                pl = rot_w_fp@np.array([x_cop_l,y_cop_l,0]) 
                pl = pl + force_plate_origins[left_plate_origin_id] 

                if right_plate_origin_id > left_plate_origin_id: 
                    cop_data.append(np.concatenate([pl, pr]))
                else: 
                    cop_data.append(np.concatenate([pr, pl]))
            
            if data[0] > roi_stop:
                break


    
    # unit convertions

    # mm to m
    marker_positions = 0.001*np.array(marker_positions)
    
    # for forces, only moments should be converted from Nmm to Nm
    grf_data = np.array(grf_data)

    # convert moments from Nmm to Nm
    for key in forces_name2id.keys():
        if 'Moment' in key:
            moment_id = forces_name2id[key]
            grf_data[:,moment_id] = 0.001*grf_data[:,moment_id]

    # cop data from mm to m
    cop_data = 0.001*np.array(cop_data)
        
    print("Marker Pos. Traj. Shape:", marker_positions.shape)
    print("GRF Traj. Shape:", grf_data.shape)
    print("COP Traj. Shape:", cop_data.shape)

    # save the npz file
    np.savez_compressed(npzFile_path,marker_positions=marker_positions,grfs=grf_data,cops=cop_data)

    if plot_results:
        # plts for visualising
        fig, axs = plt.subplots(5,1)
        fig.suptitle("Calculated COP ")

        time = 0.01*np.arange(cop_data.shape[0])

    # COP plots
        axs[0].plot(time,cop_data[:,0],label='x_cop right',color='g')
        axs[0].plot(time,cop_data[:,3],label='x_cop left',color='r')

        axs[1].plot(time,cop_data[:,1],label='y_cop right',color='g')
        axs[1].plot(time,cop_data[:,4],label='y_cop left',color='r')

        # GRF Fz
        axs[2].plot(
                    time,
                    grf_data[:,forces_name2id['Force.Fz'+str(left_plate_analog_id+1)] ],
                    label='GRF (Fz) left',
                    alpha=0.6,
                    color='red',
                    linestyle='-.'
            ) 

        axs[2].plot(
                    time,
                    grf_data[:,forces_name2id['Force.Fz'+str(right_plate_analog_id+1)] ],
                    label='GRF (Fz) right',
                    alpha=0.6,
                    color='green',
                    linestyle='-.'
            )
        
        # GRF mx
        axs[3].plot(
                    time,
                    grf_data[:,forces_name2id['Moment.Mx'+str(left_plate_analog_id+1)] ],
                    label='GRF (Mx) left',
                    alpha=0.6,
                    color='red',
                    linestyle='-.'
            )

        axs[3].plot(
                    time,
                    grf_data[:,forces_name2id['Moment.Mx'+str(right_plate_analog_id+1)] ],
                    label='GRF (Mx) right',
                    alpha=0.6,
                    color='green',
                    linestyle='-.'
            )
        
        # GRF My
        axs[4].plot(
                    time,
                    grf_data[:,forces_name2id['Moment.My'+str(left_plate_analog_id+1)] ],
                    label='GRF (My) left',
                    alpha=0.6,
                    color='red',
                    linestyle='-.'
            )
        axs[4].plot(
                    time,
                    grf_data[:,forces_name2id['Moment.My'+str(right_plate_analog_id+1)] ],
                    label='GRF (My) right',
                    alpha=0.6,
                    color='green',
                    linestyle='-.'
            )

        for ax in axs:
            ax.legend()
            ax.grid()

        axs[1].set_ylim(-3,3)
        axs[0].set_ylim(-3,3)

        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--c3d_filepath',help='name of the c3d file',default='AB1_Session1_Right6_Left6',type=str)
    parser.add_argument('--roi_start',help='start index of the region of intrest',default=0,type=int)
    parser.add_argument('--roi_stop',help='stop index of the region of intrest',default=None,type=int)
    parser.add_argument('--static',help='whether given file is a static file',default=False, action='store_true')
    parser.add_argument('--plot',help='whether to plot the (some) cleaned results',default=False, action='store_true')

    args = parser.parse_args()  

    assets_path = './gym_hmm_ec/envs/assets/'
    

    c3d_removed_path = args.c3d_filepath.replace('.c3d','')
    print(c3d_removed_path)

    # input files
    mat_filepath =  c3d_removed_path.replace('c3ds','mats')+'.mat'

    # output files
    conf_filepath = c3d_removed_path.replace('c3ds','confs')+'.yaml'
    npz_filepath = c3d_removed_path.replace('c3ds','processed_data')\
                   +'_from_'+str(args.roi_start)+'_to_'+str(args.roi_stop)+'.npz'


    generate_conf_file(
                        dataFile_path=args.c3d_filepath,
                        confFile_path=conf_filepath
                    )
    
    generate_npz_file(
                        dataFile_path=args.c3d_filepath,
                        npzFile_path=npz_filepath,
                        confFile_path=conf_filepath,
                        roi_start=args.roi_start,
                        roi_stop =args.roi_stop,
                        plot_results = args.plot
                    )    
