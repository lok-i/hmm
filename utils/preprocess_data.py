from statistics import mean
import c3d
import yaml
import scipy.io
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt



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
                        matFile_path,
                        npzFile_path,
                        confFile_path,
                        load_mat,
                        roi_start = 0.,
                        roi_stop = None,
                        ):

    if load_mat:
        mat_data = scipy.io.loadmat(matFile_path)
    
    config_file = open(confFile_path,'r+')
    marker_conf = yaml.load(config_file, Loader=yaml.FullLoader)
    forces_name2id = marker_conf['forces_name2id']
    marker_name2id = marker_conf['marker_name2id']

    with open( dataFile_path , 'rb') as handle:
        reader = c3d.Reader(handle)
        
        force_plate_origins = reader.get('FORCE_PLATFORM.ORIGIN').float_array
        print("Force plate origin:\n", force_plate_origins)
        roi_stop = reader.last_frame if roi_stop == None else roi_stop
    
        pbar = tqdm(total=roi_stop)
        pbar.set_description("Loading mocap data")   
        
        marker_positions = []
        grf_data = []
        cop_data = []

        # TODO: now walking along + Y axis, try generalising
        right_leg_plate_id = 0 
        left_leg_plate_id = 0 
        if force_plate_origins[0][0] > force_plate_origins[1][0]:
            right_leg_plate_id = 0
            left_leg_plate_id = 1
        else:
            right_leg_plate_id = 1
            left_leg_plate_id = 0            
        
        
        COPS_r = []
        COPS_l = []

        for data in reader.read_frames():
            pbar.update(1)        
            
            # TODO: Do nan chks
            if data[0] > roi_start: 
                marker_positions.append(data[1][:,0:3].tolist())
                
                # just the first analog reading in the frame
                grf_data.append(data[2][:,0].tolist())
                
                if load_mat:
                    
                    rFz = data[2][ forces_name2id['Force.Fz'+str(right_leg_plate_id+1)] , 0]
                    lFz = data[2][ forces_name2id['Force.Fz'+str(left_leg_plate_id+1)] , 0]

                    rMx = data[2][ forces_name2id['Moment.Mx'+str(right_leg_plate_id+1)] , 0]
                    rMy = data[2][ forces_name2id['Moment.My'+str(right_leg_plate_id+1)] , 0]
                    
                    lMx = data[2][ forces_name2id['Moment.Mx'+str(left_leg_plate_id+1)] , 0]
                    lMy = data[2][ forces_name2id['Moment.My'+str(left_leg_plate_id+1)] , 0]


                    x_cop_r = (rMy / rFz if rFz != 0 else 0) + force_plate_origins[right_leg_plate_id][0]
                    y_cop_r = (-rMx / rFz if rFz != 0 else 0) + force_plate_origins[right_leg_plate_id][1]
                    
                    x_cop_l = (lMy / lFz if lFz != 0 else 0) + force_plate_origins[left_leg_plate_id][0]
                    y_cop_l = (-lMx / lFz if lFz != 0 else 0) + force_plate_origins[left_leg_plate_id][1]


                    # x_cop_r = (-rMy / rFz if rFz != 0 else 0) + force_plate_origins[right_leg_plate_id][0]
                    # y_cop_r = (rMx / rFz if rFz != 0 else 0) + force_plate_origins[right_leg_plate_id][1]
                    
                    # x_cop_l = (-lMy / lFz if lFz != 0 else 0) + force_plate_origins[left_leg_plate_id][0]
                    # y_cop_l = (lMx / lFz if lFz != 0 else 0) + force_plate_origins[left_leg_plate_id][1]

                    # read the cop's
                    p1 = [x_cop_r,y_cop_r,0] #mat_data['forceStruct']['p1'][0][0][data[0]].tolist()
                    p2 = [x_cop_l,y_cop_l,0] #mat_data['forceStruct']['p2'][0][0][data[0]].tolist()
                    p3 = mat_data['forceStruct']['p3'][0][0][data[0]].tolist()

                    cop_data.append(p1+p2+p3)

            if data[0] > roi_stop:
                break


    if load_mat:
        cop_data = 0.001*np.array(cop_data)

    # mm to m
    marker_positions = 0.001*np.array(marker_positions)
    
    # for forces, only moments should be converted from Nmm to Nm
    grf_data = np.array(grf_data)

    # COPS_l = 0.001*np.array(COPS_l) # mm to m
    # COPS_r = 0.001*np.array(COPS_r) # mm to m

    time = 0.01*np.arange(cop_data.shape[0])
    # COPS_l[:,0] = clip_max_delta_filter(data=COPS_l[:,0],min_val=-1.5,max_val=0.,delta_threshold=0.5)
    # COPS_l[:,1] = clip_max_delta_filter(data=COPS_l[:,1],min_val=-3,max_val=-1.5,delta_threshold=0.02)
    
    fig, axs = plt.subplots(3,1)
    
    
    cop_data[:,3] = np.clip( cop_data[:,3],-3,3)
    cop_data[:,4] = np.clip( cop_data[:,4],-3,3)
    
    axs[0].plot(time,cop_data[:,3],label='x_cop')
    axs[1].plot(time,cop_data[:,4],label='y_cop')

    axs[0].grid()
    axs[1].grid()
    axs[0].legend()
    axs[1].legend()
    axs[2].grid()
    axs[2].plot(
                time,
                grf_data[:,forces_name2id['Force.Fz'+str(left_leg_plate_id+1)] ],
                label='GRF (Fz)',
                alpha=0.6,
                color='red',
                linestyle='-.'
        )  
    axs[2].legend()

    axs[1].set_ylim(-3,3)
    axs[0].set_ylim(-3,3)

    plt.show()



    '''
    plt.plot( 
                # time,
                COPS_l[:,0],
                COPS_l[:,1],
                label='left',
                alpha=0.5
            )

    ellipse_a = np.std(COPS_l[:,0]) 
    ellipse_b = np.std(COPS_l[:,1]) 
    
    centre = [np.mean(COPS_l[:,0]),np.mean(COPS_l[:,1])]
    
    # print(centre, ellipse_a, ellipse_b)

    angle = np.linspace( 0 , 2 * np.pi , 150 ) 
        
    x = ellipse_a * np.cos( angle ) + centre[0]
    y = ellipse_b * np.sin( angle ) + centre[1]

    plt.scatter( 
                # time,
                centre[0],
                centre[1],
                label='left mean',
                color='red'
            )

    plt.plot(x,y,color='green')

    # print(marker_positions[marker_name2id['LHEE'],0])

    plt.plot(

            marker_positions[:,marker_name2id['LHEE'],0],
            marker_positions[:,marker_name2id['LHEE'],1],
            
            color = 'green',
            label = 'LHEE',
            alpha=0.5,

            )    

    
    # plt.scatter(
            
    #         np.mean( marker_positions[:,marker_name2id['LHEE'],0] ),
    #         np.mean( marker_positions[:,marker_name2id['LHEE'],1] ),
            
    #         color = 'green',
    #         label = 'LHEE'
    #         )

    # plt.scatter(
            
    #         np.max( marker_positions[:,marker_name2id['LHEE'],0] ),
    #         np.max( marker_positions[:,marker_name2id['LHEE'],1] ),
            
    #         color = 'green',
    #         label = 'max max'
            
    #         )

    # plt.scatter(
            
    #         np.min( marker_positions[:,marker_name2id['LHEE'],0] ),
    #         np.min( marker_positions[:,marker_name2id['LHEE'],1] ),
            
    #         color = 'green',
    #         label = 'min min'
            
    #         )
    
    # plt.scatter(
            
    #         np.min( marker_positions[:,marker_name2id['LHEE'],0] ),
    #         np.max( marker_positions[:,marker_name2id['LHEE'],1] ),
            
    #         color = 'green',
    #         label = 'min max'
            
    #         )

    # plt.scatter(
            
    #         np.max( marker_positions[:,marker_name2id['LHEE'],0] ),
    #         np.min( marker_positions[:,marker_name2id['LHEE'],1] ),
            
    #         color = 'green',
    #         label = 'max min'
            
    #         )
    
    # plt.scatter(
            
    #         np.mean( marker_positions[:,marker_name2id['LHEE'],0] ),
    #         np.min( marker_positions[:,marker_name2id['LHEE'],1] ),
            
    #         color = 'green',
    #         label = 'LHEE'
            
    #         )

    plt.plot(
            marker_positions[:,marker_name2id['LM1'],0],
            marker_positions[:,marker_name2id['LM1'],1],
            
            color = 'red',
            label = 'LM1',
            alpha=0.5,
            )    

    # plt.scatter(
            
    #         np.mean( marker_positions[:,marker_name2id['LM1'],0] ),
    #         np.mean( marker_positions[:,marker_name2id['LM1'],1] ),
            
    #         color = 'red',
    #         label = 'LM1'
    #         )

    plt.plot(
            marker_positions[:,marker_name2id['LM5'],0],
            marker_positions[:,marker_name2id['LM5'],1],
            
            color = 'blue',
            label = 'LM5',
            alpha=0.5,

            )    

    # plt.scatter(
            
    #         np.mean( marker_positions[:,marker_name2id['LM5'], 0] ),
    #         np.mean( marker_positions[:,marker_name2id['LM5'], 1] ),
            
    #         color = 'blue',
    #         label = 'LM5',

    #         )
    
    ###################################
    

    # plt.plot(
    #         marker_positions[:,marker_name2id['RHEE'],0],
    #         marker_positions[:,marker_name2id['RHEE'],1],
            
    #         color = 'red',
    #         label = 'RHEE'
    #         )   
       
    # COPS_r[:,0] = clip_max_delta_filter(data=COPS_r[:,0],min_val=-1.5,max_val=0.,delta_threshold=0.5)
    # COPS_r[:,1] = clip_max_delta_filter(data=COPS_r[:,1],min_val=-3,max_val=-1.5,delta_threshold=0.02)
    # plt.plot( 
    #             # time,
    #             COPS_r[:,0],
    #             COPS_r[:,1],
    #             label='right',
    #             alpha=0.5

    #         )
    # plt.scatter( 
    #             # time,
    #             np.mean(COPS_r[:,0]),
    #             np.mean(COPS_r[:,1]),
    #             label='right mean',
    #             color='red'
    #         )
    # plt.plot(
    #             time,
    #             # COPS_l[:,1],
    #             clip_max_delta_filter(data=COPS_l[:,1],min_val=-3,max_val=-1.5,delta_threshold=0.02),
    #             label='left pos'
    #         )
    # plt.plot(
    #             time,
    #             # COPS_l[:,1],
    #             clip_max_delta_filter(data=COPS_r[:,1],min_val=-3,max_val=-1.5,delta_threshold=0.02),
    #             label='right pos'
    #         )

    # plt.plot(
    #             time,
    #             # COPS_l[:,1],
    #             clip_max_delta_filter(data=COPS_l[:,0],min_val=-1.5,max_val=0.,delta_threshold=0.5),
    #             label='left pos'
    #         )
    # plt.plot(
    #             time,
    #             # COPS_l[:,1],
    #             clip_max_delta_filter(data=COPS_r[:,1],min_val=-3,max_val=-1.5,delta_threshold=0.02),
    #             label='right pos'
    #         )


    # window = 10
    # data_filtered_y = moving_average(x=COPS_l[:,1],w=window)
    # plt.plot( 
    #             time[window-1:],
    #             # COPS_r[:,0],
    #             data_filtered_y,
    #             label='running avg'
    #         )  
    
    
    

    # plt.ylim(-10,10)
    
    # plt2 = plt.twinx()
    # plt2.plot(
    #             time,
    #             grf_data[:,forces_name2id['Force.Fz'+str(left_leg_plate_id+1)] ],
    #             label='left',
    #             alpha=0.6,
    #             color='red',
    #             linestyle='-.'
    #     )    

    plt.legend()
    plt.grid()
    plt.show()
    exit()
    '''
    
    for key in forces_name2id.keys():
        if 'Moment' in key:
            moment_id = forces_name2id[key]
            grf_data[:,moment_id] = 0.001*grf_data[:,moment_id]

        
    print("Marker Pos. Traj. Shape:", marker_positions.shape)
    print("GRF Traj. Shape:", grf_data.shape)
    if load_mat:
        print("COP Traj. Shape:", cop_data.shape)

    # save the npz file
    if load_mat:
        np.savez_compressed(npzFile_path,marker_positions=marker_positions,grfs=grf_data,cops=cop_data)
    else:
        np.savez_compressed(npzFile_path,marker_positions=marker_positions,grfs=grf_data)


# def generate_scaling_terms():

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--c3d_filepath',help='name of the c3d file',default='AB1_Session1_Right6_Left6',type=str)

    parser.add_argument('--roi_start',help='start index of the region of intrest',default=0,type=int)
    parser.add_argument('--roi_stop',help='stop index of the region of intrest',default=None,type=int)
    parser.add_argument('--static',help='whether given file is a static file',default=False, action='store_true')

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
                        matFile_path=mat_filepath,
                        confFile_path=conf_filepath,
                        roi_start=args.roi_start,
                        roi_stop =args.roi_stop,
                        load_mat = not args.static
                    )    
