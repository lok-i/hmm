import c3d
import yaml
import scipy.io
import argparse
import numpy as np
from tqdm import tqdm

# TODO: 
# 1. find why the COP is not generated in 
#    c3d and fix it, remove .mat dependency once done.
# 2. npz file naming from index to time sytems ?

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
                        roi_start = 0.,
                        roi_stop = None
                        ):


    mat_data = scipy.io.loadmat(matFile_path)
    with open( dataFile_path , 'rb') as handle:
        reader = c3d.Reader(handle)
        
        roi_stop = reader.last_frame if roi_stop == None else roi_stop
        
        pbar = tqdm(total=roi_stop)
        pbar.set_description("Loading mocap data")   
        
        marker_positions = []
        grf_data = []
        cop_data = []

        print([0])
        for data in reader.read_frames():
            pbar.update(1)        
            
            # TODO: Do nan chks
            if data[0] > roi_start: 
                marker_positions.append(data[1][:,0:3].tolist())
                
                # just the first analog reading in the frame
                grf_data.append(data[2][:,0].tolist())
                
                # read the cop's
                p1 = mat_data['forceStruct']['p1'][0][0][data[0]].tolist()
                p2 = mat_data['forceStruct']['p2'][0][0][data[0]].tolist()
                p3 = mat_data['forceStruct']['p3'][0][0][data[0]].tolist()

                cop_data.append(p1+p2+p3)

            if data[0] > roi_stop:
                break

    # mm to m
    marker_positions = 0.001*np.array(marker_positions)
    cop_data = 0.001*np.array(cop_data)

    # for forces, only moments should be converted from Nmm to Nm
    grf_data = np.array(grf_data)
    config_file = open(confFile_path,'r+')
    marker_conf = yaml.load(config_file, Loader=yaml.FullLoader)
    forces_name2id = marker_conf['forces_name2id']

    for key in forces_name2id.keys():
        if 'Moment' in key:
            moment_id = forces_name2id[key]
            grf_data[:,moment_id] = 0.001*grf_data[:,moment_id]

        
    print("Marker Pos. Traj. Shape:", marker_positions.shape)
    print("GRF Traj. Shape:", grf_data.shape)
    print("COP Traj. Shape:", cop_data.shape)

    # save the npz file
    np.savez_compressed(npzFile_path,marker_positions=marker_positions,grfs=grf_data,cops=cop_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--c3d_filename',help='name of the c3d file',default='AB1_Session1_Right6_Left6',type=str)
    parser.add_argument('--roi_start',help='start index of the region of intrest',default=0,type=int)
    parser.add_argument('--roi_stop',help='stop index of the region of intrest',default=None,type=int)

    args = parser.parse_args()  

    assets_path = './gym_hmm_ec/envs/assets/'
    
    # input files
    c3d_filepath = assets_path+ 'our_data/marker_data/c3ds/'+args.c3d_filename+'.c3d'
    mat_filepath =  assets_path+'our_data/marker_data/mats/'+args.c3d_filename+'.mat'

    # output files
    conf_filepath = assets_path+'our_data/marker_data/confs/'+args.c3d_filename+'.yaml'
    
    npz_filepath = assets_path+'our_data/marker_data/processed_data/'+args.c3d_filename\
                   +'_from_'+str(args.roi_start)+'_to_'+str(args.roi_stop)


    generate_conf_file(
                        dataFile_path=c3d_filepath,
                        confFile_path=conf_filepath
                    )
    
    generate_npz_file(
                        dataFile_path=c3d_filepath,
                        npzFile_path=npz_filepath,
                         matFile_path=mat_filepath,
                        confFile_path=conf_filepath,
                        roi_start=args.roi_start,
                        roi_stop =args.roi_stop,
                    )    
