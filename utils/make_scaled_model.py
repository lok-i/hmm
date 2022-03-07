
import numpy as np
import os
import yaml
from utils.make_humanoid_mjcf import Humanoid 
from dm_control import mjcf
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model_filename',help='name of the model file',default='default_humanoid_mocap_generated',type=str)
    parser.add_argument('--static_marker_conf',help='name of the preprocessed npz file',default='AB1_Session1_Static',type=str)

    args = parser.parse_args()

    assets_path = './gym_hmm_ec/envs/assets/'
    marker_conf_file_name = args.static_marker_conf+'.yaml'


    config_file = open(assets_path+"our_data/marker_data/confs/"+ marker_conf_file_name,'r+')
    marker_conf = yaml.load(config_file, Loader=yaml.FullLoader)
    config_file.close()
    marker_name2id = marker_conf['marker_name2id']


    to_compute = {
                    'thigh':
                        {
                            'length':{'from':['RGT'],'to':['RKNL','RKNM'],'operation':'measure'},
                            'radius':{'from':['RKNL'],'to':['RKNM'],'operation':'measure+by2'},
                            

                        },
                    'shin':
                        {
                                'length':{'from':['RKNL','RKNM'],'to':['RANL','RANM'],'operation':'measure'},
                                'radius':{'from':['RANL'],'to':['RANM'],'operation':'measure+by2'},

                        },
                    'foot':
                        {
                            'length':{'from':['RHEE'],'to':['RM1','RM5'],'operation':'measure'},
                            'radius':{'from':['RM1'],'to':['RM5'],'operation':'measure+by4'},

                        },
                    'torso_all':
                    {
                    'breadth':{'from':['RGT'],'to':['LGT'],'operation':'measure'},
                    'length':{'operation':'ignore'},


                    }
                
                
                
                }



    marker_geometry_avg = {}
    for link in to_compute:
        marker_geometry_avg[link] = {key: {'value':0.0,'scale':1.0} for key in to_compute[link].keys()}

    print(marker_geometry_avg)

    static_marker_file = 'gym_hmm_ec/envs/assets/our_data/marker_data/processed_data/AB1_Session1_Static_from_0_to_None.npz'
    static_marker_pos = np.load(static_marker_file)['marker_positions']

    n_samples = static_marker_pos.shape[0]
    for n_frame, frame in enumerate(static_marker_pos):
        
        for link in to_compute:
            for metric in to_compute[link].keys():

                if to_compute[link][metric]['operation'] != 'ignore':          
                    k = 0
                    sum_sub_samples = 0.
                    for from_marker in to_compute[link][metric]['from']:
                        for to_marker in to_compute[link][metric]['to']:
                            from_id = marker_name2id[from_marker]
                            to_id = marker_name2id[to_marker]
                            
                            sum_sub_samples += np.linalg.norm( frame[from_id] - frame[to_id])

                            k+=1

                    if to_compute[link][metric]['operation'] == 'measure':
                        marker_geometry_avg[link][metric]['value'] += float(sum_sub_samples/ (k*n_samples) )
                    elif to_compute[link][metric]['operation'] == 'measure+by2':
                        marker_geometry_avg[link][metric]['value'] += float(sum_sub_samples/ (2.0*k*n_samples) )
                    elif to_compute[link][metric]['operation'] == 'measure+by4':
                        marker_geometry_avg[link][metric]['value'] += float(sum_sub_samples/ (4.0*k*n_samples) )

    base_file_name = 'base_model_link_geometry.yaml'
    subject_file_name = args.static_marker_conf+'.yaml'

    base_file = open(assets_path+"models/model_confs/"+ base_file_name,'r')
    base_geom_conf = yaml.load(base_file, Loader=yaml.FullLoader)
    base_geom_conf = base_geom_conf['humanoid']


    for link in marker_geometry_avg.keys():
        for metric in marker_geometry_avg[link].keys():
            base_measure = base_geom_conf[link][metric]
            subj_measure = marker_geometry_avg[link][metric]['value']

            marker_geometry_avg[link][metric]['scale'] = subj_measure / base_measure if to_compute[link][metric]['operation'] != 'ignore' else 1
            print(link,metric,'base:',base_measure,'subj:',subj_measure,'scale:',marker_geometry_avg[link][metric]['scale'])

    subj_geom_conf = {'humanoid':marker_geometry_avg}
    config_file = open(assets_path+"models/model_confs/"+ subject_file_name,'w')
    yaml.dump(subj_geom_conf,config_file)


    scaled_humanoid_conf ={     'torso_h_scale': marker_geometry_avg['torso_all']['length']['scale'],
                                'torso_b_scale': marker_geometry_avg['torso_all']['breadth']['scale'],

                                'leg_scales' :  {


                                    'left_leg':                           
                                    {
                                    'thigh_h_scale':marker_geometry_avg['thigh']['length']['scale'],
                                    'thigh_r_scale':marker_geometry_avg['thigh']['radius']['scale'],

                                    'shin_h_scale':marker_geometry_avg['shin']['length']['scale'],
                                    'shin_r_scale':marker_geometry_avg['shin']['radius']['scale'],

                                    'foot_l_scale':marker_geometry_avg['foot']['length']['scale'],
                                    'foot_r_scale':marker_geometry_avg['foot']['length']['scale']
                                    },
                                
                                    'right_leg':                           
                                    {
                                    'thigh_h_scale':marker_geometry_avg['thigh']['length']['scale'],
                                    'thigh_r_scale':marker_geometry_avg['thigh']['radius']['scale'],

                                    'shin_h_scale':marker_geometry_avg['shin']['length']['scale'],
                                    'shin_r_scale':marker_geometry_avg['shin']['radius']['scale'],

                                    'foot_l_scale':marker_geometry_avg['foot']['length']['scale'],
                                    'foot_r_scale':marker_geometry_avg['foot']['length']['scale']
                                    },
                                                    
                                
                                },


                    }


    # if os.path.exists(assets_path+"models/model_confs/"+ model_file+'.yaml'):
    #     print( 'Warning: the file '+model_file+' already exists, wanna rewrite ?[y/n]',end=' ')
    #     key = input()    
    # if key == 'y':
    #     print("Model Conf. File Updated")
    #     config_file = open(assets_path+"models/model_confs/"+ model_file+'.yaml','w')
    #     marker_conf = yaml.dump(scaled_humanoid_conf, config_file)

    body = Humanoid(name='humanoid',
                    **scaled_humanoid_conf
                    )
    physics = mjcf.Physics.from_mjcf_model(body.mjcf_model)

    mjcf.export_with_assets(body.mjcf_model,"./gym_hmm_ec/envs/assets/models/",args.model_filename+'.xml')
    print("Model File Updated")
    print("written to ","./gym_hmm_ec/envs/assets/models/"+args.model_filename+'.xml')
