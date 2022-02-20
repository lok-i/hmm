
import numpy as np
import os
import yaml
from gym_hmm_ec.envs.utils.make_humanoid_mjcf import Humanoid 
from dm_control import mjcf

assets_path = './gym_hmm_ec/envs/assets/'
marker_conf_file_name = 'marker_config.yaml'


config_file = open(assets_path+"our_data/"+ marker_conf_file_name,'r+')
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
                'breadth':{'from':['RSHO'],'to':['LSHO'],'operation':'measure'},
                'length':{'from':['RSHO'],'to':['RGT'],'operation':'measure'},


                }
            
            
            
            }

from_mark = 'RKNL'
to_mark = ['RKNL', 'RKNM']

marker_geometry_avg = {}
for link in to_compute:
    marker_geometry_avg[link] = {key: {'value':0.0,'scale':1.0} for key in to_compute[link].keys()}

print(marker_geometry_avg)

n_samples = 19
for n_frame in range(n_samples):
    print("Frame:",n_frame)
    frame = np.load(assets_path+"our_data/mocap_data/frame_rand"+str(n_frame)+".npy")

    for link in to_compute:
        for metric in to_compute[link].keys():
            
                
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
            if to_compute[link][metric]['operation'] == 'measure+by2':
                marker_geometry_avg[link][metric]['value'] += float(sum_sub_samples/ (2.0*k*n_samples) )
            if to_compute[link][metric]['operation'] == 'measure+by4':
                marker_geometry_avg[link][metric]['value'] += float(sum_sub_samples/ (4.0*k*n_samples) )
                # print(marker_geometry_avg[link][metric])

base_file_name = 'base_model_link_geometry.yaml'
subject_file_name = 'heuristic_link_geometry_file.yaml'

base_file = open(assets_path+"models/model_confs/"+ base_file_name,'r')
base_geom_conf = yaml.load(base_file, Loader=yaml.FullLoader)
base_geom_conf = base_geom_conf['humanoid']


for link in marker_geometry_avg.keys():
    for metric in marker_geometry_avg[link].keys():
        base_measure = base_geom_conf[link][metric]
        subj_measure = marker_geometry_avg[link][metric]['value']

        scale = subj_measure / base_measure
        marker_geometry_avg[link][metric]['scale'] = scale
        print(link,metric,'base:',base_measure,'subj:',subj_measure,'scale:',scale)

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


model_file = 'rand_1'
if os.path.exists(assets_path+"models/model_confs/"+ model_file+'.yaml'):
    print( 'Warning: the file '+model_file+' already exists, wanna rewrite ?[y/n]',end=' ')
    key = input()    
if key == 'y':
    print("Model Conf. File Updated")
    config_file = open(assets_path+"models/model_confs/"+ model_file+'.yaml','w')
    marker_conf = yaml.dump(scaled_humanoid_conf, config_file)

body = Humanoid(name='humanoid',
                **scaled_humanoid_conf
                )
physics = mjcf.Physics.from_mjcf_model(body.mjcf_model)

mjcf.export_with_assets(body.mjcf_model,"./gym_hmm_ec/envs/assets/models",model_file+'.xml')
print("Model File Updated")
