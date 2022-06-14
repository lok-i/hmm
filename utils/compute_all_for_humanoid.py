import argparse
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--static_c3dfilepath',help='path of the static file',default='AB1_Session1_Right6_Left6',type=str)
    parser.add_argument('--trial_c3dfilepath',help='path of the trial file',default='AB1_Session1_Right6_Left6',type=str)
    parser.add_argument('--roi_start',help='start index of the region of intrest',default=0,type=int)
    parser.add_argument('--roi_stop',help='stop index of the region of intrest',default=None,type=int)

    parser.add_argument('--plot_preprocess', help='whether to plot op of pre process',default=False, action='store_true')


    parser.add_argument('--render_ik',help='whether to render while solving for ik',default=False, action='store_true')
    parser.add_argument('--plot_ik_solns', help='whether to plot the ik solns',default=False, action='store_true')

    parser.add_argument('--render_id',help='whether to render while solving for ik',default=False, action='store_true')
    parser.add_argument('--plot_id_solns', help='whether to plot the id solns',default=False, action='store_true')


    args = parser.parse_args()
    
    ############### DATA PRE-PROCESSING #######################
    
    print("Pre-processing files ...")
    print("Static File:\n")
    os.system('python3 utils/preprocess_data.py --c3d_filepath '+ args.static_c3dfilepath +' --static')
    print("\nTrial File:\n")
    

    preprocess_command = 'python3 utils/preprocess_data.py --c3d_filepath '+ args.trial_c3dfilepath+' --roi_start '+str(args.roi_start)
    if not args.roi_stop == None:
        preprocess_command += ' --roi_stop '+str(args.roi_stop)
    if args.plot_preprocess:
        preprocess_command += ' --plot '
        
    os.system(preprocess_command)
    
    ############### MODEL GENERATION #######################
    
    c3d_removed_path = args.static_c3dfilepath.replace('.c3d','')
    
    conf_filepath = c3d_removed_path.replace('c3ds','confs')+'.yaml'
    
    proceesed_filepath = c3d_removed_path.replace('c3ds','processed_data')+'_from_0_to_None.npz'

    print("\nPreparing scaled model ...")
    
    
    os.system('python3 utils/make_scaled_model.py --static_confpath '+conf_filepath+' --static_processed_filepath '+ proceesed_filepath)

    subject_file_name = conf_filepath.split('/')[-1].replace('_Static','')
    
    model_filename = subject_file_name.replace('.yaml','_humanoid.xml')
    model_filepath = "./gym_hmm_ec/envs/assets/models/"+model_filename
    
    upd_model_filename = model_filename.replace('.xml','_upd.xml')
    upd_model_filepath = model_filepath.replace('.xml','_upd.xml')
    
    key = 'y'
    if os.path.exists(upd_model_filepath):
        print( '\nWarning: the file '+upd_model_filename+' already exists, wanna edit again ?[y/n]',end=' ')
        key = input()    
    if key == 'y':
        os.system('python3 utils/mujoco_model_editor/main.py --input_modelpath '+model_filepath+' --static_filepath '+proceesed_filepath)
        print("File Updated")
    
    
    ############### COMPUTE IK #######################
    c3d_removed_path = args.trial_c3dfilepath.replace('.c3d','')
    
    conf_filepath = c3d_removed_path.replace('c3ds','confs')+'.yaml'
    
    proceesed_filepath = c3d_removed_path.replace('c3ds','processed_data')\
                   +'_from_'+str(args.roi_start)+'_to_'+str(args.roi_stop)+'.npz'


    ik_command = 'python3 utils/compute_ik.py --processed_filepath '+proceesed_filepath+' --model_filename '+upd_model_filename+' --export_solns'


    if args.render_ik:
        ik_command += ' --render'
    if args.plot_ik_solns:
        ik_command += ' --plot_solns'
    os.system(ik_command)
    

    ############### COMPUTE ID #######################

    id_command = 'python3 utils/compute_id.py --processed_filepath '+proceesed_filepath+' --model_filename '+upd_model_filename+' --export_solns'
    
    if args.render_id:
        id_command += ' --render'
    if args.plot_id_solns:
        id_command += ' --plot_solns'
    os.system(id_command)