import argparse
import subprocess
import os
import shutil
import argparse
'''
python3 generate_experiment_setup.py --f 1Oct1

python3 submit_jobs.py --f 1Oct1 --bid 1
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_job_folderpath',help='path to the folder with the job confs', default='')
    
    parser.add_argument('--output_exp_path',help='path to the location where the exp folder is to be created', default='')
    args = parser.parse_args()


    exp_confs = os.listdir(args.input_job_folderpath)
    
    for exp_name in exp_confs:
        exp_name = exp_name.replace('.yaml','')
        print("Experiment Name:",exp_name)
        
        create_exp_folders = True
        output_exp = args.output_exp_path  + exp_name 
        if os.path.isdir(output_exp):
            print( output_exp, 'already exists, wanna delete it ?[y/n]',end=' ')
            key = input()
            if key == 'y':
                shutil.rmtree(output_exp)
            else:
                print("skipped recreating the exp folders")
                create_exp_folders = False
        
        if create_exp_folders:
            subprocess.run([
                            "python3","./rl_policy/deploy_trainings/generate_experiment_setup.py",
                            '--job_folderpath',args.input_job_folderpath,
                            "--exp_path",args.output_exp_path,
                            "--exp_name",exp_name])
            
            #should later add a job submission command if required to run experiments in a cluster
            # subprocess.run(["python3","./utils/submit_jobs.py","--f",exp_name,'--bid','1'])
