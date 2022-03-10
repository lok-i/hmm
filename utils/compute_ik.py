
from gym_hmm_ec.envs.bipedal_env import BipedEnv 
import numpy as np
from tqdm import tqdm
import yaml
import argparse
from dm_control import mujoco
from utils.ik_solver import qpos_from_site_pose


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--processed_filepath',help='name of the preprocessed mocap file',default='AB1_Session1_Right6_Left6',type=str)
    parser.add_argument('--model_filename',help='name of the model file',default='default_humanoid_mocap_generated',type=str)

    parser.add_argument('--export_solns',help='whether to export the ik solns',default=False, action='store_true')
    parser.add_argument('--render',help='whether to render while solving for ik',default=False, action='store_true')

    args = parser.parse_args()

    assets_path = './gym_hmm_ec/envs/assets/'

    mocap_data = np.load(args.processed_filepath)

    # all configs
    # environment config and setup
    env_conf = {
                'set_on_rack': False,
                'render': args.render,
                'model_name': args.model_filename,
                'mocap':False
                }
    # marker config
    marker_confpath = args.processed_filepath.split('processed_data/')[0]+'confs/' \
                        + args.processed_filepath.split('processed_data/')[-1].split('_from')[0]+'.yaml' #.('_from_')[0]+'.yaml'


    marker_config_file = open(marker_confpath,'r+')
    marker_conf = yaml.load(marker_config_file, Loader=yaml.FullLoader)
    # exit()


    env = BipedEnv(**env_conf)
    env.model.opt.gravity[2] = 0

    # initialse the env,reset simualtion
    env.reset()
    physics = mujoco.Physics.from_xml_path(assets_path+"models/"+env.env_params['model_name']+".xml")



    marker_names = ['RASI','LASI','RPSI','LPSI',
                    "right_leg/RKNL", "right_leg/RKNM","right_leg/RANL","right_leg/RANM","right_leg/RHEE","right_leg/RM1","right_leg/RM5",
                    "left_leg/LKNL", "left_leg/LKNM","left_leg/LANL","left_leg/LANM","left_leg/LHEE","left_leg/LM1","left_leg/LM5"
                    ]
    relavent_doFs = ['root',
                    'right_leg/hip_x','right_leg/hip_y','right_leg/hip_z','right_leg/knee','right_leg/ankle_y','right_leg/ankle_x',
                     'left_leg/hip_x','left_leg/hip_y','left_leg/hip_z','left_leg/knee','left_leg/ankle_y','left_leg/ankle_x',
                    ]

    # keep the similation in pause until activated manually
    if env.env_params['render']:
        env.viewer._paused = True
        env.viewer.cam.distance = 3
        cam_pos = [0.0, 0.5, 0.75]

        for i in range(3):        
            env.viewer.cam.lookat[i]= cam_pos[i] 
        env.viewer.cam.elevation = -15
        env.viewer.cam.azimuth = 180

    if args.export_solns:
        ik_solns = []
        rfoot_xpos = []
        lfoot_xpos = []
        pelvis_xpos = []

    for frame in tqdm(mocap_data['marker_positions']):

        target_qpos = np.zeros(len(env.sim.data.qpos))

        target_posses = []
        for target_site in  marker_names:
            
            temp_target_site = target_site 
            if 'right_leg'in target_site:
                temp_target_site = target_site.replace('right_leg/','')
            if 'left_leg'in target_site:
                temp_target_site = target_site.replace('left_leg/','')             
            marker_id = marker_conf['marker_name2id'][temp_target_site]
            marker_name = 'm'+str(marker_id)
            env.sim.data.set_mocap_pos(marker_name, frame[marker_id,:] )
            target_posses.append( frame[ marker_conf['marker_name2id'][temp_target_site]  ,:].tolist() )
        target_posses = np.array(target_posses)

        # syncornize the mujoco_py start with ik (dm_control's) pose
        for i in range(len(env.sim.data.qpos)):
            physics.data.qpos[i] = env.sim.data.qpos[i] 
            target_qpos[i] = env.sim.data.qpos[i] 
        
        
        ik_result = qpos_from_site_pose(physics,
                                        site_names= marker_names,
                                        target_pos=target_posses,
                                        joint_names = relavent_doFs,
                                        max_steps=1000,
                                        inplace=False
                                        )

                
        target_qpos = ik_result.qpos.copy() 
        n_steps_per_pose = 1        
        for i in range(n_steps_per_pose):
            for i in range(len(env.sim.data.qpos)):
                env.sim.data.qpos[i] = target_qpos[i]#physics.data.qpos[i]

            obs,reward,done,info = env.step(action = np.zeros(shape=env.n_act_joints))
        if args.export_solns:
            ik_solns.append(target_qpos.tolist())
            body_name = 'right_leg/foot'
            body_id = env.sim.model.body_name2id(body_name)
            rf_xp = env.sim.data.get_body_xpos(body_name)
            rfoot_xpos.append(rf_xp.tolist())

            body_name = 'left_leg/foot'
            body_id = env.sim.model.body_name2id(body_name)
            lf_xp = env.sim.data.get_body_xpos(body_name)
            lfoot_xpos.append(lf_xp.tolist())

            body_name = 'pelvis'
            body_id = env.sim.model.body_name2id(body_name)
            p_xp = env.sim.data.get_body_xpos(body_name)
            pelvis_xpos.append(p_xp.tolist())

        # to rotate camera 360 around the model
        # env.viewer.cam.azimuth += 0.25#180 

    if args.export_solns:
        ik_solns = np.array(ik_solns)
        rfoot_xpos = np.array(rfoot_xpos)
        lfoot_xpos = np.array(lfoot_xpos)
        pelvis_xpos = np.array(pelvis_xpos)

        print('IK Soln Shape',ik_solns.shape)
        output_filepath = args.processed_filepath.split('marker_data/processed_data/')[0]+'ik_solns/' \
                        + args.processed_filepath.split('marker_data/processed_data/')[-1] 
        np.savez_compressed(output_filepath,
                                ik_solns=ik_solns,
                                rfoot_xpos=rfoot_xpos,
                                lfoot_xpos=lfoot_xpos,
                                pelvis_xpos = pelvis_xpos,
                            )
        print("IK Solution written to:", output_filepath)