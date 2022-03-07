    

# process motion trial, grf and cop data    

## Static Data

    python3 utils/preprocess_data.py --c3d_filename AB1_Session1_Static --static

## Dynamic Data: 
    python3 gym_hmm_ec/envs/utils/preprocess_data.py --c3d_filename AB1_Session1_Right6_Left6 --roi_start 1200 --roi_stop 1500    

# process static data    

    python3 gym_hmm_ec/envs/utils/preprocess_data.py --c3d_filename AB1_Session1_Static --static

# create nominal mujoco model

    python3 gym_hmm_ec/envs/utils/make_humanoid_mjcf.py --conf_xml_filename default_humanoid_mocap_generated

# update mujoco model form factor and marker placments

    python3 gym_hmm_ec/envs/utils/mujoco_model_editor/main.py --input gym_hmm_ec/envs/assets/models/default_humanoid_mocap_generated.xml --static_input gym_hmm_ec/envs/assets/our_data/marker_data/processed_data/AB1_Session1_Static_from_0_to_None.npz

# compute inverse kinematics

    python3 gym_hmm_ec/envs/utils/compute_ik.py --render --mocap_npz_filename AB1_Session1_Right6_Left6_from_1200_to_1500 --export_solns

# compute inverse dynamics

    python3 gym_hmm_ec/envs/utils/compute_id.py --mocap_npz_filename AB1_Session1_Right6_Left6_from_1200_to_1500 --render --export_solns --plot_solns