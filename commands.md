    

# process motion trial, grf and cop data    

## Static Data

    python3 utils/preprocess_data.py --c3d_filename AB1_Session1_Static --static

## Dynamic Data: 
    python3 utils/preprocess_data.py --c3d_filename AB1_Session1_Right6_Left6 --roi_start 1200 --roi_stop 1500    

# process static data    

    python3 utils/preprocess_data.py --c3d_filename AB1_Session1_Static --static

# make nominal mujoco model

## unscaled default model
    python3 utils/make_humanoid_mjcf.py --conf_xml_filename default_humanoid_mocap_generated

## scaled model from static file
    python3 utils/make_scaled_model.py --model_filename rand_1 --static_marker_conf AB1_Session1_Static

# update mujoco model form factor and marker placments

    python3 utils/mujoco_model_editor/main.py --input gym_hmm_ec/envs/assets/models/rand_1.xml --static_input gym_hmm_ec/envs/assets/our_data/marker_data/processed_data/AB1_Session1_Static_from_0_to_None.npzz

# compute inverse kinematics

    python3 utils/compute_ik.py --render --mocap_npz_filename AB1_Session1_Right6_Left6_from_1200_to_1500 --export_solns --model_filename rand_1_updated

# compute inverse dynamics

    python3 utils/compute_id.py --mocap_npz_filename AB1_Session1_Right6_Left6_from_1200_to_1500 --render --export_solns --plot_solns --model_filename rand_1_updated