    

For full syncronized demo and usage, checkout [./utils/compute_all.py](./compute_all.py)

# Process motion trial, grf and cop data    

## Static Data

    python3 utils/preprocess_data.py --c3d_filepath `file path` --static

## Dynamic Data: 
    python3 utils/preprocess_data.py --c3d_filepath `file path` --roi_start 1200 --roi_stop 1500    

# Make nominal mujoco model

## unscaled default model
    python3 utils/make_humanoid_mjcf.py --conf_xml_filename default_humanoid_mocap

## scaled model from static file
    python3 utils/make_scaled_model.py --static_confpath `file path` --processed_filepath `file path`

# Update mujoco model form factor and marker placements

    python3 utils/mujoco_model_editor/main.py --input gym_hmm_ec/envs/assets/models/rand_1.xml --static_input data/our_data/marker_data/processed_data/AB1_Session1_Static_from_0_to_None.npz

# Compute inverse kinematics

    python3 utils/compute_ik.py --processed_filepath `file path` --model_filename rand_1_updated --render  --export_solns

# Compute inverse dynamics

    python3 utils/compute_id.py --processed_filepath `file path` --model_filename rand_1_updated --export_solns  --render  --plot_solns 