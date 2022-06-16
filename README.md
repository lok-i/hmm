# hmm
a mujoco env for learning human motor models through deep rl with added tools for IK, ID, direct mocap data integration, and stablebaselines3 integration.

# Getting Started and Installation:
 
### Install MuJoCo

1. Download the MuJoCo version 2.1 binaries here: [Mujoco](https://mujoco.org/download).

2. Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.

If you want to specify a nonstandard location for the package, use the env variable `MUJOCO_PY_MUJOCO_PATH`.

### Install mujoco-py

To install mujoco-py, run the following

        pip3 install -U 'mujoco-py<2.2,>=2.1'


After testing the sucessfull isntallation of mujoco-py,to test the codebase,run

        cd hmm-ec/
        python3 test_env.py

### Install dm_control

This is an optional installation. Required for building a [custom scaled model from data](./utils/make_humanoid_mjcf.py)  and to [compute IK](./utils/compute_ik.py) .
        
        pip3 install dm_control

### Install Stable-Baselines3

This is an optional installation. Required for training and testing [rl policies](./rl_policy/train.py)
        
        pip3 install stable-baselines3


# Usage Instructions

Checkout the readme's inside [utils](./utils) and [rl_policy](./rl_policy) for corresponding specific usage of the scripts.


## Expected Data Directory Structure

Make a directory named data and a directory for your data with the following structure,

        hmm-ec
            |-data 
                |- your_data
                |- marker_data
                    |- c3ds # keep all your c3d files inside this folder
                |- id_solns
                |- ik_solns


### For example,
You can download our [data](https://drive.google.com/drive/folders/1zU2zTHr110v3TbVdbKXGiXlRRw0raZVT)  folder, that we use to recreate the experiments we conduct. Place it inside `hmm-ec/` 

To generate an end to end demo of the tools in [./utils] , simply run

#### For simple model (point mass massless leg)

        python3 utils/compute_all.py --model_type pm_mll --static_c3dfilepath data/our_data/marker_data/c3ds/AB3_Session1_Static.c3d --trial_c3dfilepath data/our_data/marker_data/c3ds/AB3_Session1_Right10_Left10.c3d --roi_start 2000 --roi_stop 2100 --plot_ik_solns --plot_id_solns --render_ik --render_id

#### For full humanoid model

        python3 utils/compute_all.py --model_type humanoid --static_c3dfilepath data/our_data/marker_data/c3ds/AB3_Session1_Static.c3d --trial_c3dfilepath data/our_data/marker_data/c3ds/AB3_Session1_Right10_Left10.c3d --roi_start 2000 --roi_stop 2100 --plot_ik_solns --plot_id_solns --render_ik --render_id

# Colab Notebook

You can find the detaild usage of the various tools in this code base in thie google [colab notebook](https://colab.research.google.com/drive/1C0Oatm7bamBYBOxtQU2C1DOgUbSjWFyk?usp=sharing).

# To Do

- [ ] Complete readme in utils/ and rl_policy/ (make illustrative figures aswell).
- [ ] Add other deep rl algo support from sb3 in rl_policy/
- [ ] Add an well explained colab notebook, well explained and with exhaustive examples.

