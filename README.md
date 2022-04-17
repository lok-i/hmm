# hmm-ec
an env for learning human motor models and exoskeleton control

# Getting Started and Installation:

In order to keep it experiment friendly, this codebase uses local scripts with very less modules to install/maintain. The only requirement is the sucessfull installation of mujoco and mujoco-py.

## Install MuJoCo

1. Download the MuJoCo version 2.1 binaries here: [Mujoco](https://mujoco.org/download).

2. Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.

If you want to specify a nonstandard location for the package, use the env variable `MUJOCO_PY_MUJOCO_PATH`.

## Install mujoco-py

To install mujoco-py, run the following

        pip3 install -U 'mujoco-py<2.2,>=2.1'


After testing the sucessfull isntallation of mujoco-py,to test the codebase,run

        cd hmm-ec/
        python3 test_env.py

# Usage Instructions

Since it is a evolving code base, at present kindly follow the usage and implementations in [demos](./demos) and [utils](./utils).


## Expected Data Directory Structure

Make a directory named data and a directory for your data with the following structure,

        data 
        |- your_data
           |- marker_data
              |- c3ds # keep all your c3d files inside this folder
           |- id_solns
           |- ik_solns


To generate an end to end demo, simply run

        python3 utils/compute_all.py --static_c3dfilepath data/your_data/marker_data/c3ds/static.c3d --trial_c3dfilepath data/your_data/marker_data/c3ds/trial.c3d  --plot_id_solns --plot_ik_solns 

# To Do

- [ ] data directory addition in read me.
- [ ] update commands.md.
- [ ] COP point of application to be fixed.
- [x] 
