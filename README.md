# hmm-ec
an env for learning human motor models and exoskeleton control

# Getting Started and Installation:

In order to keep it experiment friendly, this codebase uses local scripts with very less modules to install/maintain. The only requirement is the sucessfull installation of mujoco and mujoco-py.

## Install MuJoCo

1. Download the MuJoCo version 2.1 binaries here: [Mujoco](https://mujoco.org/download).

2. Extract the downloaded mujoco210 directory into ~/.mujoco/mujoco210.

If you want to specify a nonstandard location for the package, use the env variable MUJOCO_PY_MUJOCO_PATH.

## Install mujoco-py

To install mujoco-py, run the following

        pip3 install -U 'mujoco-py<2.2,>=2.1'


After testing the sucessfull isntallation of mujoco-py,to test the codebase,run

        cd hmm-ec/
        python3 test_env.py

# Usage Instructions

Since it is a evolving code base, at present kindly follow the usage and implementations in [test_env.py](./test_env.py)

