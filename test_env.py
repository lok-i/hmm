
from dm_control import suite
from dm_control import viewer
import numpy as np

env = suite.load(domain_name="humanoid", task_name="stand")
action_spec = env.action_spec()

# Define a uniform random policy.
def random_policy(time_step):
  del time_step  # Unused.
  return np.random.uniform(low=action_spec.minimum,
                           high=action_spec.maximum,
                           size=action_spec.shape)

# Launch the viewer application.
viewer.launch(env, policy=random_policy)

# from gym_hmm_ec.envs.bipedal_env import BipedEnv 
# from gym_hmm_ec.controllers.pd_controller import PDController 
# import matplotlib.pyplot as plt
# import numpy as np
# from mujoco_py import functions

# # environment config and setup
# env_conf = {
#             'set_on_rack': False,
#             'render': True,
#             'model_name':'humanoid_CMU_no_hands',
#             'mocap':False
#             }

# env = BipedEnv(**env_conf)

# # initialse the env,reset simualtion
# env.reset()

# # keep the similation in pause until activated manually
# if env.env_params['render']:
#     env.viewer._paused = True

# for _ in range(2000):
    
#     obs,reward,done,info = env.step(action = np.zeros(shape=env.n_act_joints))


# env.close()