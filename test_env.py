from gym_hmm_ec.envs.bipedal_env import BipedEnv as BipedEnv
import numpy as np

env = BipedEnv(model_name='humanoid')
done = False
env.reset()
if env.is_render:
    # env.reset_camera()
    # env.viewer._hide_overlay = TrueT
    env.viewer._paused = True

for _ in range(10000):

    o,r,d,info = env.step(action = np.zeros(env.action_dim))
    env.render()

env.close()