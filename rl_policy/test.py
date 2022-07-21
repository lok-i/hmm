from gym_hmm_ec.controllers.pd_controller import PDController 
import matplotlib.pyplot as plt
import numpy as np
import yaml
import gym
from stable_baselines3 import PPO
from gym_hmm_ec.envs.bipedal_env import BipedEnv as Env 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import argparse
import imageio

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--single_trng_path',help='path to exp folder with exp config', default='')
parser.add_argument('--checkpoint',help='timestamp of checkpoint', default=None)
parser.add_argument('--gif',help='bool save gif', default=False)
args = parser.parse_args()
config_file = open(args.single_trng_path+"conf.yaml")
training_config = yaml.load(config_file, Loader=yaml.FullLoader)
env_conf =  training_config['env_kwargs'].copy()
env_conf['render'] = True

env = make_vec_env(Env,env_kwargs=env_conf, n_envs= 1)
env = VecNormalize(venv=env,clip_obs=np.inf)

if (not args.checkpoint) :
    model = PPO.load(args.single_trng_path+"Final_Policy")
    gif_path = args.single_trng_path + "final.gif"

else:
    model = PPO.load(args.single_trng_path+ "checkpoints/rl_model_" + args.checkpoint + "_steps")
    gif_path = args.single_trng_path + args.checkpoint + '.gif'

obs = env.reset()

if(args.gif):
    
    images = []
    while True:
        control_actions, _states = model.predict(obs,deterministic = True) 
        obs,reward,done,info = env.step(control_actions)
        img = env.render(mode='rgb_array')
        images.append(img)
        if (done):
            break

    imageio.mimsave(gif_path, [np.array(img) for i, img in enumerate(images[10:]) if i%2 == 0], fps=60)

    env.close()

else:
    while True:
        control_actions, _states = model.predict(obs,deterministic = True)  ## Make it deterministic
        obs,reward,done,info = env.step(control_actions)
        # print(env.get_attr('sim')[0].data.qvel[0])
