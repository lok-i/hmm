import gym
import os
import argparse
import torch
import yaml
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
from gym_hmm_ec.envs.bipedal_env import BipedEnv as Env 


# callback to save the trinaining normalisation stastics
class SaveNormDataCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` steps
    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param verbose:
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "vec_normalize", verbose: int = 0):
        super(SaveNormDataCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}.pkl")
            self.model.env.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
        return True

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        current_reward_state = self.model.env.get_attr('reward_value_list')[0]
        for rew_term in current_reward_state.keys():
            if isinstance(current_reward_state[rew_term], dict):
                for key in current_reward_state[rew_term].keys():
                    self.logger.record('reward_terms/'+rew_term+"_"+key, current_reward_state[rew_term][key] )  
            else:
                self.logger.record('reward_terms/'+rew_term, current_reward_state[rew_term])
        return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--single_trng_path',help='path to exp folder with exp config', default='')
    args = parser.parse_args()
    # NOTE: The exp_path comes with a '/' in the last

    
    #load the configuration    
    config_file = open(args.single_trng_path+"conf.yaml")
    training_config = yaml.load(config_file, Loader=yaml.FullLoader)
    # print(training_config['env_kwrgs']['reward_weights'])
    
    # make the env
    # env = make_vec_env(training_config['rl_setting']['env'], n_envs=1)

    env = make_vec_env(Env,env_kwargs=training_config['env_kwargs'], n_envs=1)
    # check_env(env)

    # exit()

    env = VecNormalize(venv=env,clip_obs=np.inf) 
    
    
    # POLICY NETWORK PROPERTIES	
    policy_kwargs = training_config['rl_setting']['policy']['kwargs']
    if policy_kwargs['activation_fn'] == "ReLU":
        policy_kwargs['activation_fn'] =  torch.nn.ReLU

    #initialize PPO configuration
    if isinstance(training_config['rl_setting']['algo_hyperparameters'] , dict):            
    
        #  learning rate scheduling
        if 'learning_rate' in training_config['rl_setting']['algo_hyperparameters'].keys():
            if isinstance(training_config['rl_setting']['algo_hyperparameters']['learning_rate'], str):
                schedule, initial_value = training_config['rl_setting']['algo_hyperparameters']['learning_rate'].split("_")
                initial_value = float(initial_value)

                # other schedules are yet to be implemented
                if schedule == 'lin':
                    def linear_lr_schedule(traingProgess, i_lr = initial_value):
                        return i_lr*traingProgess # trainingProgress: 1 -> 0
                    training_config['rl_setting']['algo_hyperparameters']['learning_rate'] = linear_lr_schedule
                        
        model =  PPO(policy=training_config['rl_setting']['policy']['type'], 
                    env = env, 
                    tensorboard_log=args.single_trng_path, 
                    policy_kwargs=policy_kwargs,                    
                    **training_config['rl_setting']['algo_hyperparameters']
                    )
    else:
        print("No hyper parameters defined for PPO, default initialisations being used")
        model =  PPO(policy=training_config['rl_setting']['policy']['type'], 
                    env = env, 
                    tensorboard_log=args.single_trng_path, 
                    policy_kwargs=policy_kwargs,                    
                    )

    # to checkpoint polices along training
    if os.path.isdir(args.single_trng_path +"checkpoints/") == False:
        os.mkdir(args.single_trng_path +"checkpoints/")

    path_to_checkpts = args.single_trng_path +"checkpoints/"
    
    checkpoint_callback = CheckpointCallback(save_freq=25000, save_path=path_to_checkpts,name_prefix='rl_model')
    trng_norm_stat_callback = SaveNormDataCallback(save_freq=25000, save_path=path_to_checkpts)
    tensorborad_xtra_callback = TensorboardCallback()
    all_callbacks = CallbackList([checkpoint_callback,trng_norm_stat_callback , tensorborad_xtra_callback])

    # train function
    model.learn(
    total_timesteps=training_config['rl_setting']['total_timesteps'], 
    callback=all_callbacks, 
    log_interval=1,  
    n_eval_episodes=5, 
    tb_log_name='PPO', 
    eval_log_path=args.single_trng_path)

    model.save(args.single_trng_path+"Final_Policy")
    model.env.save(args.single_trng_path+"/vec_normalize.pkl")
    
	