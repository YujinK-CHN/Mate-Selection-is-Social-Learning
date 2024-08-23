from pettingzoo.mpe import simple_spread_v3
import gymnasium as gym
import gymnasium as gym

import torch
import numpy as np
from pipline.train import training
from algos.ppo import PPO
from algos.ippo import IPPO
from algos.mappo import MAPPO
from algos.gippo import GIPPO

# ,render_mode="human"
def create_env(env_name, config):
    if env_name == 'Walker2d':
        env = gym.make("Walker2d-v4")
        obs_shape = env.observation_space.shape
        num_actions = env.action_space.shape
        return env, obs_shape, num_actions, True
    if env_name == 'HalfCheetah':
        env = gym.make("HalfCheetah-v4")
        obs_shape = env.observation_space.shape
        num_actions = env.action_space.shape
        return env, obs_shape, num_actions, True
    if env_name == 'trondead': # may be solved by EA
        env = gym.make("ALE/Trondead-ram-v5")
        obs_shape = env.observation_space.shape
        num_actions = env.action_space
        return env, obs_shape, 18, False
    if env_name == 'Boxing':
        env = gym.make("ALE/Boxing-ram-v5")
        obs_shape = env.observation_space.shape
        num_actions = env.action_space
        return env, obs_shape, 18, False
    if env_name == 'BattleZone':
        env = gym.make("ALE/BattleZone-ram-v5")
        obs_shape = env.observation_space.shape
        num_actions = env.action_space
        return env, obs_shape, 18, False


if __name__ == "__main__":
    """ALGO PARAMS"""
    config = {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'obs_shape': None,
        'num_actions': None,
        'continuous': None,
        'pop_size': 1,
        'ent_coef': 0.1,
        'vf_coef': 0.1,
        'clip_coef': 0.1,
        'gamma': 0.99,
        'max_cycles': 32,
        'batch_size': 4,
        'total_episodes': 30000,
        'lr': 0.0003
    }

    """ ENV SETUP """
    env, obs_shape, num_actions, continuous = create_env("Walker2d", config)
    config['obs_shape'] = obs_shape[0]
    config['num_actions'] = num_actions[0]
    config['continuous'] = continuous

    """ ALGO SETUP """
    ppo = PPO(env, config)
    training(config, algo_list=[ppo])
    