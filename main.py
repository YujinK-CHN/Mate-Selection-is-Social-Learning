from pettingzoo.sisl import waterworld_v4, multiwalker_v9
from pettingzoo.mpe import simple_spread_v3


import torch
from pipline.train import training
from algos.ippo import IPPO
from algos.mappo import MAPPO


def create_env(config):
    if config['env_name'] == 'multiwalker':
        multiwalker = multiwalker_v9.parallel_env(n_walkers=config['n_agents'], position_noise=1e-3, angle_noise=1e-3, forward_reward=1.0, terminate_reward=-100.0, fall_reward=-10.0, shared_reward=True, \
                                                terminate_on_fall=True, remove_on_fall=True, terrain_length=200, max_cycles=config['max_cycles'])
        obs_shape = len(multiwalker.observation_space(multiwalker.possible_agents[0]).sample())
        num_actions = len(multiwalker.action_space(multiwalker.possible_agents[0]).sample())
        return multiwalker, obs_shape, num_actions
    if config['env_name'] == 'waterworld':
        waterworld = waterworld_v4.parallel_env(n_pursuers=config['n_agents'], n_evaders=8, n_poisons=10, n_coop=1, n_sensors=20,\
                                                sensor_range=0.2,radius=0.015, obstacle_radius=0.2, n_obstacles=1,\
                                                obstacle_coord=[(0.5, 0.5)], pursuer_max_accel=0.01, evader_speed=0.01,\
                                                poison_speed=0.01, poison_reward=-1.0, food_reward=10.0, encounter_reward=0.01,\
                                                thrust_penalty=-0.5, local_ratio=1.0, speed_features=True, max_cycles=config['max_cycles'])
        obs_shape = len(waterworld.observation_space(waterworld.possible_agents[0]).sample())
        num_actions = len(waterworld.action_space(waterworld.possible_agents[0]).sample())
        return waterworld, obs_shape, num_actions
    if config['env_name'] == 'simple_spread':
        simple_spread = simple_spread_v3.parallel_env(N=config['n_agents'], local_ratio=0.5, max_cycles=config['max_cycles'], continuous_actions=config['continuous'])
        obs_shape = len(simple_spread.observation_space(simple_spread.possible_agents[0]).sample())
        num_actions = simple_spread.action_space(simple_spread.possible_agents[0]).n
        return simple_spread, obs_shape, num_actions


if __name__ == "__main__":
    """ALGO PARAMS"""
    config = {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'env_name': "simple_spread",
        'obs_shape': None,
        'num_actions': None,
        'continuous': False,
        'n_agents': 3,
        'ent_coef': 0.1,
        'vf_coef': 0.1,
        'clip_coef': 0.1,
        'gamma': 0.99,
        'max_cycles': 32,
        'total_episodes': 100000,
        'lr': 0.0001
    }

    """ ENV SETUP """
    env, obs_shape, num_actions = create_env(config)
    config['obs_shape'] = obs_shape
    config['num_actions'] = num_actions

    """ ALGO SETUP """
    ippo = IPPO(env, config)
    mappo = MAPPO(env, config)
    training(config, algo_list=[mappo])
    