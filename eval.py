
import torch
from processing.batching import batchify, batchify_obs, unbatchify

from pettingzoo.sisl import waterworld_v4, multiwalker_v9
from policies.hier_policy_prompt import HierPolicy_prompt
from policies.independent_policy import IndependentPolicy
from policies.centralized_policy import CentralizedPolicy

def create_env(config):
    if config['env_name'] == 'multiwalker':
        multiwalker = multiwalker_v9.parallel_env(render_mode="human", n_walkers=config['n_agents'], position_noise=1e-3, angle_noise=1e-3, forward_reward=1.0, terminate_reward=-100.0, fall_reward=-10.0, shared_reward=True, \
                                                terminate_on_fall=True, remove_on_fall=True, terrain_length=200, max_cycles=config['max_cycles'])
        obs_shape = len(multiwalker.observation_space(multiwalker.possible_agents[0]).sample())
        num_actions = len(multiwalker.action_space(multiwalker.possible_agents[0]).sample())
        return multiwalker, obs_shape, num_actions
    if config['env_name'] == 'waterworld':
        waterworld = waterworld_v4.parallel_env(render_mode="human", n_pursuers=config['n_agents'], n_evaders=8, n_poisons=10, n_coop=2, n_sensors=20,\
                                                sensor_range=0.2,radius=0.015, obstacle_radius=0.2, n_obstacles=1,\
                                                obstacle_coord=[(0.5, 0.5)], pursuer_max_accel=0.01, evader_speed=0.01,\
                                                poison_speed=0.01, poison_reward=-1.0, food_reward=10.0, encounter_reward=0.01,\
                                                thrust_penalty=-0.5, local_ratio=1.0, speed_features=True, max_cycles=config['max_cycles'])
        obs_shape = len(waterworld.observation_space(waterworld.possible_agents[0]).sample())
        num_actions = len(waterworld.action_space(waterworld.possible_agents[0]).sample())
        return waterworld, obs_shape, num_actions

def run_trained_model(env, model):


    with torch.no_grad():
        # render 5 episodes out
        for episode in range(5):
            obs, infos = env.reset(seed=None)
            obs = batchify_obs(obs, device)
            terms = [False]
            truncs = [False]
            while not any(terms) and not any(truncs):
                actions = model.run(obs)
                obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
                obs = batchify_obs(obs, device)
                terms = [terms[a] for a in terms]
                truncs = [truncs[a] for a in truncs]


config = {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'env_name': "waterworld",
        'obs_shape': None,
        'num_actions': None,
        'continuous': True,
        'n_agents': 5,
        'n_skills': 1,
        'ent_coef': 0.1,
        'vf_coef': 0.1,
        'clip_coef': 0.1,
        'gamma': 0.99,
        'max_cycles': 512,
        'total_episodes': 128,
        'manager_lr': 0.0001,
        'skill_lr': 0.0001
    }

env, obs_shape, num_actions = create_env(config)
config['obs_shape'] = obs_shape
config['num_actions'] = num_actions
device = config['device']
# model = HierPolicy_prompt(env, config)
'''
model = IndependentPolicy(
            n_agents = config['n_agents'], 
            input_dim = config['obs_shape'],
            output_dim = config['num_actions'],
            continuous = config['continuous'],
            device = config['device']
        ).to(config['device'])
'''
model = CentralizedPolicy(
            n_agents = config['n_agents'], 
            input_dim = config['obs_shape'],
            output_dim = config['num_actions'],
            continuous = config['continuous'],
            device = config['device']
        ).to(config['device'])
model.load_state_dict(torch.load('./models/waterworld_mappo_5_1_512_128.pt'))
model.eval()
model = model.to(device)
run_trained_model(env, model)


