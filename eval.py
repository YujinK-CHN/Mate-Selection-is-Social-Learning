
import torch
from processing.batching import batchify, batchify_obs, unbatchify

from policies.hier_policy_prompt import HierPolicy_prompt
from policies.independent_policy import IndependentPolicy
from policies.centralized_policy import CentralizedPolicy
from policies.multitask_policy import MultiTaskPolicy

import gymnasium as gym

import numpy as np

# ,render_mode="human"
def create_multitask_env():
    env1 = gym.make("Walker2d-v5", render_mode="human")
    env2 = gym.make("HalfCheetah-v5", render_mode="human")
    return [env1] #, env2]

class MultiTaskEnv(gym.Env):
    def __init__(self, tasks):
        self.tasks = tasks # [task1, task2]
        self.current_task = None
        self.observation_space = tasks[0].observation_space
        self.action_space = tasks[0].action_space

    def reset(self):
        self.current_task = self.select_task()
        return self.current_task.reset(seed=0)

    def select_task(self):
        # You can implement random task selection, cyclic switching, or other strategies
        return self.tasks[np.random.randint(len(self.tasks))]

    def step(self, action):
        return self.current_task.step(action)

    def render(self, mode='human'):
        self.current_task.render(mode)


def run_trained_model(env, model, config):


    with torch.no_grad():
        # render 5 episodes out
        for episode in range(5):
            next_obs, infos = env.reset()
            obs = torch.FloatTensor(next_obs).to(config['device'])
            terms = False
            truncs = False
            while not terms and not truncs:
                actions = model.run(obs, 0)
                obs, rewards, terms, truncs, infos = env.step(actions.cpu().numpy())
                obs = torch.FloatTensor(obs).to(config['device'])
                terms = terms
                truncs = truncs
                print(rewards)


"""ALGO PARAMS"""
config = {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'obs_shape': None,
        'num_actions': None,
        'continuous': True,
        'pop_size': 1,
        'max_cycles': 32,
        'batch_size': 4,
    }

tasks = create_multitask_env()
multi_task_env = MultiTaskEnv(tasks)


model = MultiTaskPolicy(
            pop_size = config['pop_size'], 
            env = multi_task_env,
            num_tasks = len(multi_task_env.tasks),
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
'''
model.load_state_dict(torch.load('./models/walker_1_32_10000_0.0001.pt'))
model.eval()
model = model.to(config['device'])
run_trained_model(multi_task_env, model, config)


