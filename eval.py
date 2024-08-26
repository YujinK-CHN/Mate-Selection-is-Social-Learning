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
                task_id = env.tasks.index(env.current_task)
                terms = False
                truncs = False
                while not terms and not truncs:
                    # rollover the observation 
                    #obs = batchify_obs(next_obs, self.device)
                    obs = torch.FloatTensor(next_obs).to(config['device'])

                    # get actions from skills
                    actions, logprobs, entropy, values = model.act(obs, task_id)

                    # execute the environment and log data
                    next_obs, rewards, terms, truncs, infos = env.step(actions.cpu().numpy())
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
model.load_state_dict(torch.load('./models/mt_1_64_100000_1e-4_old.pt'))
model.eval()
run_trained_model(multi_task_env, model, config)




