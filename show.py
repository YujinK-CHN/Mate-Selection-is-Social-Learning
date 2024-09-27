import torch
from processing.batching import batchify, batchify_obs, unbatchify

from policies.hier_policy_prompt import HierPolicy_prompt
from policies.independent_policy import IndependentPolicy
from policies.centralized_policy import CentralizedPolicy
from policies.multitask_policy import MultiTaskPolicy
import matplotlib.pyplot as plt
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


x = [i for i in range(200)]
MT = 3
POP = 3
seed = 861
date = '2024-09-26'
eval_returns = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_30000_100_{date}/{seed}/eval_returns.npy")
eval_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_30000_100_{date}/{seed}/eval_sr.npy")
eval_tasks_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_30000_100_{date}/{seed}/eval_tasks_sr.npy")
training_returns = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_30000_100_{date}/{seed}/training_returns.npy")
training_tasks_return = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_30000_100_{date}/{seed}/training_tasks_return.npy")
training_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_30000_100_{date}/{seed}/training_sr.npy")
training_tasks_sr = np.load(f"logs/sle-mtppo_{MT}tasks_3agents_30000_100_{date}/{seed}/training_tasks_sr.npy")

print(eval_returns)
print(eval_sr)
print(eval_tasks_sr)
print(training_returns)
print(training_tasks_return)
print(training_sr)
print(training_tasks_sr)
'''
plt.plot(x, eval_sr)
plt.title(f"Episode returns (train and eval) for seed {seed}")
plt.xlabel("Episodes")
plt.ylabel("Success Rate")
plt.show()

'''


