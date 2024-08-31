import gymnasium as gym
import random
import torch
import numpy as np
from pipline.train import training
from algos.ppo import PPO
from algos.ippo import IPPO
from algos.mappo import MAPPO
from algos.gippo import GIPPO
from algos.mtppo import MTPPO

# ,render_mode="human"
def create_multitask_env():
    env1 = gym.make("Walker2d-v5")
    env2 = gym.make("HalfCheetah-v5")
    return [env1] #, env2]


class MultiTaskEnv():
    def __init__(self, tasks):
        self.tasks = tasks # [task1, task2]
        self.current_task = None
        self.observation_space = tasks[0].observation_space
        self.action_space = tasks[0].action_space

    def reset(self):
        self.current_task = self.select_task()
        return self.current_task.reset()

    def select_task(self):
        # You can implement random task selection, cyclic switching, or other strategies
        return self.tasks[np.random.randint(len(self.tasks))]

    def step(self, action):
        return self.current_task.step(action)


if __name__ == "__main__":
    """ALGO PARAMS"""
    config = {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'continuous': True,
        'pop_size': 1,
        'ent_coef': 5e-3,
        'vf_coef': 0.1,
        'lr_clip_range': 0.2,
        'discount': 0.99,
        'gae_lambda': 0.97,
        'batch_size': 100000,
        'max_path_length': 500,
        'min_batch': 32,
        'epoch_opt': 16,
        'total_episodes': 4000,
        'lr': 0.0005
    }

    """ ENV SETUP """
    print(create_multitask_env())
    multi_task_env = MultiTaskEnv(create_multitask_env())

    """ ALGO SETUP """
    mtppo = MTPPO(multi_task_env, config)
    training(config, algo_list=[mtppo])
    