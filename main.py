import torch
import numpy as np
from pipline.train import training
from algos.ppo import PPO
from algos.ippo import IPPO
from algos.mappo import MAPPO
from algos.gippo import GIPPO
from algos.mtppo import MTPPO

import metaworld
import random

#mt = metaworld.MT1('reach-v2', seed=0) # Construct the benchmark, sampling tasks
mt = metaworld.MT10() # Construct the benchmark, sampling tasks

def create_metaworld():
    training_envs = []
    for name, env_cls in mt.train_classes.items():
        env = env_cls()
        task = random.choice([task for task in mt.train_tasks
                                if task.env_name == name])
        env.set_task(task)
        env.seed(0)
        training_envs.append(env)
    return training_envs


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
    print(create_metaworld())
    multi_task_env = MultiTaskEnv(create_metaworld())

    """ ALGO SETUP """
    mtppo = MTPPO(multi_task_env, config)
    training(config, algo_list=[mtppo])
    