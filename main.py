import torch
import numpy as np
import matplotlib.pyplot as plt
import multiprocess as mp
from pipline.train import training
from algos.ppo import PPO
from algos.ippo import IPPO
from algos.mappo import MAPPO
from algos.mtppo_copy import MTPPO
from algos.sle_ppo import SLE_MTPPO
from algos.mtsac import MultiTaskSAC

import metaworld
import random

#mt = metaworld.MT1('reach-v2', seed=0) # Construct the benchmark, sampling tasks
mt = metaworld.MT10() # Construct the benchmark, sampling tasks

def create_metaworld(seed):
    training_envs = []
    for name, env_cls in mt.train_classes.items():
        env = env_cls()
        task = random.choice([task for task in mt.train_tasks
                                if task.env_name == name])
        env.set_task(task)
        env.seed(seed)
        training_envs.append(env)
    return training_envs

def run_seeds(seeds):
        pool = mp.Pool()
        process_inputs = [(config, seeds[i]) for i in range(len(seeds))]
        results = pool.starmap(training, process_inputs)
        pool.close()
        pool.join()

        seeds_episodic_x = [res[0] for res in results]  # receive from multi-process
        seeds_episodic_return = [res[1] for res in results]  # receive from multi-process
        seeds_episodic_x_eval = [res[2] for res in results]  # receive from multi-process
        seeds_episodic_return_eval = [res[3] for res in results]  # receive from multi-process

        x = seeds_episodic_x[0]
        y = np.mean(np.asarray(seeds_episodic_return), axis=0)

        plt.plot(x, y)
        plt.show()

class MultiTaskEnv():
    def __init__(self, seed):
        self.tasks = [create_metaworld(seed)[i] for i in [0, 5, 8]] # [task1, task2]
        self.current_task = None
        self.observation_space = self.tasks[0].observation_space
        self.action_space = self.tasks[0].action_space
        self.seed = seed

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
        'pop_size': 3,
        'ent_coef': 0.01,
        'vf_coef': 0.1,
        'lr_clip_range': 0.2,
        'discount': 0.99,
        'gae_lambda': 0.97,
        'batch_size': 100000,
        'max_path_length': 500,
        'min_batch': 256,
        'epoch_merging': 4,
        'epoch_finetune': 8,
        'epoch_opt': 16,
        'total_episodes': 100,
        'hidden_size': 512,
        'lr': 0.001
    }

    config_mtsac = {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'hidden_dim': (400, 400),  # from hidden_sizes
        'discount': 0.99,  # discount
        'tau': 5e-3,  # target_update_tau
        'alpha': 0.2,  # entropy regularization
        'buffer_capacity': 50000,  # or larger based on your needs
        'batch_size': 5000,
        'policy_lr': 3e-4,  # policy_lr
        'qf_lr': 3e-4,  # qf_lr
        'min_std': -20,  # min_std
        'max_std':  2,  # max_std
        'gradient_steps_per_itr': 500,  # gradient_steps_per_itr
        'epoch_opt': 200,
        'total_episodes': 500,
        'min_buffer_size': 1500,  # min_buffer_size
        'use_automatic_entropy_tuning': True,  # use_automatic_entropy_tuning
        'max_path_length': 500
    }

        
    


    """ ENV SETUP """
    multi_task_env_0 = MultiTaskEnv(0)
    multi_task_env_42 = MultiTaskEnv(42)
    multi_task_env_100 = MultiTaskEnv(100)
    print(multi_task_env_0.tasks)

    """ SLEPPO SETUP """
    sle = SLE_MTPPO(multi_task_env_0, config)

    """ MTPPO SETUP """
    mtppo1 = MTPPO(multi_task_env_0, config)
    mtppo2 = MTPPO(multi_task_env_42, config)
    mtppo3 = MTPPO(multi_task_env_100, config)
    seeds_ppo = [mtppo1, mtppo2, mtppo3]

    """ MTSAC SETUP """
    mtsac1 = MultiTaskSAC(multi_task_env_0, config_mtsac)
    mtsac2 = MultiTaskSAC(multi_task_env_42, config_mtsac)
    mtsac3 = MultiTaskSAC(multi_task_env_100, config_mtsac)
    seeds_sac = [mtsac1, mtsac2, mtsac3]
    ''''''
    

    #run_seeds(seeds_ppo)
    #training(config_mtsac, mtsac2)
    training(config, mtppo3)
    #training(config, sle)
    