import torch
import numpy as np
import matplotlib.pyplot as plt
import multiprocess as mp
from pipline.train import training
from algos.ppo import PPO
from algos.ippo import IPPO
from algos.mappo import MAPPO
from algos.mtppo import MTPPO
from algos.sle_ppo import SLE_MTPPO
from algos.mtsac import MultiTaskSAC

import metaworld
import random
import time

def set_seed(seed):
    """Set the random seed for reproducibility."""
    # Set seed for PyTorch
    torch.manual_seed(seed)
    # Set seed for NumPy
    np.random.seed(seed)
    # Set seed for Python's random module
    random.seed(seed)

    # If using CUDA (for PyTorch)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For all GPUs

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed
seed_value = 0
set_seed(seed_value)

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

def random_seeds(min=0, max=1024, num_seeds = 10):
     seeds = [random.randint(min, max) for _ in range(num_seeds)]
     return seeds

def seeding(algo_name, seeds, config):
        envs = []
        for seed in seeds:
            envs.append(MultiTaskEnv(seed))
        seeds = []
        if algo_name == 'mtppo':
            for env in envs:
                seeds.append(MTPPO(env, config))
        if algo_name == 'mtsac':
            for env in envs:
                seeds.append(MultiTaskSAC(env, config))
        return seeds

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
        seeds_episodic_sr = [res[4] for res in results]  # receive from multi-process
        seeds_episodic_sr_eval = [res[5] for res in results]  # receive from multi-process

        x = seeds_episodic_x[0]
        y = np.mean(np.asarray(seeds_episodic_return), axis=0)
        x_eval = seeds_episodic_x_eval[0]
        y_eval = np.mean(np.asarray(seeds_episodic_return_eval), axis=0)
        sr = np.mean(np.asarray(seeds_episodic_sr), axis=0)
        sr_eval = np.mean(np.asarray(seeds_episodic_sr_eval), axis=0)

        plt.figure()
        plt.plot(x, y)
        plt.title("Seeds mean training return for MT3 (Metaworld num_tasks=3)")
        plt.xlabel("episodes")
        plt.ylabel("mean rewards")

        plt.figure()
        plt.plot(x_eval, y_eval)
        plt.title("Seeds mean evaluating return for MT3 (Metaworld num_tasks=3)")
        plt.xlabel("episodes")
        plt.ylabel("mean rewards")

        plt.figure()
        plt.plot(x, sr)
        plt.title("Seeds mean evaluating return for MT3 (Metaworld num_tasks=3)")
        plt.xlabel("episodes")
        plt.ylabel("sr")

        plt.figure()
        plt.plot(x_eval, sr_eval)
        plt.title("Seeds mean evaluating return for MT3 (Metaworld num_tasks=3)")
        plt.xlabel("episodes")
        plt.ylabel("sr_eval")
        plt.show()

class MultiTaskEnv():
    def __init__(self, seed):
        #self.tasks = create_metaworld(seed)
        self.tasks = create_metaworld(seed) # [create_metaworld(seed)[i] for i in [0,5,8]]
        print(self.tasks)
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
    config_mtppo = {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"), #"cuda" if torch.cuda.is_available() else "cpu" 
        'continuous': True,
        'normalize_states': True,
        'normalize_values': False,
        'normalize_rewards': True,
        'ent_coef': 5e-3,
        'vf_coef': 0.1,
        'lr_clip_range': 0.2,
        'discount': 0.99,
        'gae_lambda': 0.97,
        'batch_size': 50000,
        'max_path_length': 500,
        'min_batch': 32,
        'epoch_opt': 16,
        'total_episodes': 500,
        'hidden_size': 512,
        'lr': 0.0005
    }

    config = {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"), #"cuda" if torch.cuda.is_available() else "cpu" 
        'continuous': True,
        'normalize_states': True,
        'normalize_values': False,
        'normalize_rewards': True,
        'pop_size': 3,
        'ent_coef': 5e-3,
        'vf_coef': 0.1,
        'lr_clip_range': 0.2,
        'discount': 0.99,
        'gae_lambda': 0.97,
        'batch_size': 50000,
        'max_path_length': 500,
        'min_batch': 32, 
        'epoch_merging': 4,
        'epoch_finetune': 12,
        'total_episodes': 200,
        'hidden_size': 512,
        'lr': 0.0005
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
    # Random Seed(0): [0] 788x [1] 861 [2] 82 [3] 530 [4] 995 [5] 829
    # Random Seed(42): [0] 228 [1] 51 [2] 563 [3]  [4]  [5] 
    seeds = random_seeds()
    #seeds_ppo = seeding('mtppo', seeds, config_mtppo)
    #run_seeds(seeds_ppo)
    #training(config_mtsac, mtsac2)
    #training(config_mtppo, seeds_ppo[3])
    total_start_time = time.time()
    training(config, SLE_MTPPO(MultiTaskEnv(seeds[1]), config))
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"Total training runtime: {total_duration:.2f} seconds")
    