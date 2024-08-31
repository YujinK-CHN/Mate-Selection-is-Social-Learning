import torch
import numpy as np
import matplotlib.pyplot as plt
import multiprocess as mp
from pipline.train import training
from algos.ppo import PPO
from algos.ippo import IPPO
from algos.mappo import MAPPO
from algos.gippo import GIPPO
from algos.mtppo import MTPPO
from algos.sle_ppo import SLE_MTPPO

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
        'pop_size': 8,
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
        'hidden_size': 512,
        'lr': 0.0005
    }

    """ ENV SETUP """
    print(create_metaworld())
    multi_task_env_0 = MultiTaskEnv(create_metaworld(0))
    multi_task_env_42 = MultiTaskEnv(create_metaworld(42))
    multi_task_env_100 = MultiTaskEnv(create_metaworld(100))

    """ ALGO SETUP """
    sle = SLE_MTPPO(multi_task_env_0, config)
    mtppo1 = MTPPO(multi_task_env_0, config)
    mtppo2 = MTPPO(multi_task_env_42, config)
    mtppo3 = MTPPO(multi_task_env_100, config)
    seeds = [mtppo1, mtppo2, mtppo3]

    try:
        pool = mp.Pool()
        process_inputs = [(config, seeds[i]) for i in range(len(seeds))]
        results = pool.starmap(training, process_inputs)
        pool.close()
        pool.join()

        seeds_episodic_x = [res[0] for res in results]  # receive from multi-process
        seeds_episodic_return = [res[1] for res in results]  # receive from multi-process

        x = seeds_episodic_x[0]
        y = np.mean(np.asarray(seeds_episodic_return), axis=0)

        plt.plot(x, y)
        plt.show()
    except KeyboardInterrupt:
        print("Main process interrupted, terminating workers...")
        plt.close('all')
        pool.terminate()  # Terminate all workers immediately
        pool.join()       # Wait for the workers to terminate
        print("All workers terminated.")
    finally:
        pool.close()  # Close the pool to prevent new tasks from being submitted
        pool.join()   # Wait for all worker processes to finish

    # training(config, sle)
    