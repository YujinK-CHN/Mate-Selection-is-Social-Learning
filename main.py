import gymnasium as gym
import random
import torch
import numpy as np
import multiprocess as mp
import matplotlib.pyplot as plt
from pipline.train import training
from algos.ppo import PPO
from algos.ippo import IPPO
from algos.mappo import MAPPO
from algos.gippo import GIPPO
from algos.mtppo import MTPPO
from algos.sle_ppo import SLE_MTPPO
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
        'pop_size': 8,
        'ent_coef': 5e-3,
        'vf_coef': 0.1,
        'lr_clip_range': 0.2,
        'discount': 0.99,
        'gae_lambda': 0.97,
        'batch_size': 5000,
        'max_path_length': 500,
        'min_batch': 32,
        'epoch_opt': 64,
        'total_episodes': 2000,
        'hidden_size': 128,
        'lr': 0.0005
    }

    """ ENV SETUP """
    print(create_multitask_env())
    multi_task_env = MultiTaskEnv(create_multitask_env())

    """ ALGO SETUP """
    sle = SLE_MTPPO(multi_task_env, config)
    mtppo1 = MTPPO(multi_task_env, 0, config)
    mtppo2 = MTPPO(multi_task_env, 42, config)
    mtppo3 = MTPPO(multi_task_env, 100, config)
    seeds = [mtppo1, mtppo2, mtppo3]

    try:
        pool = mp.Pool()
        process_inputs = [(config, seeds[i]) for i in range(3)]
        results = pool.starmap(training, process_inputs)
        pool.close()  # Close the pool to prevent new tasks from being submitted
        pool.join()   # Wait for all worker processes to finish

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
    