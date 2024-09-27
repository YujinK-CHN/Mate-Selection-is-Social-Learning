import numpy as np
import os
from datetime import date
def training(config, algo):
    
    if algo.name == 'mtppo':
        x, y, x_eval, y_eval, sr, sr_eval, tasks_sr = algo.train()

        # logging
        path_to_exp = f"./logs/{algo.name}_{algo.num_tasks}_{config['batch_size']}_{config['epoch_opt']}_{config['total_episodes']}_{date.today()}/seed{algo.seed}"
        os.makedirs(path_to_exp, exist_ok=True)
        os.makedirs(f"{path_to_exp}/", exist_ok=True)
        algo.save(f"{path_to_exp}/seed{algo.seed}.pt")
        np.save(f"{path_to_exp}/train_returns_seed{algo.seed}.npy", np.array(y))
        np.save(f"{path_to_exp}/eval_returns_seed{algo.seed}.npy", np.array(y_eval))
        np.save(f"{path_to_exp}/sr_seed{algo.seed}.npy", np.array(sr))
        np.save(f"{path_to_exp}/sr_eval_seed{algo.seed}.npy", np.array(sr_eval))
        np.save(f"{path_to_exp}/tasks_sr_seed{algo.seed}.npy", np.array(tasks_sr))
        return x, y, x_eval, y_eval, sr, sr_eval

    if algo.name == 'sle-mtppo':
        algo.evolve()
        '''
        path_to_exp = f"./logs/{algo.name}_{algo.num_tasks}tasks_{algo.pop_size}agents_{algo.batch_size}_{algo.total_episodes}_{date.today()}"
        os.makedirs(path_to_exp, exist_ok=True)
        np.save(f"{path_to_exp}/training_returns.npy", np.array(y))  # [total_epi, pop_size]
        np.save(f"{path_to_exp}/training_tasks_return.npy", np.array(y_tasks))  # [total_epi, pop_size, num_tasks]
        np.save(f"{path_to_exp}/training_sr.npy", np.array(z))  # [total_epi, pop_size]
        np.save(f"{path_to_exp}/training_tasks_sr.npy", np.array(z_tasks))  # [total_epi, pop_size, num_tasks]
        np.save(f"{path_to_exp}/eval_returns.npy", np.array(eval_fitness))  # [total_epi, pop_size, num_tasks]
        np.save(f"{path_to_exp}/eval_sr.npy", np.array(eval_sr))  # [total_epi, pop_size]
        np.save(f"{path_to_exp}/eval_tasks_sr.npy", np.array(eval_sr_tasks))  # [total_epi, pop_size, num_tasks]
        '''

    if algo.name == 'mtsac':
        x, y = algo.train()
        algo.save(f"./models/{config['hidden_size']}_{config['batch_size']}_{config['epoch_opt']}_{config['total_episodes']}_seed{algo.seed}.pt")

    
