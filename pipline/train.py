import numpy as np
import os
from datetime import date
def training(config, algo):
    
    if algo.name == 'mtppo':
        x, y, x_eval, y_eval, sr, tasks_sr = algo.train()

        # logging
        path_to_exp = f"./logs/{algo.name}_{algo.num_tasks}_{config['batch_size']}_{config['epoch_opt']}_{config['total_episodes']}_{date.today()}"
        os.makedirs(path_to_exp, exist_ok=True)
        os.makedirs(f"{path_to_exp}/seed{algo.seed}", exist_ok=True)
        algo.save(f"{path_to_exp}/seed{algo.seed}/seed{algo.seed}.pt")
        np.save(f"{path_to_exp}/seed{algo.seed}/train_returns_seed{algo.seed}.npy", np.array(y))
        np.save(f"{path_to_exp}/seed{algo.seed}/eval_returns_seed{algo.seed}.npy", np.array(y_eval))
        np.save(f"{path_to_exp}/seed{algo.seed}/sr_seed{algo.seed}.npy", np.array(sr))
        np.save(f"{path_to_exp}/seed{algo.seed}/tasks_sr_seed{algo.seed}.npy", np.array(tasks_sr))
        return x, y, x_eval, y_eval

    if algo.name == 'sle-mtppo':
        x, y, sr, y_pop, fitness_pop, sr_pop, gen_mates = algo.evolve()
        path_to_exp = f"./logs/{algo.name}_{algo.num_tasks}_{config['batch_size']}_{config['epoch_opt']}_{config['total_episodes']}_{date.today()}"
        os.makedirs(path_to_exp, exist_ok=True)
        np.save(f"{path_to_exp}/algo_returns.npy", np.array(y))
        np.save(f"{path_to_exp}/algo_sr.npy", np.array(sr))
        np.save(f"{path_to_exp}/pop_returns.npy", np.array(y_pop))
        np.save(f"{path_to_exp}/pop_sr.npy", np.array(sr_pop))
        np.save(f"{path_to_exp}/pop_fitness.npy", np.array(fitness_pop))
        np.save(f"{path_to_exp}/gen_mates.npy", np.array(gen_mates))

    if algo.name == 'mtsac':
        x, y = algo.train()
        algo.save(f"./models/{config['hidden_size']}_{config['batch_size']}_{config['epoch_opt']}_{config['total_episodes']}_seed{algo.seed}.pt")

    
