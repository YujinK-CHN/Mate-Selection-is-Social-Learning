import numpy as np
from datetime import date
def training(config, algo):
    
    if algo.name == 'mtppo':
        x, y, x_eval, y_eval = algo.train()
        formatted_date = date.today().strftime("%d-%m-%Y")
        algo.save(f"./models/{config['hidden_size']}_{config['batch_size']}_{config['epoch_opt']}_{config['total_episodes']}_seed{algo.seed}.pt")
        np.save(f"./models/train_{config['batch_size']}_{config['epoch_opt']}_{config['total_episodes']}_seed{algo.seed}_{formatted_date}.npy", np.array(y))
        np.save(f"./models/eval_{config['batch_size']}_{config['epoch_opt']}_{config['total_episodes']}_seed{algo.seed}_{formatted_date}.npy", np.array(y_eval))
        return x, y, x_eval, y_eval

    if algo.name == 'sle-mtppo':
        algo.evolve()
        algo.save(f"./models/{config['hidden_size']}_{config['batch_size']}_{config['epoch_opt']}_{config['total_episodes']}_seed{algo.seed}.pt")

    if algo.name == 'mtsac':
        x, y = algo.train()
        algo.save(f"./models/{config['hidden_size']}_{config['batch_size']}_{config['epoch_opt']}_{config['total_episodes']}_seed{algo.seed}.pt")

    
