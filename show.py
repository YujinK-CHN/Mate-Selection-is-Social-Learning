import torch
import pandas as pd
import seaborn as sns
from processing.batching import batchify, batchify_obs, unbatchify

from policies.hier_policy_prompt import HierPolicy_prompt
from policies.independent_policy import IndependentPolicy
from policies.centralized_policy import CentralizedPolicy
from policies.multitask_policy import MultiTaskPolicy
import matplotlib.pyplot as plt

import numpy as np
'''
data = {
    'epoch': ,  # epochs repeated for each seed
    'metric': ,  # performance metric for each seed
    'seed': ,  # different seeds
}
df = pd.DataFrame(data)
'''

MT = 3
POP = 3
###### SLE
seed = [861, 82, 530, 995, 829]
date = ['2024-09-29', '2024-10-04', '2024-10-06', '2024-10-07', '2024-10-08']
seeds_sr_eval_sle = []
seeds_sr_sle = []
for i in range(len(seed)):
    eval_returns = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_30000_100_{date[i]}/{seed[i]}(done)/eval_returns.npy")
    eval_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_30000_100_{date[i]}/{seed[i]}(done)/eval_sr.npy")
    eval_tasks_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_30000_100_{date[i]}/{seed[i]}(done)/eval_tasks_sr.npy")
    training_returns = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_30000_100_{date[i]}/{seed[i]}(done)/training_returns.npy")
    training_tasks_return = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_30000_100_{date[i]}/{seed[i]}(done)/training_tasks_return.npy")
    training_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_30000_100_{date[i]}/{seed[i]}(done)/training_sr.npy")
    training_tasks_sr = np.load(f"logs/sle-mtppo_{MT}tasks_3agents_30000_100_{date[i]}/{seed[i]}(done)/training_tasks_sr.npy")

    seeds_sr_eval_sle.append(np.max(eval_sr, axis=-1))
    seeds_sr_sle.append(np.max(training_sr, axis=-1))

###### MTPPO
seed_mtppo = [861, 82, 530, 995, 829]
date_mtppo = ['2024-10-01', '2024-10-02', '2024-10-03', '2024-10-04', '2024-10-08']
seeds_sr_eval = []
seeds_sr = []
for i in range(len(seed)):
    eval_returns = np.load(f"logs/mtppo_{MT}_30000_16_200_{date_mtppo[i]}/{seed_mtppo[i]}(done)/eval_returns_{seed_mtppo[i]}.npy")
    train_returns = np.load(f"logs/mtppo_{MT}_30000_16_200_{date_mtppo[i]}/{seed_mtppo[i]}(done)/train_returns_{seed_mtppo[i]}.npy")
    sr_eval = np.load(f"logs/mtppo_{MT}_30000_16_200_{date_mtppo[i]}/{seed_mtppo[i]}(done)/sr_eval_{seed_mtppo[i]}.npy")
    sr = np.load(f"logs/mtppo_{MT}_30000_16_200_{date_mtppo[i]}/{seed_mtppo[i]}(done)/sr_{seed_mtppo[i]}.npy")
    tasks_sr = np.load(f"logs/mtppo_{MT}_30000_16_200_{date_mtppo[i]}/{seed_mtppo[i]}(done)/tasks_sr_{seed_mtppo[i]}.npy")

    seeds_sr_eval.append(sr_eval)
    seeds_sr.append(sr)

# Prepare the data for model 1 (100 epochs)
epochs_model_sle = np.tile(np.arange(1, 101) / 100, (len(seed), 1))
df_model_sle = pd.DataFrame({
    'epoch': epochs_model_sle.flatten(),
    'metric': np.array(seeds_sr_eval_sle).flatten(),
    'seed': np.repeat(seed, 100),
    'model': 'SLE (ours)'
})

# Prepare the data for model 2 (200 epochs)
epochs_model_mtppo = np.tile(np.arange(1, 201) / 200, (len(seed_mtppo), 1))
df_model_mtppo = pd.DataFrame({
    'epoch': epochs_model_mtppo.flatten(),
    'metric': np.array(seeds_sr).flatten(),
    'seed': np.repeat(seed_mtppo, 200),
    'model': 'MTPPO'
})

# Combine both models into one DataFrame
df_combined = pd.concat([df_model_sle, df_model_mtppo])

# Plot with Seaborn (comparison between models)
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_combined,
    x='epoch',
    y='metric',
    hue='model',  # Different colors for each model
    ci='sd'  # Standard deviation as shaded area
)

# Customize the plot
plt.title(f'SLE (Ours) vs MTPPO Results (n={len(seed)} seeds)')
plt.xlabel('Episodes')
plt.ylabel('Success Rates')
plt.legend(title='Model')
#plt.show()


def plot_for_model(seed_list, results, total_episodes, algo_name):
    # Prepare the data (4 seeds, 100 epochs)
    epochs = np.tile(np.arange(1, total_episodes+1), (results.shape[0], 1))
    df_model = pd.DataFrame({
        'epoch': epochs.flatten(),
        'metric': results.flatten(),
        'seed': np.repeat(seed_list, total_episodes)
    })

    # Plot each seed's results on the same figure using Seaborn
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_model,
        x='epoch',
        y='metric',
        hue='seed',  # Different color for each seed
        palette='tab10',  # Nice color palette
        linewidth=1.5
    )

    # Customize the plot
    plt.title(f'{algo_name} Performance Across Different Seeds')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rates')
    plt.legend(title='Seeds')
    

plot_for_model(seed, np.array(seeds_sr_eval_sle), 100, 'SLE(ours)')
plot_for_model(seed, np.array(seeds_sr), 200, 'MTPPO')
plt.show()