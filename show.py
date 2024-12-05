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

MT = 10
POP = 3
###### SLE
seed = []
date = []
batch_merging = []
batch_finetune = []
seeds_sr_eval_sle = []
seeds_sr_sle = []
runtimes_sle = []
mean_episodic_runtimes_sle = []
max_sr_sle = []
for i in range(len(seed)):
    eval_returns = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/eval_returns.npy")
    eval_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/eval_sr.npy")
    eval_tasks_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/eval_tasks_sr.npy")
    training_returns = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_returns.npy")
    training_tasks_return = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_tasks_return.npy")
    training_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_sr.npy")
    training_tasks_sr = np.load(f"logs/sle-mtppo_{MT}tasks_3agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_tasks_sr.npy")
    runtimes = np.load(f"logs/sle-mtppo_{MT}tasks_3agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/runtimes.npy")

    seeds_sr_eval_sle.append(np.max(eval_sr, axis=-1))
    seeds_sr_sle.append(np.max(training_sr, axis=-1))
    runtimes_sle.append(np.sum(runtimes))
    mean_episodic_runtimes_sle.append(np.mean(runtimes))
    seed_max_sr_sle = np.max(training_tasks_sr, axis=0)[0]
    max_sr_sle.append(seed_max_sr_sle)

seed2 = []
date2 = []
batch_merging2 = []
batch_finetune2 = []
seeds_sr_eval_sle2 = []
seeds_sr_sle2 = []
runtimes_sle2 = []
mean_episodic_runtimes_sle2 = []
max_sr_sle2 = []
for i in range(len(seed2)):
    eval_returns = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/eval_returns.npy")
    eval_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/eval_sr.npy")
    eval_tasks_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/eval_tasks_sr.npy")
    training_returns = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_returns.npy")
    training_tasks_return = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_tasks_return.npy")
    training_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_sr.npy")
    training_tasks_sr = np.load(f"logs/sle-mtppo_{MT}tasks_3agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_tasks_sr.npy")
    runtimes = np.load(f"logs/sle-mtppo_{MT}tasks_3agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/runtimes.npy")

    seeds_sr_eval_sle2.append(np.max(eval_sr, axis=-1))
    seeds_sr_sle2.append(np.max(training_sr, axis=-1))
    runtimes_sle2.append(np.sum(runtimes))
    mean_episodic_runtimes_sle2.append(np.mean(runtimes))
    seed_max_sr_sle2 = np.max(training_tasks_sr, axis=0)[0]
    max_sr_sle2.append(seed_max_sr_sle2)

seed3 = []
date3 = []
batch_merging3 = []
batch_finetune3 = []
seeds_sr_eval_sle3 = []
seeds_sr_sle3 = []
runtimes_sle3 = []
mean_episodic_runtimes_sle3 = []
max_sr_sle3 = []
for i in range(len(seed3)):
    eval_returns = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/eval_returns.npy")
    eval_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/eval_sr.npy")
    eval_tasks_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/eval_tasks_sr.npy")
    training_returns = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_returns.npy")
    training_tasks_return = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_tasks_return.npy")
    training_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_sr.npy")
    training_tasks_sr = np.load(f"logs/sle-mtppo_{MT}tasks_3agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_tasks_sr.npy")
    runtimes = np.load(f"logs/sle-mtppo_{MT}tasks_3agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/runtimes.npy")

    seeds_sr_eval_sle3.append(np.max(eval_sr, axis=-1))
    seeds_sr_sle3.append(np.max(training_sr, axis=-1))
    runtimes_sle3.append(np.sum(runtimes))
    mean_episodic_runtimes_sle3.append(np.mean(runtimes))
    seed_max_sr_sle3 = np.max(training_tasks_sr, axis=0)[0]
    max_sr_sle3.append(seed_max_sr_sle3)

###### SLE-entropy
seed11 = []
date11 = []
batch_merging11 = []
batch_finetune11 = []
seeds_sr_eval_sle11 = []
seeds_sr_sle11 = []
runtimes_sle11 = []
mean_episodic_runtimes_sle11 = []
max_sr_sle11 = []
for i in range(len(seed11)):
    eval_returns = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/eval_returns.npy")
    eval_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/eval_sr.npy")
    eval_tasks_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/eval_tasks_sr.npy")
    training_returns = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_returns.npy")
    training_tasks_return = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_tasks_return.npy")
    training_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_sr.npy")
    training_tasks_sr = np.load(f"logs/sle-mtppo_{MT}tasks_3agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_tasks_sr.npy")
    runtimes = np.load(f"logs/sle-mtppo_{MT}tasks_3agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/runtimes.npy")

    seeds_sr_eval_sle11.append(np.max(eval_sr, axis=-1))
    seeds_sr_sle11.append(np.max(training_sr, axis=-1))
    runtimes_sle11.append(np.sum(runtimes))
    mean_episodic_runtimes_sle11.append(np.mean(runtimes))
    seed_max_sr_sle1 = np.max(training_tasks_sr, axis=0)[0]
    max_sr_sle11.append(seed_max_sr_sle1)

seed12 = []
date12 = []
batch_merging12 = []
batch_finetune12 = []
seeds_sr_eval_sle12 = []
seeds_sr_sle12 = []
runtimes_sle12 = []
mean_episodic_runtimes_sle12 = []
max_sr_sle12 = []
for i in range(len(seed12)):
    eval_returns = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/eval_returns.npy")
    eval_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/eval_sr.npy")
    eval_tasks_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/eval_tasks_sr.npy")
    training_returns = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_returns.npy")
    training_tasks_return = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_tasks_return.npy")
    training_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_sr.npy")
    training_tasks_sr = np.load(f"logs/sle-mtppo_{MT}tasks_3agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_tasks_sr.npy")
    runtimes = np.load(f"logs/sle-mtppo_{MT}tasks_3agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/runtimes.npy")

    seeds_sr_eval_sle12.append(np.max(eval_sr, axis=-1))
    seeds_sr_sle12.append(np.max(training_sr, axis=-1))
    runtimes_sle12.append(np.sum(runtimes))
    mean_episodic_runtimes_sle12.append(np.mean(runtimes))
    seed_max_sr_sle12 = np.max(training_tasks_sr, axis=0)[0]
    max_sr_sle12.append(seed_max_sr_sle12)

seed13 = []
date13 = []
batch_merging13 = []
batch_finetune13 = []
seeds_sr_eval_sle13 = []
seeds_sr_sle13 = []
runtimes_sle13 = []
mean_episodic_runtimes_sle13 = []
max_sr_sle13 = []
for i in range(len(seed13)):
    eval_returns = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/eval_returns.npy")
    eval_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/eval_sr.npy")
    eval_tasks_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/eval_tasks_sr.npy")
    training_returns = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_returns.npy")
    training_tasks_return = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_tasks_return.npy")
    training_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_sr.npy")
    training_tasks_sr = np.load(f"logs/sle-mtppo_{MT}tasks_3agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/training_tasks_sr.npy")
    runtimes = np.load(f"logs/sle-mtppo_{MT}tasks_3agents_{batch_merging[i]}_{batch_finetune[i]}_200_{date[i]}/{seed[i]}(done)/runtimes.npy")

    seeds_sr_eval_sle13.append(np.max(eval_sr, axis=-1))
    seeds_sr_sle13.append(np.max(training_sr, axis=-1))
    runtimes_sle13.append(np.sum(runtimes))
    mean_episodic_runtimes_sle13.append(np.mean(runtimes))
    seed_max_sr_sle13 = np.max(training_tasks_sr, axis=0)[0]
    max_sr_sle13.append(seed_max_sr_sle13)

###### MTPPO
seed_mtppo = [861]
date_mtppo = ['2024-10-31']
batch_size = 300000
seeds_sr_eval = []
seeds_sr = []
runtimes_mtppo = []
mean_episodic_runtimes_mtppo = []
max_sr_mtppo = []
for i in range(len(seed_mtppo)):
    eval_returns = np.load(f"logs/mtppo_{MT}_{batch_size}_16_200_{date_mtppo[i]}/{seed_mtppo[i]}(done)/eval_returns_{seed_mtppo[i]}.npy")
    train_returns = np.load(f"logs/mtppo_{MT}_{batch_size}_16_200_{date_mtppo[i]}/{seed_mtppo[i]}(done)/train_returns_{seed_mtppo[i]}.npy")
    sr_eval = np.load(f"logs/mtppo_{MT}_{batch_size}_16_200_{date_mtppo[i]}/{seed_mtppo[i]}(done)/sr_eval_{seed_mtppo[i]}.npy")
    sr = np.load(f"logs/mtppo_{MT}_{batch_size}_16_200_{date_mtppo[i]}/{seed_mtppo[i]}(done)/sr_{seed_mtppo[i]}.npy")
    tasks_sr = np.load(f"logs/mtppo_{MT}_{batch_size}_16_200_{date_mtppo[i]}/{seed_mtppo[i]}(done)/tasks_sr_{seed_mtppo[i]}.npy")
    runtimes = np.load(f"logs/mtppo_{MT}_{batch_size}_16_200_{date_mtppo[i]}/{seed_mtppo[i]}(done)/runtimes_{seed_mtppo[i]}.npy")

    seeds_sr_eval.append(sr_eval)
    seeds_sr.append(sr)
    runtimes_mtppo.append(np.sum(runtimes))
    mean_episodic_runtimes_mtppo.append(np.mean(runtimes))
    seed_max_sr_mtppo = np.max(tasks_sr, axis=0)
    max_sr_mtppo.append(seed_max_sr_mtppo)


def plot_general_performance(seed_indices):
    # Prepare the data for model 1 (200 epochs)
    print(np.array(seeds_sr_eval_sle)[seed_indices].shape)
    print(np.array(seed)[seed_indices])
    epochs_model_sle = np.tile(np.arange(0, 200), (len(seed_indices), 1))
    df_model_sle = pd.DataFrame({
        'epoch': epochs_model_sle.flatten(),
        'metric': np.array(seeds_sr_eval_sle)[seed_indices].flatten(),
        'seed': np.repeat(np.array(seed)[seed_indices], 200),
        'model': 'SLE (ours)'
    })

    # Prepare the data for model 2 (200 epochs)
    epochs_model_mtppo = np.tile(np.arange(0, 200), (len(seed_indices), 1))
    df_model_mtppo = pd.DataFrame({
        'epoch': epochs_model_mtppo.flatten(),
        'metric': np.array(seeds_sr)[seed_indices].flatten(),
        'seed': np.repeat(np.array(seed_mtppo)[seed_indices], 200),
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
    plt.title(f'SLE (Ours) vs MTPPO Results (n={len(seed_indices)} seeds {np.array(seed)[seed_indices]})')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rates')
    plt.legend(title='Model')
    #plt.show()

def plot_for_model(seed_list, results, total_episodes, algo_name):
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

seed_indices = [0]
seed_indices_sle = [0, 1, 2, 3]
seed_indices_mtppo = [0]
plot_general_performance(seed_indices)
plot_for_model(np.array(seed)[seed_indices_sle], np.array(seeds_sr_sle)[seed_indices_sle], 200, 'SLE(ours)')
plot_for_model(np.array(seed)[seed_indices_mtppo], np.array(seeds_sr)[seed_indices_mtppo], 200, 'MTPPO')
plt.show()

print('Mean episodic runtimes (per seed) for SLE: ', np.array(mean_episodic_runtimes_sle)[seed_indices_sle])
print('Avg episodic runtimes (all seeds) for SLE: ', np.array(mean_episodic_runtimes_sle)[seed_indices_sle].mean(axis=0))
print('Mean episodic runtimes (per seed) for MTPPO: ', np.array(mean_episodic_runtimes_mtppo)[seed_indices_mtppo])
print('Avg episodic runtimes (all seeds) for MTPPO: ', np.array(mean_episodic_runtimes_mtppo)[seed_indices_mtppo].mean(axis=0))
print('Total runtimes (per seed) for SLE: ', np.array(runtimes_sle)[seed_indices_sle])
print('Total runtimes (per seed) for MTPPO: ', np.array(runtimes_mtppo)[seed_indices_mtppo])
print('Maximum task success rate ever (per seed) for SLE: \n', np.round(np.array(max_sr_sle)[seed_indices_sle], 2))
print('Maximum task success rate ever (per seed) for MTPPO: \n', np.round(np.array(max_sr_mtppo)[seed_indices_mtppo], 2))
print(f'Avg maximum task success rate ever (all seeds) for SLE: \n {np.round(np.array(max_sr_sle)[seed_indices_sle].mean(axis=0), 2)}')
print(f'Avg maximum task success rate ever (all seeds) for MTPPO: \n {np.round(np.array(max_sr_mtppo)[seed_indices_mtppo].mean(axis=0), 2)}')