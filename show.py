import torch
from processing.batching import batchify, batchify_obs, unbatchify

from policies.hier_policy_prompt import HierPolicy_prompt
from policies.independent_policy import IndependentPolicy
from policies.centralized_policy import CentralizedPolicy
from policies.multitask_policy import MultiTaskPolicy
import matplotlib.pyplot as plt

import numpy as np


'''
x1 = [i for i in range(100)]
MT = 3
POP = 3
seed = [861, 82, 530]
date = ['2024-09-29', '2024-10-04', '2024-10-06']
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


x2 = [i for i in range(200)]
MT = 3
POP = 3
seed = [861, 82, 530]
date = ['2024-10-01', '2024-10-02', '2024-10-03']
seeds_sr_eval = []
seeds_sr = []
for i in range(len(seed)):
    eval_returns = np.load(f"logs/mtppo_{MT}_30000_16_200_{date[i]}/{seed[i]}(done)/eval_returns_{seed[i]}.npy")
    train_returns = np.load(f"logs/mtppo_{MT}_30000_16_200_{date[i]}/{seed[i]}(done)/train_returns_{seed[i]}.npy")
    sr_eval = np.load(f"logs/mtppo_{MT}_30000_16_200_{date[i]}/{seed[i]}(done)/sr_eval_{seed[i]}.npy")
    sr = np.load(f"logs/mtppo_{MT}_30000_16_200_{date[i]}/{seed[i]}(done)/sr_{seed[i]}.npy")
    tasks_sr = np.load(f"logs/mtppo_{MT}_30000_16_200_{date[i]}/{seed[i]}(done)/tasks_sr_{seed[i]}.npy")

    seeds_sr_eval.append(sr_eval)
    seeds_sr.append(sr)

mean_seeds_sr = np.array(seeds_sr).sum(0) / len(seed)
mean_seeds_sr_eval = np.array(seeds_sr_eval).sum(0) / len(seed)
mean_seeds_sr_sle = np.array(seeds_sr_sle).sum(0) / len(seed)
mean_seeds_sr_eval_sle = np.array(seeds_sr_eval_sle).sum(0) / len(seed)

plt.plot(np.array(x1)/100, mean_seeds_sr_eval_sle, color='blue', label='SLE (ours)')
plt.plot(np.array(x2)/200, mean_seeds_sr, color='green', label='MTPPO')
plt.title(f"Mean episodic returns for seeds {seed}")
plt.xlabel("Episodes")
plt.ylabel("Success Rate")
plt.legend()
plt.show()

'''
x1 = [i for i in range(100)]
x2 = [i for i in range(200)]
MT = 3
POP = 3
seed = 228 #[788, 861, 82, 530, 995, 829, 228]
date1 = '2024-10-09' #['2024-10-02', '2024-09-29', '2024-10-04', '2024-10-06', '2024-10-07', '2024-10-08', '2024-10-09']
date2 = '2024-10-09' #['2024-09-15', '2024-10-01', '2024-10-02', '2024-10-03', '2024-10-04', '2024-10-08', '2024-10-09']

#eval_returns = np.load(f"logs/mtppo_{MT}_30000_16_200_{date}/{seed}(done)/eval_returns_{seed}.npy")
#train_returns = np.load(f"logs/mtppo_{MT}_30000_16_200_{date}/{seed}(done)/train_returns_{seed}.npy")

sr_sle = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_30000_100_{date1}/{seed}(done)/training_sr.npy")
sr_eval_sle = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_30000_100_{date1}/{seed}(done)/eval_sr.npy")
sr = np.load(f"logs/mtppo_{MT}_30000_16_200_{date2}/{seed}(done)/sr_{seed}.npy")
sr_eval = np.load(f"logs/mtppo_{MT}_30000_16_200_{date2}/{seed}(done)/sr_eval_{seed}.npy")

plt.plot(np.array(x1)/100, np.max(sr_eval_sle, axis=-1), color='blue', label='SLE (ours)')
plt.plot(np.array(x2)/200, sr, color='green', label='MTPPO')
plt.title(f"Episode returns (train and eval) for seed {seed}")
plt.xlabel("Episodes")
plt.ylabel("Success Rate")
plt.show()
