import torch
from processing.batching import batchify, batchify_obs, unbatchify

from policies.hier_policy_prompt import HierPolicy_prompt
from policies.independent_policy import IndependentPolicy
from policies.centralized_policy import CentralizedPolicy
from policies.multitask_policy import MultiTaskPolicy
import matplotlib.pyplot as plt

import numpy as np



x = [i for i in range(200)]
MT = 3
POP = 3
seed = 861
date = '2024-09-26'
eval_returns = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_30000_100_{date}/{seed}/eval_returns.npy")
eval_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_30000_100_{date}/{seed}/eval_sr.npy")
eval_tasks_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_30000_100_{date}/{seed}/eval_tasks_sr.npy")
training_returns = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_30000_100_{date}/{seed}/training_returns.npy")
training_tasks_return = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_30000_100_{date}/{seed}/training_tasks_return.npy")
training_sr = np.load(f"logs/sle-mtppo_{MT}tasks_{POP}agents_30000_100_{date}/{seed}/training_sr.npy")
training_tasks_sr = np.load(f"logs/sle-mtppo_{MT}tasks_3agents_30000_100_{date}/{seed}/training_tasks_sr.npy")

print(eval_returns)
print(eval_sr)
print(eval_tasks_sr)
print(training_returns)
print(training_tasks_return)
print(training_sr)
print(training_tasks_sr)
'''
plt.plot(x, eval_sr)
plt.title(f"Episode returns (train and eval) for seed {seed}")
plt.xlabel("Episodes")
plt.ylabel("Success Rate")
plt.show()

'''


