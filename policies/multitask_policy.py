import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal

class MultiTaskPolicy(nn.Module):
    def __init__(self, pop_size, observation_space, action_space, num_tasks, continuous, device):
        super(MultiTaskPolicy, self).__init__()
        self.pop_size = pop_size
        self.continuous = continuous
        self.device = device
        self.action_var = torch.full((self.output_dim,), 0.6*0.6)

        self.shared_layers = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.task_heads = nn.ModuleList([
            nn.Linear(128, action_space.n) for _ in range(num_tasks)
        ])

        self.critic = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 1)
        )
        

    def act(self, x, task_id):
        
        x = self.shared_layers(x)
        action_probs = self.task_heads[task_id](x)

        if self.continuous == False:
            action_dist = Categorical(probs=action_probs)
        else:
            if self.pop_size == 1:
                cov_matrix = torch.diag(self.action_var).to(self.device)
            else:
                cov_matrix = torch.diag(self.action_var).unsqueeze(dim=0).to(self.device)
            action_dist = MultivariateNormal(action_probs, cov_matrix)
        
        actions = action_dist.sample()

        values = self.critic(x)

        return actions, action_dist.log_prob(actions), action_dist.entropy(), values
    
    def evaluate(self, x, task_id, actions):

        x = self.shared_layers(x)
        action_probs = self.task_heads[task_id](x)

        if self.continuous == False:
            action_dist = Categorical(probs=action_probs)
        else:
            if self.pop_size == 1:
                cov_matrix = torch.diag(self.action_var).to(self.device)
            else:
                cov_matrix = torch.diag(self.action_var).unsqueeze(dim=0).to(self.device)
            action_dist = MultivariateNormal(action_probs, cov_matrix)

        values = self.critic(x)

        return actions, action_dist.log_prob(actions), action_dist.entropy(), values
    
    def run(self, obs):
        actions, _, _, _ = self.act(obs)
        return actions