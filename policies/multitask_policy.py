import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal

class MultiTaskPolicy(nn.Module):
    def __init__(self, pop_size, env, num_tasks, continuous, device):
        super(MultiTaskPolicy, self).__init__()
        self.env = env
        self.pop_size = pop_size
        self.continuous = continuous
        self.device = device
        self.log_std = nn.Parameter(torch.full((env.action_space.shape[0],), 1.0))

        self.shared_layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 512),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, env.action_space.shape[0]),
            nn.Tanh(),
        )
        '''
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, env.tasks[i].action_space.shape[0]),
                nn.Tanh()
            ) 
            for i in range(num_tasks)
        ])
        '''
        self.critic = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 512),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )
        

    def act(self, x, task_id):
        means = self.shared_layers(x)
        # action_probs = self.task_heads[task_id](latent)

        if self.continuous == False:
            action_dist = Categorical(probs=means)
        else:
            clamped_diagonal = torch.clamp(self.log_std, min=0.5, max=1.5)
            clamped_cov_matrix = torch.diag_embed(clamped_diagonal) + (torch.diag(self.log_std) - torch.diag_embed(self.log_std)).to(self.device)
            action_dist = MultivariateNormal(means, clamped_cov_matrix)
        
        actions = action_dist.sample()
        values = self.critic(x)

        return actions, action_dist.log_prob(actions), action_dist.entropy(), values
    
    def evaluate(self, x, task_id, actions):
        means = self.shared_layers(x)
        #action_probs = self.task_heads[task_id](latent)

        if self.continuous == False:
            action_dist = Categorical(probs=means)
        else:
            action_var = self.log_std.expand_as(means)
            clamped_diagonal = torch.clamp(self.log_std, min=0.5, max=1.5)
            clamped_cov_matrix = torch.diag_embed(clamped_diagonal) + (torch.diag_embed(action_var) - torch.diag_embed(action_var)).to(self.device)
            action_dist = MultivariateNormal(means, clamped_cov_matrix)

        values = self.critic(x)
        # print(self.shared_layers[2].weight.grad)
        return actions, action_dist.log_prob(actions), action_dist.entropy(), values

    

