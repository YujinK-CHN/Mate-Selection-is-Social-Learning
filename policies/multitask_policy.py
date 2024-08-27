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
            nn.Linear(env.observation_space.shape[0], 128),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, env.action_space.shape[0]),
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
            nn.Linear(env.observation_space.shape[0], 128),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        

    def act(self, x, task_id):
        means = self.shared_layers(x)
        # action_probs = self.task_heads[task_id](latent)

        if self.continuous == False:
            action_dist = Categorical(probs=means)
        else:
            if self.pop_size == 1:
                cov_matrix = torch.diag(self.log_std).to(self.device)
            else:
                cov_matrix = torch.diag(self.log_std).unsqueeze(dim=0).to(self.device)
            action_dist = MultivariateNormal(means, cov_matrix)
        
        actions = action_dist.sample()
        values = self.critic(x)

        return actions, action_dist.log_prob(actions), action_dist.entropy(), values
    
    def evaluate(self, x, task_id, actions):
        means = self.shared_layers(x)
        #action_probs = self.task_heads[task_id](latent)

        if self.continuous == False:
            action_dist = Categorical(probs=means)
        else:
            if self.pop_size == 1:
                action_var = self.log_std.expand_as(means)
                cov_matrix = torch.diag_embed(action_var).to(self.device)
            else:
                action_var = self.log_std.expand_as(means)
                cov_matrix = torch.diag_embed(action_var).unsqueeze(dim=0).to(self.device)
                
            action_dist = MultivariateNormal(means, cov_matrix)

        values = self.critic(x)
        # print(self.shared_layers[2].weight.grad)
        return actions, action_dist.log_prob(actions), action_dist.entropy(), values
    
    def run(self, obs, task_id):
        actions, _, _, _ = self.act(obs, task_id)
        return actions
    

