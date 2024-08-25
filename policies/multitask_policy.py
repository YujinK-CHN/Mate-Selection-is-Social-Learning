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
        self.action_var = [torch.full((env.tasks[i].action_space.shape[0],), 0.6*0.6)  for i in range(num_tasks)]

        self.shared_layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, env.tasks[i].action_space.shape[0]),
                #nn.Softmax(dim=-1)
                nn.Tanh()
            ) 
            for i in range(num_tasks)
        ])

        self.critic = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 64),
            nn.Linear(64, 64),
            nn.Linear(64, 1)
        )
        

    def act(self, x, task_id):
        latent = self.shared_layers(x)
        action_probs = self.task_heads[task_id](latent)

        if self.continuous == False:
            action_dist = Categorical(probs=action_probs)
        else:
            if self.pop_size == 1:
                cov_matrix = torch.diag(self.action_var[task_id]).to(self.device)
            else:
                cov_matrix = torch.diag(self.action_var[task_id]).unsqueeze(dim=0).to(self.device)
            action_dist = MultivariateNormal(action_probs, cov_matrix)
        
        actions = action_dist.sample()

        values = self.critic(x)

        return actions, action_dist.log_prob(actions), action_dist.entropy(), values
    
    def evaluate(self, x, task_id, actions):
        
        
        latent = self.shared_layers(x)
        action_probs = self.task_heads[task_id](latent)

        if self.continuous == False:
            action_dist = Categorical(probs=action_probs)
        else:
            if self.pop_size == 1:
                cov_matrix = torch.diag(self.action_var[task_id]).to(self.device)
            else:
                cov_matrix = torch.diag(self.action_var[task_id]).unsqueeze(dim=0).to(self.device)
            action_dist = MultivariateNormal(action_probs, cov_matrix)

        values = self.critic(x)
        return actions, action_dist.log_prob(actions), action_dist.entropy(), values
    
    def run(self, obs):
        actions, _, _, _ = self.act(obs)
        return actions