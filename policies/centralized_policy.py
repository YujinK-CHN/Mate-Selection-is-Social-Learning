import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal

class CentralizedPolicy(nn.Module):
    def __init__(self, pop_size, observation_space, action_space, device, continuous=True):
        super().__init__()
        self.input_dim = observation_space
        self.output_dim = action_space
        self.pop_size = pop_size
        self.continuous = continuous
        self.device = device
        self.action_var = torch.full((self.output_dim,), 0.6*0.6)

        if continuous == False:
            self.actor = nn.Sequential(
                nn.Linear(self.input_dim, 32),
                nn.Linear(32, 32),
                nn.Linear(32, self.output_dim),
                nn.Softmax(dim=-1),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(self.input_dim, 32),
                nn.Linear(32, 32),
                nn.Linear(32, self.output_dim),
                nn.Tanh(),
            )
        
        self.critic = nn.Sequential(
                nn.Linear(self.input_dim, 32),
                nn.Linear(32, 32),
                nn.Linear(32, 1)
            )

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):

        values = self.critic(x)
        return values

    def act(self, x):
        
        action_probs = self.actor(x)

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
    
    def evaluate(self, x, actions):

        action_probs = self.actor(x)

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