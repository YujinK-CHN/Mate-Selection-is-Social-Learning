import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal

class CentralizedPolicy(nn.Module):
    def __init__(self, n_agents, input_dim, output_dim, device, continuous=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_agents = n_agents
        self.continuous = continuous
        self.device = device
        self.action_var = torch.full((self.output_dim,), 0.6*0.6)

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
            cov_matrix = torch.diag(self.action_var).to(self.device)
            action_dist = MultivariateNormal(action_probs, cov_matrix)
        
        actions = action_dist.sample()

        values = self.critic(x)

        return actions, action_dist.log_prob(actions), action_dist.entropy(), values
    
    def evaluate(self, x, actions):

        action_probs = self.actor(x)

        if self.continuous == False:
            action_dist = Categorical(probs=action_probs)
        else:
            cov_matrix = torch.diag(self.action_var).unsqueeze(dim=0).to(self.device)
            action_dist = MultivariateNormal(action_probs, cov_matrix)

        values = self.critic(x)

        return actions, action_dist.log_prob(actions), action_dist.entropy(), values
    
    def run(self, obs):
        actions, _, _, _ = self.act(obs)
        return actions