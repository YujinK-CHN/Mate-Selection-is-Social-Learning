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

        self.mean = nn.Sequential(
            self._layer_init(nn.Linear(self.input_dim, 32)),
            nn.Tanh(),
            self._layer_init(nn.Linear(32, self.output_dim)),
            nn.Tanh(),
        )
        
        self.critic = self._layer_init(nn.Linear(self.input_dim, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):

        values = self.critic(x)
        return values

    def act(self, x):

        means = self.mean(x)

        if self.continuous == False:
            probs = Categorical(logits=means)
        else:
            if torch.isnan(torch.sum(means)):
                means = torch.zeros(means.shape).to(self.device)
            cov_matrix = torch.diag(self.action_var).unsqueeze(dim=0).to(self.device)
            probs = MultivariateNormal(means, cov_matrix)
        
        actions = torch.tanh(probs.sample())

        values = self.critic(x)

        entropy = torch.fmax(probs.entropy(), torch.full(probs.entropy().shape, 1e-6).to(self.device))

        return actions, probs.log_prob(actions), entropy, values
    
    def evaluate(self, x, actions):

        means = self.mean(x)

        if self.continuous == False:
            probs = Categorical(logits=means)
        else:
            if torch.isnan(torch.sum(means)):
                means = torch.zeros(means.shape).to(self.device)
            action_var = self.action_var.expand_as(means)
            cov_matrix = torch.diag_embed(action_var).to(self.device)
            probs = MultivariateNormal(means, cov_matrix)

            if self.output_dim == 1:
                actions = actions.reshape(-1, self.output_dim)

        values = self.critic(x)

        entropy = torch.fmax(probs.entropy(), torch.full(probs.entropy().shape, 1e-6).to(self.device))

        return actions, probs.log_prob(actions), entropy, values
    
    def run(self, obs):
        actions, _, _, _ = self.act(obs)
        return actions