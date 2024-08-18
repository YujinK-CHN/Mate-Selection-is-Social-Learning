import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision.ops import MLP

class IndependentPolicy(nn.Module):
    def __init__(self, n_agents, input_dim, output_dim, device, continuous=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_agents = n_agents
        self.continuous = continuous
        self.device = device

        '''
        self.actor = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.Linear(32, self.output_dim),
            nn.Softmax(),
        )
        '''

        self.pop_actors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, 32),
                nn.Linear(32, 32),
                nn.Linear(32, self.output_dim),
                nn.Softmax()
            )
            for _ in range(n_agents)
        ])
        
        self.critic = nn.Linear(self.input_dim, 1)

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):

        values = self.critic(x)
        return values

    def act(self, x):
        
        action_probs = torch.stack(
            [
                actor(x)
                for i, actor in enumerate(self.pop_actors)
            ],
            dim=-2,
        )

        if self.continuous == False:
            action_dist = Categorical(probs=action_probs)
        else:
            means = action_probs
            if torch.isnan(torch.sum(means)):
                means = torch.zeros(means.shape).to(self.device)
            cov_matrix = torch.diag(self.action_var).unsqueeze(dim=0).to(self.device)
            action_dist = MultivariateNormal(means, cov_matrix)
        
        actions = action_dist.sample()

        values = self.critic(x)

        return actions, action_dist.log_prob(actions), action_dist.entropy(), values
    
    def evaluate(self, x, actions):

        action_probs = torch.stack(
            [
                actor(x)
                for i, actor in enumerate(self.pop_actors)
            ],
            dim=-2,
        )

        if self.continuous == False:
            action_dist = Categorical(probs=action_probs)
        else:
            means = action_probs
            if torch.isnan(torch.sum(means)):
                means = torch.zeros(means.shape).to(self.device)
            action_var = self.action_var.expand_as(means)
            cov_matrix = torch.diag_embed(action_var).to(self.device)
            action_dist = MultivariateNormal(means, cov_matrix)

            if self.output_dim == 1:
                actions = actions.reshape(-1, self.output_dim)

        values = self.critic(x)

        return actions, action_dist.log_prob(actions), action_dist.entropy(), values
    
    def run(self, obs):
        actions, _, _, _ = self.act(obs)
        return actions
    