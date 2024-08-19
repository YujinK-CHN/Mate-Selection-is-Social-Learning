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
        self.action_var = torch.full((self.output_dim,), 0.6*0.6)

        self.pop_actors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, 32),
                nn.Linear(32, 32),
                nn.Linear(32, self.output_dim),
                nn.Tanh(),
                # nn.Softmax(dim=-1)
            )
            for _ in range(n_agents)
        ])
        
        self.pop_critic = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, 32),
                nn.Linear(32, 32),
                nn.Linear(32, 1)
            )
            for _ in range(n_agents)
        ])

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):

        values = torch.stack(
            [
                critic(x[i, :])
                for i, critic in enumerate(self.pop_critic)
            ],
            dim=-2,
        )
        return values

    def act(self, x):
        
        action_probs = torch.stack(
            [
                actor(x[i, :])
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
        
        actions = torch.tanh(action_dist.sample())

        values = torch.stack(
            [
                critic(x[i, :])
                for i, critic in enumerate(self.pop_critic)
            ],
            dim=-2,
        )

        return actions, action_dist.log_prob(actions), action_dist.entropy(), values
    
    def evaluate(self, x, actions):

        action_probs = torch.stack(
            [
                actor(x[:, i, :])
                for i, actor in enumerate(self.pop_actors)
            ],
            dim=-2,
        )
        #print(action_probs.shape) [B, N, 2]

        if self.continuous == False:
            action_dist = Categorical(probs=action_probs)
        else:
            means = action_probs
            if torch.isnan(torch.sum(means)):
                means = torch.zeros(means.shape).to(self.device)
            action_var = self.action_var.expand_as(means)
            cov_matrix = torch.diag_embed(action_var).to(self.device)
            action_dist = MultivariateNormal(means, cov_matrix)

        values = torch.stack(
            [
                critic(x[:, i, :])
                for i, critic in enumerate(self.pop_critic)
            ],
            dim=-2,
        )

        return actions, action_dist.log_prob(actions), action_dist.entropy(), values
    
    def run(self, obs):
        actions, _, _, _ = self.act(obs)
        return actions
    