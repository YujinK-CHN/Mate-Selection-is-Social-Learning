import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision.ops import MLP

class IndependentPolicy(nn.Module):
    def __init__(self, n_agents, input_dim, output_dim, device, hidden_dim=(32, 32), continuous=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents
        self.continuous = continuous
        self.device = device
        self.action_var = torch.full((self.output_dim,), 0.6*0.6)

        self.mean = nn.ModuleList(
            [
                MLP(
                    in_channels = self.input_dim,
                    hidden_channels = [dim for dim in self.hidden_dim] + [self.output_dim]
                )
                for _ in range(self.n_agents)
            ]
        )

        self.critic = nn.ModuleList(
            [
                self._layer_init(nn.Linear(self.input_dim, 1))
                for _ in range(self.n_agents)
            ]
        )

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):

        values = torch.stack(
            [
                critic(x[i, :])
                for i, critic in enumerate(self.critic)
            ],
            dim=-2,
        )
        return values
    
    def act(self, x):

        means = torch.stack(
            [
                actor(x[i, :])
                for i, actor in enumerate(self.mean)
            ],
            dim=-2,
        )

        if self.continuous == False:
            probs = Categorical(logits=means)
        else:
            if torch.isnan(torch.sum(means)):
                means = torch.zeros(means.shape).to(self.device)
            cov_matrix = torch.diag(self.action_var).unsqueeze(dim=0).to(self.device)
            probs = MultivariateNormal(means, cov_matrix)
        
        actions = torch.tanh(probs.sample())

        values = torch.stack(
            [
                critic(x[i, :])
                for i, critic in enumerate(self.critic)
            ],
            dim=-2,
        )

        entropy = torch.fmax(probs.entropy(), torch.full(probs.entropy().shape, 1e-6).to(self.device))

        return actions, probs.log_prob(actions), entropy, values
    
    def evaluate(self, x, actions):

        means = torch.stack(
            [
                actor(x[i, :])
                for i, actor in enumerate(self.mean)
            ],
            dim=-2,
        )

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

        values = torch.stack(
            [
                critic(x[i, :])
                for i, critic in enumerate(self.critic)
            ],
            dim=-2,
        )

        entropy = torch.fmax(probs.entropy(), torch.full(probs.entropy().shape, 1e-6).to(self.device))

        return actions, probs.log_prob(actions), entropy, values
    
    def run(self, obs):
        actions, _, _, _ = self.get_action_and_value(obs)
        return actions
    

    