import numpy as np
import torch
import torch.nn as nn
from itertools import chain
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal

class MultiTaskPolicy(nn.Module):
    
    def __init__(self, env, num_tasks, hidden_size, continuous, device):
        super(MultiTaskPolicy, self).__init__()
        self.env = env
        self.continuous = continuous
        self.device = device
        self.log_std = nn.Parameter(torch.full((env.action_space.shape[0],), 1.0))

        self.shared_layers = nn.Sequential(
            self._layer_init(nn.Linear(env.observation_space.shape[0]+num_tasks, hidden_size)),
            nn.Tanh(),
            self._layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            self._layer_init(nn.Linear(hidden_size, env.action_space.shape[0]), std=0.01),
        )
        
        self.critic = nn.Sequential(
            self._layer_init(nn.Linear(env.observation_space.shape[0]+num_tasks, hidden_size)),
            nn.Tanh(),
            self._layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            self._layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.shared_layers(x)
        action_logstd = self.log_std.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        #clamped_diagonal = torch.clamp(self.log_std, min=0.5, max=1.5)
        #clamped_cov_matrix = torch.diag_embed(clamped_diagonal) + (torch.diag_embed(action_var) - torch.diag_embed(action_var))
        #probs = MultivariateNormal(means, clamped_cov_matrix)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.critic(x)
    
    
    def actor(self):
        return [self.log_std] + list(self.shared_layers.parameters())

    def gate_parameters(self):
        return chain(self.shared_layers[2].parameters(), self.shared_layers[5].parameters())

    def non_gate_parameters(self):
        return chain(self.log_std, self.shared_layers[0].parameters(), self.shared_layers[3].parameters(), self.shared_layers[-2].parameters())
    
    def l0_loss(self):
        return self.shared_layers[2].l0_loss() + self.shared_layers[5].l0_loss()
        

    
##########################################


    
class TaskConditionedNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(400, 400), hidden_nonlinearity=nn.ReLU):
        super(TaskConditionedNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU(),
            nn.Linear(400, output_dim)
        )

    def forward(self, x):
        x.requires_grad_(True)
        return self.network(x)

class TaskConditionedPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_tasks, hidden_sizes=(400, 400), hidden_nonlinearity=nn.ReLU, min_log_std=-20, max_log_std=2):
        super(TaskConditionedPolicyNetwork, self).__init__()
        self.num_tasks = num_tasks
        self.network = nn.Sequential(
            nn.Linear(state_dim + num_tasks, 400),
            nn.ReLU(),
            nn.Linear(400, 400)
        )
        self.mean_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, state):
        x = self.network(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # Reparameterization trick
        action = torch.tanh(z)  # Apply tanh to bound actions
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob