import numpy as np
import torch
import torch.nn as nn
from itertools import chain
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal

class MultiTaskPolicy(nn.Module):
    def __init__(self, env, num_tasks, hidden_size, continuous, device):
        super(MultiTaskPolicy, self).__init__()
        self.env = env
        self.continuous = continuous
        self.device = device
        self.log_std = nn.Parameter(torch.full((env.action_space.shape[0],), 1.0))

        self.shared_layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0]+num_tasks, hidden_size),
            #nn.Tanh(),
            #nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, env.action_space.shape[0]),
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
            nn.Linear(env.observation_space.shape[0]+num_tasks, hidden_size),
            #nn.Tanh(),
            #nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        

    def act(self, x):
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
    
    def evaluate(self, x, actions):
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

    def gate_parameters(self):
        return self.shared_layers[2].parameters()

    def non_gate_parameters(self):
        return chain(self.shared_layers[0].parameters(), self.shared_layers[-1].parameters())


##########################################



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(400, 400), hidden_nonlinearity=nn.ReLU):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(hidden_nonlinearity())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class TaskConditionedNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(400, 400), hidden_nonlinearity=nn.ReLU):
        super(TaskConditionedNetwork, self).__init__()
        self.network = MLP(input_dim, output_dim, hidden_sizes, hidden_nonlinearity)

    def forward(self, state, task_embedding):
        x = torch.cat([state, task_embedding], dim=-1)
        return self.network(x)

class TaskConditionedPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_tasks, hidden_sizes=(400, 400), hidden_nonlinearity=nn.ReLU, min_log_std=-20, max_log_std=2):
        super(TaskConditionedPolicyNetwork, self).__init__()
        self.num_tasks = num_tasks
        self.network = TaskConditionedNetwork(state_dim + num_tasks, hidden_sizes[-1], hidden_sizes, hidden_nonlinearity)
        self.mean_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, state, task_embedding):
        x = self.network(state, task_embedding)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = log_std.exp()
        return mean, std

    def sample(self, state, task_embedding):
        mean, std = self.forward(state, task_embedding)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # Reparameterization trick
        action = torch.tanh(z)  # Apply tanh to bound actions
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob