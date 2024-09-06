import numpy as np
import torch
import torch.nn as nn
from itertools import chain
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal

class MultiTaskPolicy(nn.Module):
    
    def __init__(self, env, num_tasks, hidden_size, continuous, normalize_states, device):
        super(MultiTaskPolicy, self).__init__()
        self.env = env
        self.continuous = continuous
        self.normalize_states = normalize_states
        self.device = device
        self.log_std = nn.Parameter(torch.full((env.action_space.shape[0],), 1.0))

        if self.normalize_states:
            self.obs_means = [torch.zeros(env.observation_space.shape[0]+num_tasks).to(device) for _ in range(len(env.tasks))]
            self.obs_stds = [torch.ones(env.observation_space.shape[0]+num_tasks).to(device) for _ in range(len(env.tasks))]
            self.counts = [1e-8 for _ in range(len(env.tasks))]  # Avoid division by zero initially
        '''
        self.embedding = nn.Sequential(
                nn.Embedding(num_embeddings = num_tasks, embedding_dim = hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
        '''
        self.shared_layers = nn.Sequential(
            self._layer_init(nn.Linear(env.observation_space.shape[0]+num_tasks, hidden_size)),
            nn.Tanh(),
            self._layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            self._layer_init(nn.Linear(hidden_size, env.action_space.shape[0])),
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
            self._layer_init(nn.Linear(env.observation_space.shape[0]+num_tasks, hidden_size)),
            nn.Tanh(),
            self._layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            self._layer_init(nn.Linear(hidden_size, 1))
        )

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def act(self, x, task_id):
        means = self.shared_layers(x)
        # action_probs = self.task_heads[task_id](latent)

        if self.continuous == False:
            action_dist = Categorical(probs=means)
        else:
            clamped_diagonal = torch.clamp(self.log_std, min=0.5, max=1.5)
            clamped_cov_matrix = torch.diag_embed(clamped_diagonal) + (torch.diag(self.log_std) - torch.diag_embed(self.log_std)).to(self.device)
            action_dist = MultivariateNormal(means, clamped_cov_matrix)
        
        actions = action_dist.sample()

        if self.normalize_states:
            x = self.normalize_states(x, self.obs_means[task_id], self.obs_stds[task_id])
        values = self.critic(x)

        return actions, action_dist.log_prob(actions), action_dist.entropy(), values
    
    def evaluate(self, x, actions, task_id):
        means = self.shared_layers(x)
        #action_probs = self.task_heads[task_id](latent)

        if self.continuous == False:
            action_dist = Categorical(probs=means)
        else:
            action_var = self.log_std.expand_as(means)
            clamped_diagonal = torch.clamp(self.log_std, min=0.5, max=1.5)
            clamped_cov_matrix = torch.diag_embed(clamped_diagonal) + (torch.diag_embed(action_var) - torch.diag_embed(action_var)).to(self.device)
            action_dist = MultivariateNormal(means, clamped_cov_matrix)

        if self.normalize_states:
            x = self.normalize_state(x, self.obs_means[task_id], self.obs_stds[task_id])
        values = self.critic(x)
        # print(self.shared_layers[2].weight.grad)
        return actions, action_dist.log_prob(actions), action_dist.entropy(), values

    def gate_parameters(self):
        return self.shared_layers[2].parameters()

    def non_gate_parameters(self):
        return chain(self.shared_layers[0].parameters(), self.shared_layers[-1].parameters())

    def normalize_state(self, state, mean_state, std_state, epsilon=1e-8):
        return (state - mean_state) / (std_state + epsilon)
    
    def update_normalization_stats(self, task_id, states):
        """
        Update mean and std for state normalization using a moving average.
        """
        batch_mean = torch.mean(states, dim=0)
        batch_var = torch.var(states, dim=0)
        batch_count = states.shape[0]

        # Update using running mean and variance
        self.obs_means[task_id], self.obs_stds[task_id], self.counts[task_id] = self.update_mean_var_count(
            self.obs_means[task_id], self.obs_stds[task_id], self.counts[task_id], batch_mean, batch_var, batch_count
        )
    
    def update_mean_var_count(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        total_count = count + batch_count

        new_mean = mean + delta * batch_count / total_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * count * batch_count / total_count
        new_var = M2 / total_count

        return new_mean, torch.sqrt(new_var), total_count
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