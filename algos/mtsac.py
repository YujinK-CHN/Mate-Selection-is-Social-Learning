import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

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



########################################


class MultiTaskSAC:
    def __init__(
        self, state_dim, action_dim, num_tasks, hidden_sizes=(400, 400), hidden_nonlinearity=nn.ReLU,
        policy_lr=3e-4, qf_lr=3e-4, min_std=-20, max_std=2, gamma=0.99, tau=5e-3,
        buffer_capacity=1000000, batch_size=5000, gradient_steps_per_itr=500, use_automatic_entropy_tuning=True
    ):
        self.num_tasks = num_tasks
        self.policy_net = TaskConditionedPolicyNetwork(state_dim, action_dim, num_tasks, hidden_sizes, hidden_nonlinearity)
        self.qf1_net = TaskConditionedNetwork(state_dim + action_dim + num_tasks, 1, hidden_sizes, hidden_nonlinearity)
        self.qf2_net = TaskConditionedNetwork(state_dim + action_dim + num_tasks, 1, hidden_sizes, hidden_nonlinearity)
        self.qf1_target = TaskConditionedNetwork(state_dim + action_dim + num_tasks, 1, hidden_sizes, hidden_nonlinearity)
        self.qf2_target = TaskConditionedNetwork(state_dim + action_dim + num_tasks, 1, hidden_sizes, hidden_nonlinearity)

        self.qf1_target.load_state_dict(self.qf1_net.state_dict())
        self.qf2_target.load_state_dict(self.qf2_net.state_dict())

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.qf1_optimizer = optim.Adam(self.qf1_net.parameters(), lr=qf_lr)
        self.qf2_optimizer = optim.Adam(self.qf2_net.parameters(), lr=qf_lr)
        self.replay_buffer = deque(maxlen=buffer_capacity)

        if use_automatic_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=policy_lr)
            self.target_entropy = -np.prod(action_dim).item()
        else:
            self.alpha = 0.2

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.gradient_steps_per_itr = gradient_steps_per_itr

    def select_action(self, state, task_id):
        task_embedding = torch.eye(self.num_tasks)[task_id].unsqueeze(0)  # One-hot encoding
        state = torch.FloatTensor(state).unsqueeze(0)
        action, _ = self.policy_net.sample(state, task_embedding)
        return action.detach().cpu().numpy()[0]

    def store_transition(self, transition):
        self.replay_buffer.append(transition)

    def sample_batch(self):
        batch = np.random.choice(len(self.replay_buffer), self.batch_size)
        state, action, reward, next_state, done, task_id = zip(*[self.replay_buffer[idx] for idx in batch])
        task_embedding = torch.eye(self.num_tasks)[torch.LongTensor(task_id)]
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1),
            task_embedding
        )

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(self.gradient_steps_per_itr):
            state, action, reward, next_state, done, task_embedding = self.sample_batch()

            # Update Q-functions
            with torch.no_grad():
                next_action, log_prob = self.policy_net.sample(next_state, task_embedding)
                qf1_next_target = self.qf1_target(next_state, torch.cat([next_action, task_embedding], dim=1))
                qf2_next_target = self.qf2_target(next_state, torch.cat([next_action, task_embedding], dim=1))
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = reward + (1 - done) * self.gamma * (min_qf_next_target - log_prob)

            qf1 = self.qf1_net(state, torch.cat([action, task_embedding], dim=1))
            qf2 = self.qf2_net(state, torch.cat([action, task_embedding], dim=1))

            qf1_loss = nn.MSELoss()(qf1, next_q_value)
            qf2_loss = nn.MSELoss()(qf2, next_q_value)

            self.qf1_optimizer.zero_grad()
            qf1_loss.backward()
            self.qf1_optimizer.step()

            self.qf2_optimizer.zero_grad()
            qf2_loss.backward()
            self.qf2_optimizer.step()

            # Update policy network
            pi, log_prob = self.policy_net.sample(state, task_embedding)
            qf1_pi = self.qf1_net(state, torch.cat([pi, task_embedding], dim=1))
            qf2_pi = self.qf2_net(state, torch.cat([pi, task_embedding], dim=1))
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            if hasattr(self, 'log_alpha'):
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha = self.log_alpha.exp()
            else:
                alpha = self.alpha

            policy_loss = (alpha * log_prob - min_qf_pi).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Soft update target networks
            with torch.no_grad():
                for target_param, param in zip(self.qf1_target.parameters(), self.qf1_net.parameters()):
                    target_param.data.mul_(1 - self.tau)
                    target_param.data.add_(self.tau * param.data)

                for target_param, param in zip(self.qf2_target.parameters(), self.qf2_net.parameters()):
                    target_param.data.mul_(1 - self.tau)
                    target_param.data.add_(self.tau * param.data)

# Training Loop for Multi-Task SAC
def train_multi_task_sac(envs, agent, num_epochs=500, max_path_length=500, epoch_cycles=200):
    for epoch in range(num_epochs):
        for _ in range(epoch_cycles):
            for task_id, env in enumerate(envs):
                state = env.reset()
                task_embedding = torch.eye(agent.num_tasks)[task_id]

                for _ in range(max_path_length):
                    action = agent.select_action(state, task_id)
                    next_state, reward, done, _ = env.step(action)
                    agent.store_transition((state, action, reward, next_state, done, task_id))

                    state = next_state
                    if done:
                        break

            agent.train()

envs = ...  # Initialize your MT10 environment here
agent = MultiTaskSAC(
    state_dim=envs.observation_space.shape[0],
    action_dim=envs.action_space.shape[0],
    num_tasks=10,  # MT10
    batch_size=5000,
    buffer_capacity=1000000,
    gradient_steps_per_itr=500,
    use_automatic_entropy_tuning=True
)
train_multi_task_sac(envs, agent, num_epochs=500, max_path_length=500, epoch_cycles=200)
