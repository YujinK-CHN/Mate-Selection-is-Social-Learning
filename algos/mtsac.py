import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from policies.multitask_policy import TaskConditionedNetwork, TaskConditionedPolicyNetwork


class MultiTaskSAC:
    def __init__(
        self, envs, config
    ):
        self.envs = envs
        self.state_dim = self.envs.observation_space.shape[0]
        self.action_dim = self.envs.action_space.shape[0]
        self.hidden_sizes = config['hidden_dim']
        self.gamma = config['discount']
        self.tau = config['tau']
        self.alpha = config['alpha']
        self.buffer_capacity = config['buffer_capacity']
        self.batch_size = config['batch_size']
        self.policy_lr = config['policy_lr']
        self.qf_lr = config['qf_lr']
        self.min_std = config['min_std']
        self.max_std = config['max_std']
        self.gradient_steps_per_itr = config['gradient_steps_per_itr']
        self.min_buffer_size = config['min_buffer_size']
        self.use_automatic_entropy_tuning = config['use_automatic_entropy_tuning']
        self.epoch_cycles = config['epoch_cycles']
        self.num_epochs = config['num_epochs']
        self.max_path_length = config['max_path_length']


        self.num_tasks = len(envs.tasks)
        self.hidden_nonlinearity = nn.ReLU
        self.policy_net = TaskConditionedPolicyNetwork(self.state_dim, self.action_dim, self.num_tasks, self.hidden_sizes, self.hidden_nonlinearity, self.min_std, self.max_std)
        self.qf1_net = TaskConditionedNetwork(self.state_dim + self.action_dim + self.num_tasks, 1, self.hidden_sizes, self.hidden_nonlinearity)
        self.qf2_net = TaskConditionedNetwork(self.state_dim + self.action_dim + self.num_tasks, 1, self.hidden_sizes, self.hidden_nonlinearity)
        self.qf1_target = TaskConditionedNetwork(self.state_dim + self.action_dim + self.num_tasks, 1, self.hidden_sizes, self.hidden_nonlinearity)
        self.qf2_target = TaskConditionedNetwork(self.state_dim + self.action_dim + self.num_tasks, 1, self.hidden_sizes, self.hidden_nonlinearity)

        self.qf1_target.load_state_dict(self.qf1_net.state_dict())
        self.qf2_target.load_state_dict(self.qf2_net.state_dict())

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)
        self.qf1_optimizer = optim.Adam(self.qf1_net.parameters(), lr=self.qf_lr)
        self.qf2_optimizer = optim.Adam(self.qf2_net.parameters(), lr=self.qf_lr)
        self.replay_buffer = deque(maxlen=self.buffer_capacity)

        if self.use_automatic_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.policy_lr)
            self.target_entropy = -np.prod(self.action_dim).item()
        else:
            self.alpha = 0.2


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
        for epoch in range(self.num_epochs):
            for _ in range(self.epoch_cycles):
                for task_id, env in enumerate(self.envs.tasks):
                    state = env.reset()
                    task_embedding = torch.eye(self.num_tasks)[task_id]

                    for _ in range(self.max_path_length):
                        action = self.select_action(state, task_id)
                        next_state, reward, done, _ = env.step(action)
                        self.store_transition((state, action, reward, next_state, done, task_id))

                        state = next_state
                        if done:
                            break

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

