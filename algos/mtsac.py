import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from policies.multitask_policy import TaskConditionedNetwork, TaskConditionedPolicyNetwork

class MultiTaskSuccessTracker:
    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.success_counts = [0] * num_tasks
        self.total_counts = [0] * num_tasks

    def update(self, task_id, success):
        """Update success counts based on task_id."""
        self.total_counts[task_id] += 1
        if success:
            self.success_counts[task_id] += 1

    def task_success_rate(self, task_id):
        """Calculate success rate for a specific task."""
        if self.total_counts[task_id] == 0:
            return 0.0
        return self.success_counts[task_id] / self.total_counts[task_id]

    def overall_success_rate(self):
        """Calculate the overall success rate across all tasks."""
        total_successes = sum(self.success_counts)
        total_attempts = sum(self.total_counts)
        if total_attempts == 0:
            return 0.0
        return total_successes / total_attempts

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

        self.seed = envs.seed
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
        y1 = []
        y2 = []
        y3 = []
        for epoch in range(self.num_epochs):
            cycle_returns = []
            success_tracker = MultiTaskSuccessTracker(len(self.env.tasks))
            for _ in range(self.epoch_cycles):
                task_returns = []
                for task_id, env in enumerate(self.envs.tasks):
                    state = env.reset()
                    task_embedding = torch.eye(self.num_tasks)[task_id]
                    step_return = 0
                    for _ in range(self.max_path_length):
                        action = self.select_action(state, task_id)
                        next_state, reward, done, truncs, info= env.step(action)
                        success = info.get('success', False)
                        success_tracker.update(task_id, success)
                        self.store_transition((state, action, reward, next_state, done, task_id))

                        state = next_state
                        step_return += reward
                        
                        if done:
                            break
                    task_returns.append(step_return)
                cycle_returns.append(np.mean(task_returns))

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

            mean_eval_return, mean_success_rate = self.eval()
            print(f"Training episode {epoch}")
            print(f"Training seed {self.seed}")
            print(f"Episodic Return: {np.mean(cycle_returns)}")
            print(f"Episodic success rate: {success_tracker.overall_success_rate()}")
            print(f"Evaluation Return: {mean_eval_return}")
            print(f"Evaluation success rate: {mean_success_rate}")
            print(f"Episodic Loss: {policy_loss.item()}")
            #print(f"overall success rate: {success_tracker.overall_success_rate() * 100:.2f}")
            print("\n-------------------------------------------\n")

            x = np.linspace(0, epoch, epoch+1)
            y1.append(np.mean(task_returns))
            y2.append(mean_eval_return)
            #y3.append(success_tracker.overall_success_rate())
            if epoch % 10 == 0:
                plt.plot(x, y1)
                plt.plot(x, y2)
                #plt.plot(x, y3)
                plt.pause(0.05)
        plt.show(block=False)
        
        return x, y1

    def eval(self):
        episodic_return = []
        success_tracker_eval = MultiTaskSuccessTracker(len(self.env.tasks))
        self.policy.eval()
        with torch.no_grad():
            # render 5 episodes out
            for episode in range(5):
                next_obs, infos = self.env.reset()
                task_id = self.env.tasks.index(self.env.current_task)
                one_hot_id = torch.diag(torch.ones(len(self.env.tasks)))[task_id]
                terms = False
                truncs = False
                step_return = 0
                while not terms and not truncs:
                    # rollover the observation 
                    #obs = batchify_obs(next_obs, self.device)
                    obs = torch.FloatTensor(next_obs)
                    obs = torch.concatenate((obs, one_hot_id), dim=-1).to(self.device)

                    # get actions from skills
                    actions, logprobs, entropy, values = self.policy.act(obs)

                    # execute the environment and log data
                    next_obs, rewards, terms, truncs, infos = self.env.step(actions.cpu().numpy())
                    success = infos.get('success', False)
                    success_tracker_eval.update(task_id, success)
                    terms = terms
                    truncs = truncs
                    step_return += rewards
                episodic_return.append(step_return)

        return np.mean(episodic_return), success_tracker_eval.overall_success_rate()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)