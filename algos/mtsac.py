import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
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
        self.name = 'mtsac'
        self.device = config['device']
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
        self.epoch_opt = config['epoch_opt']
        self.total_episodes = config['total_episodes']
        self.max_path_length = config['max_path_length']
        self.continuous = True
        self.min_batch = 32

        self.seed = envs.seed
        self.num_tasks = len(envs.tasks)
        self.hidden_nonlinearity = nn.ReLU
        self.policy_net = TaskConditionedPolicyNetwork(self.state_dim, self.action_dim, self.num_tasks, self.hidden_sizes, self.hidden_nonlinearity, self.min_std, self.max_std).to(self.device)
        self.qf1_net = TaskConditionedNetwork(self.state_dim + self.action_dim + self.num_tasks, 1, self.hidden_sizes, self.hidden_nonlinearity).to(self.device)
        self.qf2_net = TaskConditionedNetwork(self.state_dim + self.action_dim + self.num_tasks, 1, self.hidden_sizes, self.hidden_nonlinearity).to(self.device)
        self.qf1_target = TaskConditionedNetwork(self.state_dim + self.action_dim + self.num_tasks, 1, self.hidden_sizes, self.hidden_nonlinearity).to(self.device)
        self.qf2_target = TaskConditionedNetwork(self.state_dim + self.action_dim + self.num_tasks, 1, self.hidden_sizes, self.hidden_nonlinearity).to(self.device)

        self.qf1_target.load_state_dict(self.qf1_net.state_dict())
        self.qf2_target.load_state_dict(self.qf2_net.state_dict())

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)
        self.qf1_optimizer = optim.Adam(self.qf1_net.parameters(), lr=self.qf_lr)
        self.qf2_optimizer = optim.Adam(self.qf2_net.parameters(), lr=self.qf_lr)

        if self.use_automatic_entropy_tuning:
            self.log_alpha = torch.nn.Parameter(
                torch.tensor(
                    [
                        np.log(1.0, dtype=np.float32)
                        for _ in range(self.num_tasks)
                    ]
                ).to(self.device)
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.policy_lr)
            self.target_entropy = -np.prod(self.action_dim).item()
        else:
            self.alpha = 0.2

 

    def train(self):
        y1 = []
        y2 = []
        y3 = []
        for episode in range(self.total_episodes):
            self.policy_net.train()
            self.qf1_net.train()
            self.qf2_net.train()
            # clear memory
            rb_obs = torch.zeros((self.num_tasks, int(self.batch_size / self.num_tasks), self.state_dim + len(self.envs.tasks))).to(self.device)
            rb_next_obs = torch.zeros((self.num_tasks, int(self.batch_size / self.num_tasks), self.state_dim + len(self.envs.tasks))).to(self.device)
            if self.continuous == True:
                rb_actions = torch.zeros((self.num_tasks, int(self.batch_size / self.num_tasks), self.action_dim)).to(self.device)
            else:
                rb_actions = torch.zeros((self.num_tasks, int(self.batch_size / self.num_tasks), 1)).to(self.device) # [10, 10000, 1] if batch size 100000. 10000 = 500 * 20
            rb_logprobs = torch.zeros((self.num_tasks, int(self.batch_size / self.num_tasks), 1)).to(self.device)
            rb_rewards = torch.zeros((self.num_tasks, int(self.batch_size / self.num_tasks), 1)).to(self.device)
            rb_advantages = torch.zeros((self.num_tasks, int(self.batch_size / self.num_tasks), 1)).to(self.device)
            rb_terms = torch.zeros((self.num_tasks, int(self.batch_size / self.num_tasks), 1)).to(self.device)
            rb_values = torch.zeros((self.num_tasks, int(self.batch_size / self.num_tasks), 1)).to(self.device)

            # sampling
            
            task_returns = []
            success_tracker = MultiTaskSuccessTracker(self.num_tasks)
            with torch.no_grad():
                for i, task in enumerate(self.envs.tasks): # 10
                    index = 0
                    episodic_return = []
                    for epoch in range(int((self.batch_size / self.num_tasks) / 500)): # 20
                        next_obs, infos = task.reset()
                        one_hot_id = torch.diag(torch.ones(self.num_tasks))[i]
                        step_return = 0
                        for step in range(0, 500): # 500
                            # rollover the observation 
                            # obs = batchify_obs(next_obs, self.device)
                            obs = torch.FloatTensor(next_obs)
                            obs = torch.cat((obs, one_hot_id), dim=-1).to(self.device)

                            # get actions from skills
                            actions, logprobs = self.policy_net.sample(obs)

                            # execute the environment and log data
                            next_obs, rewards, terms, truncs, infos = task.step(actions.cpu().numpy())
                            success = infos.get('success', False)
                            success_tracker.update(i, success)

                            # add to episode storage
                            rb_obs[i, index] = obs
                            rb_next_obs[i, index] = torch.cat((torch.FloatTensor(next_obs), one_hot_id), dim=-1)
                            rb_rewards[i, index] = rewards
                            rb_terms[i, index] = terms
                            rb_actions[i, index] = actions
                            rb_logprobs[i, index] = logprobs
                            # compute episodic return
                            step_return += rb_rewards[i, index].cpu().numpy()

                            # if we reach termination or truncation, end
                            index += 1
                            if terms or truncs:
                                break
                            

                        episodic_return.append(step_return)
                    task_returns.append(np.mean(episodic_return))

            rb_index = np.arange(rb_obs.shape[1])
                
            for epoch in range(self.epoch_opt):
                # shuffle the indices we use to access the data
                np.random.shuffle(rb_index)
                for start in range(self.gradient_steps_per_itr):
                    end = start + self.min_batch
                    batch_index = rb_index[start:end]
                    for i, task in enumerate(self.envs.tasks): # 10

                        # Update Q-functions
                        with torch.no_grad():
                            next_action, log_prob = self.policy_net.sample(rb_next_obs[i, batch_index, :])
                            qf1_next_target = self.qf1_target(torch.cat((rb_next_obs[i, batch_index, :], next_action), dim=-1))
                            qf2_next_target = self.qf2_target(torch.cat((rb_next_obs[i, batch_index, :], next_action), dim=-1))
                            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                            next_q_value = rb_rewards[i, batch_index, :] + (1 - rb_terms[i, batch_index, :]) * self.gamma * (min_qf_next_target - log_prob)
                            
                        
                        qf1 = self.qf1_net(torch.cat((rb_next_obs[i, batch_index, :], next_action), dim=-1))
                        qf2 = self.qf2_net(torch.cat((rb_next_obs[i, batch_index, :], next_action), dim=-1))

                        qf1_loss = F.mse_loss(qf1, next_q_value)
                        qf2_loss = F.mse_loss(qf2, next_q_value)

                            
                        self.qf1_optimizer.zero_grad()
                        qf1_loss.backward(retain_graph=True)
                        self.qf1_optimizer.step()

                        self.qf2_optimizer.zero_grad()
                        qf2_loss.backward(retain_graph=True)
                        self.qf2_optimizer.step()

                        # Update policy network
                        pi, log_prob = self.policy_net.sample(rb_obs[i, batch_index, :])
                        qf1_pi = self.qf1_net(torch.cat((rb_next_obs[i, batch_index, :], pi), dim=-1))
                        qf2_pi = self.qf2_net(torch.cat((rb_next_obs[i, batch_index, :], pi), dim=-1))
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)

                        if hasattr(self, 'log_alpha'):
                            self.log_alpha = self.log_alpha.to(self.device)
                            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy)).mean()
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
            print(f"Episodic Return: {np.mean(task_returns)}")
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

    def eval(self, policy):
        task_returns = []
        success_tracker_eval = MultiTaskSuccessTracker(self.num_tasks)
        policy.eval()
        with torch.no_grad():
            for i, task in enumerate(self.env.tasks):
                # render 5 episodes out
                episodic_return = []
                for episode in range(5):
                    next_obs, infos = task.reset()
                    one_hot_id = torch.diag(torch.ones(self.num_tasks))[i]
                    terms = False
                    truncs = False
                    step_return = 0
                    while not terms and not truncs:
                        # rollover the observation 
                        #obs = batchify_obs(next_obs, self.device)
                        obs = torch.FloatTensor(next_obs)
                        obs = torch.concatenate((obs, one_hot_id), dim=-1).to(self.device)

                        # get actions from skills
                        actions, logprobs, entropy, values = policy.act(obs)

                        # execute the environment and log data
                        next_obs, rewards, terms, truncs, infos = task.step(actions.cpu().numpy())
                        success = infos.get('success', False)
                        success_tracker_eval.update(i, success)
                        terms = terms
                        truncs = truncs
                        step_return += rewards
                    episodic_return.append(step_return)
                task_returns.append(np.mean(episodic_return))
        return task_returns, success_tracker_eval.overall_success_rate()
    

    def save(self, path):
        torch.save(self.policy.state_dict(), path)