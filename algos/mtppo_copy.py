import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from processing.batching import batchify, batchify_obs, unbatchify
from processing.task_encoder import TaskEncoder
from policies.centralized_policy import CentralizedPolicy
from policies.multitask_policy import MultiTaskPolicy

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

    
class RewardsNormalizer:
    def __init__(self, num_tasks, epsilon=1e-8):
        self.mean = torch.zeros(num_tasks)
        self.var = torch.ones(num_tasks)
        self.count = torch.zeros(num_tasks)
        self.epsilon = epsilon

    def update(self, task_id, reward):
        # Increment count for the current task
        self.count[task_id] += 1
        
        # Store the old mean before updating
        old_mean = self.mean[task_id]
        
        # Update the running mean using Welford's algorithm
        self.mean[task_id] += (reward - old_mean) / self.count[task_id]
        
        # Update the running variance (using the difference between the new reward and the updated mean)
        self.var[task_id] += (reward - old_mean) * (reward - self.mean[task_id])

    def normalize(self, task_id, reward):
        # Compute the standard deviation
        std = torch.sqrt(self.var[task_id] / (self.count[task_id] + self.epsilon))
        
        # Normalize the reward using the updated mean and standard deviation
        return (reward - self.mean[task_id]) / (std + self.epsilon)
    
class ValueNormalizer:
    def __init__(self, num_tasks, epsilon=1e-8):
        self.mean = torch.zeros(num_tasks)
        self.var = torch.ones(num_tasks)
        self.count = torch.zeros(num_tasks)
        self.epsilon = epsilon

    def update_value_normalization(self, x, task_id):
        batch_mean = torch.mean(x)
        batch_var = torch.var(x)
        batch_count = len(x)

        self.mean[task_id], self.var[task_id], self.count[task_id] = self.update_mean_var_count(
            self.mean[task_id], self.var[task_id], self.count[task_id], batch_mean, batch_var, batch_count
        )

    @staticmethod
    def update_mean_var_count(mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        total_count = count + batch_count

        new_mean = mean + delta * batch_count / total_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * count * batch_count / total_count
        new_var = M2 / total_count

        return new_mean, new_var, total_count

    def normalize(self, x, task_id):
        return (x - self.mean[task_id]) / (torch.sqrt(self.var[task_id]) + 1e-8)



class MTPPO():

    def __init__(
            self,
            env,
            config
    ):
        self.env = env
        self.num_tasks = len(env.tasks)
        self.seed = env.seed
        self.obs_shape = env.observation_space.shape[0]
        self.device = config['device']
        self.name = 'mtppo'
        self.hidden_size = config['hidden_size']

        self.task_weights = torch.Tensor([1.5, 1.0, 1.0])

        self.policy = MultiTaskPolicy(
            env = env,
            num_tasks = len(env.tasks),
            hidden_size = config['hidden_size'],
            continuous = config['continuous'],
            normalize_states = config['normalize_states'],
            device = config['device']
        ).to(config['device'])

        self.task_encoder = TaskEncoder(
            num_embeddings=self.num_tasks,
            embedding_dim=config['hidden_size'],
            hidden_dim=config['hidden_size'],
            output_dim=config['hidden_size']
        ).to(config['device'])

        self.opt = optim.Adam(self.policy.parameters(), lr=config['lr'], eps=1e-8)

        self.max_cycles = config['max_path_length']
        self.pop_size = config['pop_size']
        self.total_episodes = config['total_episodes']
        self.epoch_opt = config['epoch_opt']
        self.batch_size = config['batch_size']
        self.min_batch = config['min_batch']
        self.discount = config['discount']
        self.gae_lambda = config['gae_lambda']
        self.clip_coef = config['lr_clip_range']
        self.ent_coef = config['ent_coef']
        self.vf_coef = config['vf_coef']
        self.continuous = config['continuous']
        self.normalize_states = config['normalize_states']
        self.normalize_values = config['normalize_values']
        self.normalize_rewards = config['normalize_rewards']
        
    
    """ TRAINING LOGIC """
    
    def train(self):
        x = []
        y = []
        x_eval = []
        y_eval = []
        y3 = []
        
        # main loop
        for episode in range(self.total_episodes):
            self.policy.train()

            # clear memory
            # [num_tasks, num_samples for each task, ...]
            rb_obs = torch.zeros((self.num_tasks, int(self.batch_size / self.num_tasks), self.obs_shape + len(self.env.tasks))).to(self.device)
            if self.continuous == True:
                rb_actions = torch.zeros((self.num_tasks, int(self.batch_size / self.num_tasks), self.env.action_space.shape[0])).to(self.device)
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
                # set/reset up normalizer, for new samples
                if self.normalize_rewards:
                    reward_normalizer = RewardsNormalizer(num_tasks=self.num_tasks)
                if self.normalize_values:
                    value_normalizer = ValueNormalizer(num_tasks=self.num_tasks)
                
                # sampling each task
                for i, task in enumerate(self.env.tasks): # [3, 5, 7]
                    index = 0
                    episodic_return = []
                    # batch_size(30000, 50000, 70000) = num_task(3, 5, 7) * max_cycles(500) * num_cycles(20)
                    for epoch in range(int((self.batch_size / len(self.env.tasks)) / self.max_cycles)): # num_cycles(20)
                        next_obs, infos = task.reset(self.seed)
                        one_hot_id = torch.diag(torch.ones(len(self.env.tasks)))[i]
                        step_return = 0
                        for step in range(0, self.max_cycles): # 500
                            # rollover the observation 
                            # obs = batchify_obs(next_obs, self.device)
                            obs = torch.FloatTensor(next_obs)

                            obs = torch.concatenate((obs, one_hot_id), dim=-1).to(self.device)

                            # get actions from skills
                            actions, logprobs, entropy, values = self.policy.act(obs, i)

                            # execute the environment and log data
                            next_obs, reward, terms, truncs, infos = task.step(actions.cpu().numpy())
                            success = infos.get('success', False)
                            success_tracker.update(i, success)

                            if self.normalize_rewards:
                                reward_normalizer.update(i, reward)
                                reward = reward_normalizer.normalize(i, reward)
                            if self.normalize_values:
                                values = value_normalizer.normalize(values, i)

                            # add to episode storage
                            rb_obs[i, index] = obs
                            rb_rewards[i, index] = reward
                            rb_terms[i, index] = terms
                            rb_actions[i, index] = actions
                            rb_logprobs[i, index] = logprobs
                            rb_values[i, index] = values.flatten()
                            # compute episodic return
                            step_return += rb_rewards[i, index].cpu().numpy()

                            # if we reach termination or truncation, end
                            index += 1
                            if terms or truncs:
                                break
                            
                        
                        episodic_return.append(step_return)

                        # skills advantage
                        gae = 0
                        for t in range(index-2, (index-self.max_cycles)-1, -1):
                            delta = rb_rewards[i, t] + self.discount * rb_values[i, t + 1] * rb_terms[i, t + 1] - rb_values[i, t]
                            gae = delta + self.discount * self.gae_lambda * rb_terms[i, t] * gae
                            rb_advantages[i, t] = gae

                        
                    task_returns.append(np.mean(episodic_return))

                    # not sure if is right.
                    if self.normalize_states:
                        self.policy.update_normalization_stats(i, rb_obs[i, :, :])
                    if self.normalize_values:
                        value_normalizer.update_value_normalization(rb_values[i, :, :], i)   
                    #      
            rb_returns = rb_advantages + rb_values

            # Optimizing the policy and value network
         
            rb_index = np.arange(rb_obs.shape[1])
            clip_fracs = []
            for epoch in range(self.epoch_opt): # 16
                # shuffle the indices we use to access the data
                np.random.shuffle(rb_index)
                
                for start in range(0, rb_obs.shape[1], self.min_batch):
                    task_losses = torch.zeros(self.num_tasks)
                    end = start + self.min_batch
                    batch_index = rb_index[start:end]
                    for i, task in enumerate(self.env.tasks): # 10
                        # select the indices we want to train on
                        if self.continuous == True:
                            old_actions = rb_actions.long()[i, batch_index, :]
                        else:
                            old_actions = rb_actions.long()[i, batch_index, :]
                        _, newlogprob, entropy, values = self.policy.evaluate(
                            x = rb_obs[i, batch_index, :],
                            actions = old_actions,
                            task_id = i
                        )

                        if self.normalize_values:
                            values = value_normalizer.normalize(values, i)
                        
                        ratio = torch.exp(newlogprob.unsqueeze(-1) - rb_logprobs[i, batch_index, :])

                        # normalize advantaegs
                        advantages = rb_advantages[i, batch_index, :]
                        advantages = (advantages - advantages.mean()) / (
                            advantages.std() + 1e-8
                        )

                        # Policy loss
                        pg_loss1 = rb_advantages[i, batch_index, :] * ratio
                        pg_loss2 = rb_advantages[i, batch_index, :] * torch.clamp(
                            ratio, 1 - self.clip_coef, 1 + self.clip_coef
                        )
                        pg_loss = -torch.mean(torch.min(pg_loss1, pg_loss2))

                        # Value loss
                        v_loss_unclipped = (values - rb_returns[i, batch_index, :]) ** 2
                        v_clipped = rb_values[i, batch_index, :] + torch.clamp(
                            values - rb_values[i, batch_index, :],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - rb_returns[i, batch_index, :]) ** 2
                        v_loss_min = torch.min(v_loss_unclipped, v_loss_clipped)
                        v_loss = torch.mean(0.5 * v_loss_min)

                        entropy_loss = entropy.max()
                        loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                        task_losses[i] = loss

                    self.opt.zero_grad()
                    torch.sum(task_losses*self.task_weights).backward()
                    self.opt.step()

                    
            print(f"Training episode {episode}")
            print(f"Training seed {self.seed}")
            print(f"Episodic Return: {np.mean(task_returns)}")
            print(f"Episodic success rate: {success_tracker.overall_success_rate()}")
            print(f"Episodic Loss: {loss.item()}")
            print("\n-------------------------------------------\n")
            if episode % 10 == 0:
                eval_return, mean_success_rate = self.eval(self.policy)
                x_eval.append(episode)
                y_eval.append(np.mean(eval_return))
                print(f"Evaluating episode {episode}")
                print(f"Evaluating seed {self.seed}")
                print(f"Evaluation Return: {eval_return}")
                print(f"Evaluation Mean Return: {np.mean(eval_return)}")
                print(f"Evaluation success rate: {mean_success_rate}")
                print("\n-------------------------------------------\n")

            x.append(episode)
            y.append(np.mean(task_returns))
            
            #y3.append(success_tracker.overall_success_rate())
            if episode % 10 == 0:
                plt.figure()
                plt.plot(x, y)
                plt.title(f"Training return for {self.seed}")
                plt.xlabel("episodes")
                plt.ylabel("mean rewards")
                plt.pause(0.05)

                plt.figure()
                plt.plot(x_eval, y_eval)
                plt.title(f"Evaluating return for {self.seed}")
                plt.xlabel("episodes")
                plt.ylabel("mean rewards")
                plt.pause(0.05)
        plt.show(block=False)
        
        return x, y, x_eval, y_eval       
        
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

                    if self.normalize_rewards:
                        normalizer = RewardsNormalizer(num_tasks=self.num_tasks)

                    while not terms and not truncs:
                        # rollover the observation 
                        #obs = batchify_obs(next_obs, self.device)
                        obs = torch.FloatTensor(next_obs)
                        obs = torch.concatenate((obs, one_hot_id), dim=-1).to(self.device)

                        # get actions from skills
                        actions, logprobs, entropy, values = policy.act(obs, i)

                        # execute the environment and log data
                        next_obs, reward, terms, truncs, infos = task.step(actions.cpu().numpy())
                        success = infos.get('success', False)
                        success_tracker_eval.update(i, success)
                        if self.normalize_rewards:
                            normalizer.update(i, reward)
                            reward = normalizer.normalize(i, reward)
                        terms = terms
                        truncs = truncs
                        step_return += reward
                    episodic_return.append(step_return)
                
                task_returns.append(np.mean(episodic_return))
        return task_returns, success_tracker_eval.overall_success_rate()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)


