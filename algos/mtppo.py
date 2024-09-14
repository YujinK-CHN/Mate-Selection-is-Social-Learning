import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from itertools import chain
from processing.batching import batchify, batchify_obs, unbatchify
from policies.multitask_policy import MultiTaskPolicy
from processing.normalizer import NormalizeObservation, NormalizeReward

class MultiTaskSuccessTracker:
    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.success_counts = [0] * num_tasks
        self.total_counts = [0] * num_tasks

    def count(self, task_id):
        """Update success counts based on task_id."""
        self.total_counts[task_id] += 1

    def success(self, task_id):
        """Update success counts based on task_id."""
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
    
class MTPPO():

    def __init__(
            self,
            env,
            config
    ):
        self.env = env
        self.seed = env.seed
        self.obs_shape = env.observation_space.shape[0]
        self.device = config['device']
        self.name = 'mtppo'
        self.hidden_size = config['hidden_size']
        self.normalize_states = config['normalize_states']
        self.normalize_values = config['normalize_values']
        self.normalize_rewards = config['normalize_rewards']
        self.num_tasks = len(env.tasks)

        self.policy = MultiTaskPolicy(
            env = env,
            num_tasks = len(env.tasks),
            hidden_size = config['hidden_size'],
            continuous = config['continuous'],
            device = config['device']
        ).to(config['device'])

        self.max_grad_norm = 0.5
        self.lr = config['lr']
        self.opt = optim.Adam(self.policy.parameters(), lr=config['lr'], eps=1e-8)
        self.actor_opt = optim.Adam(self.policy.actor(), lr=config['lr'], eps=1e-8)
        self.critic_opt = optim.Adam(self.policy.critic.parameters(), lr=config['lr'], eps=1e-8)

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
        
    
    """ TRAINING LOGIC """
    
    def train(self):

        x = []
        y = []
        x_eval = []
        y_eval = []
        sr = []
        tasks_sr = []

        # clear memory
        rb_obs = torch.zeros((self.batch_size, self.obs_shape + self.num_tasks)).to(self.device)
        if self.continuous == True:
            rb_actions = torch.zeros((self.batch_size, self.env.action_space.shape[0])).to(self.device)
        else:
            rb_actions = torch.zeros((self.batch_size, 1)).to(self.device)
        rb_logprobs = torch.zeros((self.batch_size, 1)).to(self.device)
        rb_rewards = torch.zeros((self.batch_size, 1)).to(self.device)
        rb_advantages = torch.zeros((self.batch_size, 1)).to(self.device)
        rb_terms = torch.zeros((self.batch_size, 1)).to(self.device)
        rb_values = torch.zeros((self.batch_size, 1)).to(self.device)
        
        # train for n number of episodes
        for episode in range(1, self.total_episodes+1): 

            ''''''
            # learning Rate annealing
            frac = 1.0 - (episode - 1.0) / self.total_episodes
            new_lr = self.lr * (1.0 - frac)
            new_lr = max(new_lr, 0.0)
            self.actor_opt.param_groups[0]["lr"] = new_lr
            self.critic_opt.param_groups[0]["lr"] = new_lr
            
            self.policy.train()
            
            # sampling
            index = 0
            
            task_returns = []
            episodic_tasks_sr = []
            success_tracker = MultiTaskSuccessTracker(self.num_tasks)
            norm_obs = NormalizeObservation(self.num_tasks)
            norm_rew = NormalizeReward(self.num_tasks)
            
            with torch.no_grad():
                for i, task in enumerate(self.env.tasks): # 10
                    episodic_return = []
                    
                    for epoch in range(int((self.batch_size / self.num_tasks) / self.max_cycles)): # 10         

                        next_obs, infos = task.reset(self.seed)
                        success_tracker.count(i)
                        if self.normalize_states:
                            next_obs = norm_obs.normalize(torch.FloatTensor(next_obs), i)
                            next_obs = torch.clip(next_obs, -10, 10)

                        one_hot_id = torch.diag(torch.ones(self.num_tasks))[i]
                        
                        step_return = 0
                        for step in range(0, self.max_cycles): # 500
                            # rollover the observation 
                            # obs = batchify_obs(next_obs, self.device)
                            obs = torch.FloatTensor(next_obs)
                            obs = torch.concatenate((obs, one_hot_id), dim=-1).to(self.device)

                            # get actions from skills
                            action, logprob, entropy, value = self.policy.get_action_and_value(obs)
                            # add to episode storage
                            rb_actions[index] = action
                            rb_logprobs[index] = logprob
                            rb_values[index] = value.flatten()

                            # execute the environment and log data
                            action = torch.clip(action, -1, 1)  # action clip
                            next_obs, reward, term, trunc, info = task.step(action.cpu().numpy())
                            step_return += reward
                            

                            success = info.get('success', 0.0)
                            if success != 0.0:
                                print("!!!!!!!!!!", epoch, step, success)
                                success_tracker.success(i)
                                term = 1.0
                                next_obs, infos = task.reset(self.seed)
                                success_tracker.count(i)

                            if self.normalize_states:
                                next_obs = norm_obs.normalize(torch.FloatTensor(next_obs), i)
                                next_obs = torch.clip(next_obs, -10, 10)
                                
                            if self.normalize_rewards:
                                reward = norm_rew.normalize(reward, term, i)
                                reward = torch.clip(reward, -10, 10)

                            
                            
                            
                            # add to episode storage
                            rb_obs[index] = obs
                            rb_rewards[index] = reward
                            rb_terms[index] = term
                            
                            
                            index += 1
                            

                        episodic_return.append(step_return)

                        # advantage
                        gae = 0
                        for t in range(index-2, (index-self.max_cycles)-1, -1):
                            delta = rb_rewards[t] + self.discount * rb_values[t + 1] * rb_terms[t + 1] - rb_values[t]
                            gae = delta + self.discount * self.gae_lambda * rb_terms[t] * gae
                            rb_advantages[t] = gae

                        
                    task_returns.append(np.mean(episodic_return))
                    episodic_tasks_sr.append(success_tracker.task_success_rate(i))
            tasks_sr.append(episodic_tasks_sr)     

            # normalize advantaegs
            rb_returns = rb_advantages + rb_values

            # Optimizing the policy and value network
         
            rb_index = np.arange(rb_obs.shape[0])
            clip_fracs = []
            for epoch in range(self.epoch_opt): # 16
                # shuffle the indices we use to access the data
                np.random.shuffle(rb_index)
                
                
                for start in range(0, rb_obs.shape[0], self.min_batch):
                    # select the indices we want to train on
                    end = start + self.min_batch
                    batch_index = rb_index[start:end]

                    if self.continuous == True:
                        old_actions = rb_actions[batch_index, :]
                    else:
                        old_actions = rb_actions.long()[batch_index, :]
                    _, newlogprob, entropy, values = self.policy.get_action_and_value(x = rb_obs[batch_index, :], action = old_actions)
                    
                    logratio = newlogprob.unsqueeze(-1) - rb_logprobs[batch_index, :]
                    ratio = logratio.exp()

                    # debug variable
                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_fracs += [
                            ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                        ]

                    # Advantage normalization
                    advantages = rb_advantages[batch_index, :]
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)    

                    # Policy loss
                    pg_loss1 = -advantages * ratio
                    pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    v_loss_unclipped = (values - rb_returns[batch_index, :]) ** 2
                    v_clipped = rb_values[batch_index, :] + torch.clamp(
                        values - rb_values[batch_index, :],
                        -self.clip_coef,
                        self.clip_coef,
                    )

                    v_loss_clipped = (v_clipped - rb_returns[batch_index, :]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                    
                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.opt.step()

                    '''
                    # Calculate gradients and perform backward propagation for actor network
                    self.actor_opt.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.policy.shared_layers.parameters(), self.max_grad_norm)
                    self.actor_opt.step()

                    # Calculate gradients and perform backward propagation for critic network
                    self.critic_opt.zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
                    self.critic_opt.step()
                    '''
            y_pred, y_true = rb_values.cpu().numpy(), rb_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            print(f"Training episode {episode}")
            print(f"Episodic Return: {np.mean(task_returns)}")
            print(f"Episodic success rate: {success_tracker.overall_success_rate()}")
            print(f"Training seed {self.seed}")
            print("")
            print(f"Value Loss: {v_loss.item()}")
            print(f"Policy Loss: {pg_loss.item()}")
            print(f"Old Approx KL: {old_approx_kl.item()}")
            print(f"Approx KL: {approx_kl.item()}")
            print(f"Clip Fraction: {np.mean(clip_fracs)}")
            print(f"Explained Variance: {explained_var.item()}")
            print("\n-------------------------------------------\n")
            if episode % 10 == 0:
                eval_return, mean_success_rate = self.eval(self.policy, norm_obs)
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
            sr.append(success_tracker.overall_success_rate)
            
            if episode % 10 == 0:
                
                plt.plot(x, y)
                plt.plot(x_eval, y_eval)
                plt.title(f"Episode returns (train and eval) for seed {self.seed}")
                plt.xlabel("Episodes")
                plt.ylabel("Mean rewards")
                plt.pause(0.05)

        plt.show(block=False)
        
        return x, y, x_eval, y_eval, sr, tasks_sr       
        

    def eval(self, policy, norm_obs):
        task_returns = []
        success_tracker_eval = MultiTaskSuccessTracker(self.num_tasks)
        policy.eval()
        with torch.no_grad():
            for i, task in enumerate(self.env.tasks): # 10
                episodic_return = []
                    
                for epoch in range(5):      

                    next_obs, infos = task.reset(self.seed)
                    success_tracker_eval.count(i)
                    if self.normalize_states:
                        next_obs = norm_obs.normalize(torch.FloatTensor(next_obs), i)
                        next_obs = torch.clip(next_obs, -10, 10)

                    one_hot_id = torch.diag(torch.ones(self.num_tasks))[i]
                    step_return = 0
                    for step in range(0, self.max_cycles): # 500
                        # rollover the observation 
                        # obs = batchify_obs(next_obs, self.device)
                        obs = torch.FloatTensor(next_obs)
                        obs = torch.concatenate((obs, one_hot_id), dim=-1).to(self.device)

                        # get actions from skills
                        action, logprob, entropy, value = self.policy.get_action_and_value(obs)

                        # execute the environment and log data
                        action = torch.clip(action, -1, 1)  # action clip
                        next_obs, reward, term, trunc, info = task.step(action.cpu().numpy())
                        step_return += reward

                        success = info.get('success', 0.0)
                        if success != 0.0:
                            print("!!!!!!!!!!", epoch, step, success)
                            success_tracker_eval.success(i)
                            next_obs, infos = task.reset(self.seed)
                            success_tracker_eval.count(i)

                        if self.normalize_states:
                            next_obs = norm_obs.normalize(torch.FloatTensor(next_obs), i)
                            next_obs = torch.clip(next_obs, -10, 10)
                                
                    episodic_return.append(step_return)
                task_returns.append(np.mean(episodic_return))
        return task_returns, success_tracker_eval.overall_success_rate()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)


