import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from processing.batching import batchify, batchify_obs, unbatchify
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
    
class MTPPO():

    def __init__(
            self,
            env,
            config
    ):
        self.env = env
        self.obs_shape = env.observation_space.shape[0]
        self.device = config['device']
        self.name = 'mtppo'
        self.policy = MultiTaskPolicy(
            pop_size = config['pop_size'], 
            env = env,
            num_tasks = len(env.tasks),
            continuous = config['continuous'],
            device = config['device']
        ).to(config['device'])
        self.opt = optim.Adam(self.policy.parameters(), lr=config['lr'], eps=1e-5)

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

        y1 = []
        y2 = []
        y3 = []
        
        # train for n number of episodes
        for episode in range(self.total_episodes): # 4000
            self.policy.train()
            # clear memory
            rb_obs = torch.zeros((self.batch_size, self.obs_shape + len(self.env.tasks))).to(self.device)
            if self.continuous == True:
                rb_actions = torch.zeros((self.batch_size, self.env.action_space.shape[0])).to(self.device)
            else:
                rb_actions = torch.zeros((self.batch_size, self.pop_size)).to(self.device)
            rb_logprobs = torch.zeros((self.batch_size, self.pop_size)).to(self.device)
            rb_rewards = torch.zeros((self.batch_size, self.pop_size)).to(self.device)
            rb_advantages = torch.zeros((self.batch_size, self.pop_size)).to(self.device)
            rb_terms = torch.zeros((self.batch_size, self.pop_size)).to(self.device)
            rb_values = torch.zeros((self.batch_size, self.pop_size)).to(self.device)

            # sampling
            index = 0
            task_returns = []
            success_tracker = MultiTaskSuccessTracker(len(self.env.tasks))
            with torch.no_grad():
                for i, task in enumerate(self.env.tasks): # 10
                    episodic_return = []
                    for epoch in range(int((self.batch_size / len(self.env.tasks)) / self.max_cycles)): # 10
                        next_obs, infos = task.reset(seed=0)
                        one_hot_id = torch.diag(torch.ones(len(self.env.tasks)))[i]
                        step_return = 0
                        for step in range(0, self.max_cycles): # 500
                            # rollover the observation 
                            # obs = batchify_obs(next_obs, self.device)
                            obs = torch.FloatTensor(next_obs)
                            obs = torch.concatenate((obs, one_hot_id), dim=-1).to(self.device)

                            # get actions from skills
                            actions, logprobs, entropy, values = self.policy.act(obs)

                            # execute the environment and log data
                            next_obs, rewards, terms, truncs, infos = task.step(actions.cpu().numpy())
                            success = infos.get('success', False)
                            success_tracker.update(i, success)

                            # add to episode storage
                            rb_obs[index] = obs
                            rb_rewards[index] = rewards
                            rb_terms[index] = terms
                            rb_actions[index] = actions
                            rb_logprobs[index] = logprobs
                            rb_values[index] = values.flatten()
                            # compute episodic return
                            step_return += rb_rewards[index].cpu().numpy()

                            # if we reach termination or truncation, end
                            index += 1
                            if terms or truncs:
                                break
                            

                        episodic_return.append(step_return)

                        # skills advantage
                        gae = 0
                        for t in range(index-2, (index-self.max_cycles)-1, -1):
                            delta = rb_rewards[t] + self.discount * rb_values[t + 1] * rb_terms[t + 1] - rb_values[t]
                            gae = delta + self.discount * self.gae_lambda * rb_terms[t] * gae
                            rb_advantages[t] = gae

                    task_returns.append(np.mean(episodic_return))
                                
            rb_returns = rb_advantages + rb_values

            # Optimizing the policy and value network
         
            rb_index = np.arange(rb_obs.shape[0])
            clip_fracs = []
            for epoch in range(self.epoch_opt): # 256
                # shuffle the indices we use to access the data
                np.random.shuffle(rb_index)
                for start in range(0, rb_obs.shape[0], self.min_batch):
                    # select the indices we want to train on
                    end = start + self.min_batch
                    batch_index = rb_index[start:end]

                    if self.continuous == True:
                        old_actions = rb_actions.long()[batch_index, :]
                    else:
                        old_actions = rb_actions.long()[batch_index, :]
                    _, newlogprob, entropy, values = self.policy.evaluate(
                        x = rb_obs[batch_index, :],
                        actions = old_actions
                    )
                    
                    logratio = newlogprob.unsqueeze(-1) - rb_logprobs[batch_index, :]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_fracs += [
                            ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                        ]

                    # normalize advantaegs
                    advantages = rb_advantages[batch_index, :]
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                    # Policy loss
                    pg_loss1 = -rb_advantages[batch_index, :] * ratio
                    pg_loss2 = -rb_advantages[batch_index, :] * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    v_loss_unclipped = (values - rb_returns[batch_index, :]) ** 2
                    v_clipped = rb_values[batch_index, :] + torch.clamp(
                        values - rb_values[batch_index, :],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - rb_returns[batch_index, :]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    entropy_loss = entropy.max()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()

            mean_eval_return, mean_success_rate = self.eval()
            
            print(f"Training episode {episode}")
            print(f"Episodic Return: {np.mean(task_returns)}")
            print(f"Episodic success rate: {success_tracker.overall_success_rate()}")
            print(f"Evaluation Return: {mean_eval_return}")
            print(f"Evaluation success rate: {mean_success_rate}")
            print(f"Episodic Loss: {loss.item()}")
            #print(f"overall success rate: {success_tracker.overall_success_rate() * 100:.2f}")
            print("\n-------------------------------------------\n")

            x = np.linspace(0, episode, episode+1)
            y1.append(np.mean(task_returns))
            y2.append(mean_eval_return)
            #y3.append(success_tracker.overall_success_rate())
            if episode % 10 == 0:
                plt.plot(x, y1)
                plt.plot(x, y2)
                #plt.plot(x, y3)
                plt.pause(0.05)
        plt.show()
        

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


