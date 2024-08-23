import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from processing.batching import batchify, batchify_obs, unbatchify
from loss.ppo_loss import clip_ppo_loss
from policies.centralized_policy import CentralizedPolicy
from policies.independent_policy import IndependentPolicy
from policies.encoder_independent_policy import Encoder_independentPolicy

class PPO():

    def __init__(
            self,
            env,
            config
    ):
        self.env = env
        self.device = config['device']
        self.name = 'ppo'
        self.policy = CentralizedPolicy(
            n_agents = config['n_agents'], 
            input_dim = config['obs_shape'],
            output_dim = config['num_actions'],
            continuous = config['continuous'],
            device = config['device']
        ).to(config['device'])
        self.opt = optim.Adam(self.policy.parameters(), lr=config['lr'], eps=1e-5)

        self.max_cycles = config['max_cycles']
        self.n_agents = config['n_agents']
        self.num_actions = config['num_actions']
        self.obs_shape = config['obs_shape']
        self.curr_latent = None
        self.total_episodes = config['total_episodes']
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.clip_coef = config['clip_coef']
        self.ent_coef = config['ent_coef']
        self.vf_coef = config['vf_coef']
        self.continuous = config['continuous']
    
    """ TRAINING LOGIC """
    def train(self):
        
        y = []
        end_step = 0
        total_episodic_return = 0
        rb_obs = torch.zeros((self.max_cycles, self.n_agents, self.obs_shape)).to(self.device)
        if self.continuous == True:
            rb_actions = torch.zeros((self.max_cycles, self.n_agents, self.num_actions)).to(self.device)
        else:
            rb_actions = torch.zeros((self.max_cycles, self.n_agents)).to(self.device)
        rb_logprobs = torch.zeros((self.max_cycles, self.n_agents)).to(self.device)
        rb_rewards = torch.zeros((self.max_cycles, self.n_agents)).to(self.device)
        rb_terms = torch.zeros((self.max_cycles, self.n_agents)).to(self.device)
        rb_values = torch.zeros((self.max_cycles, self.n_agents)).to(self.device)
        
        # train for n number of episodes
        for episode in range(self.total_episodes):
            # collect an episode
            with torch.no_grad():
                # collect observations and convert to batch of torch tensors
                next_obs, info = self.env.reset(seed=0)
                
                # reset the episodic return
                total_episodic_return = 0

                # each episode has num_steps
                for step in range(0, self.max_cycles):
                    # rollover the observation 
                    #obs = batchify_obs(next_obs, self.device)
                    obs = torch.FloatTensor(next_obs).to(self.device)

                    # get actions from skills
                    actions, logprobs, entropy, values = self.policy.act(obs)

                    # execute the environment and log data
                    next_obs, rewards, terms, truncs, infos = self.env.step(actions.cpu().numpy())

                    # add to episode storage
                    rb_obs[step] = obs
                    rb_rewards[step] = rewards
                    rb_terms[step] = terms
                    rb_actions[step] = actions
                    rb_logprobs[step] = logprobs
                    rb_values[step] = values.flatten()
                    # compute episodic return
                    total_episodic_return += rb_rewards[step].cpu().numpy()

                    # if we reach termination or truncation, end
                    if terms or truncs:
                        end_step = step
                        break

            # skills advantage
            with torch.no_grad():
                rb_advantages = torch.zeros_like(rb_rewards).to(self.device)
                for t in reversed(range(end_step)):
                    delta = (
                        rb_rewards[t]
                        + self.gamma * rb_values[t + 1] * rb_terms[t + 1]
                        - rb_values[t]
                    )
                    rb_advantages[t] = delta + self.gamma * self.gamma * rb_advantages[t + 1]
                rb_returns = rb_advantages + rb_values



            # Optimizing the policy and value network
            rb_index = np.arange(rb_obs.shape[0])
            clip_fracs = []
            
            # shuffle the indices we use to access the data
            np.random.shuffle(rb_index)
            for start in range(0, rb_obs.shape[0], self.batch_size):
                # select the indices we want to train on
                end = start + self.batch_size
                batch_index = rb_index[start:end]

                if self.continuous == True:
                    old_actions = rb_actions.long()[batch_index, :, :]
                else:
                    old_actions = rb_actions.long()[batch_index, :]
                _, newlogprob, entropy, values = self.policy.evaluate(
                    x = rb_obs[batch_index, :, :],
                    actions = old_actions
                )

                
                logratio = newlogprob - rb_logprobs[batch_index, :]
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
                values = values.squeeze(-1)
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

                '''
                loss = clip_ppo_loss(
                    newlogprob,
                    entropy,
                    value,
                    b_values[batch_index],
                    b_logprobs[batch_index],
                    b_advantages[batch_index],
                    b_returns[batch_index],
                    self.clip_coef,
                    self.ent_coef,
                    self.vf_coef
                )
                '''
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            print(f"Training episode {episode}")
            print(f"Episodic Return: {total_episodic_return}")
            print(f"Episodic Mean Return: {np.mean(total_episodic_return)}")
            print(f"Episodic Loss: {loss.item()}")
            print(f"Episode Length: {end_step}")
            print("\n-------------------------------------------\n")

            x = np.linspace(0, episode, episode+1)
            y.append(np.mean(total_episodic_return))
            if episode % 10 == 0:
                plt.plot(x, y)
                plt.pause(0.05)
        plt.show()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)


    
        