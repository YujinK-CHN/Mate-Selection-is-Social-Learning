#!/usr/bin/env python
# coding: utf-8
import torch
import torch.optim as optim
import numpy as np

from processing.batching import batchify, batchify_obs, unbatchify
from processing.l0module import L0GateLayer1d, concat_first_linear, concat_last_linear, compress_first_linear, \
    compress_final_linear
from loss.ppo_loss import clip_ppo_loss
from policies.independent_policy import IndependentPolicy
from policies.centralized_policy import CentralizedPolicy
import matplotlib.pyplot as plt

class Gippo():
    def __init__(
            self,
            env,
            config
    ):
        self.env = env
        self.device = config['device']
        self.name = 'ippo'
        self.policy = IndependentPolicy(
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
        self.batch_size = config['n_agents']
        self.gamma = config['gamma']
        self.clip_coef = config['clip_coef']
        self.ent_coef = config['ent_coef']
        self.vf_coef = config['vf_coef']

        self.crossover_rate = 0.8
        self.mutation_rate = 0.003
        self.n_generations = config['total_episodes']
        self.pop_size = config['n_agents'] * 2

    """ TRAINING LOGIC """
    def train(self):
        
        y = []
        end_step = 0
        total_episodic_return = 0
        rb_obs = torch.zeros((self.max_cycles, self.n_agents, self.obs_shape)).to(self.device)
        rb_actions = torch.zeros((self.max_cycles, self.n_agents, self.num_actions)).to(self.device)
        rb_logprobs = torch.zeros((self.max_cycles, self.n_agents)).to(self.device)
        rb_rewards = torch.zeros((self.max_cycles, self.n_agents)).to(self.device)
        rb_terms = torch.zeros((self.max_cycles, self.n_agents)).to(self.device)
        rb_values = torch.zeros((self.max_cycles, self.n_agents)).to(self.device)
        
        # train for n number of episodes
        for episode in range(self.total_episodes):
            # collect an episode
            with torch.no_grad():
                # collect observations and convert to batch of torch tensors
                next_obs, info = self.env.reset(seed=None)
                # reset the episodic return
                total_episodic_return = 0

                # each episode has num_steps
                for step in range(0, self.max_cycles):
                    # rollover the observation 
                    obs = batchify_obs(next_obs, self.device)

                    # get actions from skills
                    actions, logprobs, entropy, values = self.policy.act(obs)

                    # execute the environment and log data
                    next_obs, rewards, terms, truncs, infos = self.env.step(
                        unbatchify(actions, self.env)
                    )

                    # add to episode storage
                    rb_obs[step] = obs
                    rb_rewards[step] = batchify(rewards, self.device)
                    rb_terms[step] = batchify(terms, self.device)
                    rb_actions[step] = actions
                    rb_logprobs[step] = logprobs
                    rb_values[step] = values.flatten()
                    # compute episodic return
                    total_episodic_return += rb_rewards[step].cpu().numpy()

                    # if we reach termination or truncation, end
                    if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
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

            # convert our episodes to batch of individual transitions
            b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
            b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
            b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
            b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
            b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
            b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)


            # Optimizing the policy and value network
            b_index = np.arange(len(b_obs))
            clip_fracs = []
            
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            for start in range(0, len(b_obs), self.batch_size):
                # select the indices we want to train on
                end = start + self.batch_size
                batch_index = b_index[start:end]

                _, newlogprob, entropy, value = self.policy.evaluate(
                    x = b_obs[batch_index],
                    actions = b_actions.long()[batch_index]
                )

                logratio = newlogprob - b_logprobs[batch_index]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    ]

                # normalize advantaegs
                advantages = b_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -b_advantages[batch_index] * ratio
                pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.flatten()
                v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                v_clipped = b_values[batch_index] + torch.clamp(
                    value - b_values[batch_index],
                    -self.clip_coef,
                    self.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
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
            if episode % 100 == 0:
                plt.plot(x, y)
                plt.pause(0.05)

            ###########################################################################################
            fitness = self.get_fitness(total_episodic_return)
            pop = self.select(pop, fitness)
            pop_copy = pop.copy()
            for parent in pop:
                child = self.crossover(parent, pop_copy)
                child = self.mutate(child)
                parent[:] = child
            ###########################################################################################
        plt.show()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)


    def get_fitness(self, rewards):
        fitness = rewards
        return fitness


    def select(self, pop, fitness):
        idx = np.random.choice(np.arange(self.n_agents), size=self.n_agents, p=fitness/fitness.sum())
        return pop[idx]


    def crossover(self, parent1, pop):
        if np.random.rand() < self.crossover_rate:
            i_ = np.random.randint(0, self.n_agents, size=1)
            parent2 = pop[i_]
        return 0


    def mutate(self, child):
        return child


