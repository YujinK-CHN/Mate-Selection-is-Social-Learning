#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

from processing.batching import batchify, batchify_obs, unbatchify
from processing.l0module import L0GateLayer1d, concat_first_linear, concat_middle_linear, concat_last_linear,  \
    compress_first_linear, compress_middle_linear, compress_final_linear
from loss.ppo_loss import clip_ppo_loss
from policies.independent_policy import IndependentPolicy
from policies.centralized_policy import CentralizedPolicy
import matplotlib.pyplot as plt

class GIPPO():
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

        self.pop_gates = nn.ModuleList([
            L0GateLayer1d(n_features=64)
            for _ in range(config['n_agents'])
        ])
        self.opt_gate = optim.Adam(self.pop_gates.parameters(), lr=0.01, eps=1e-5)

        self.max_cycles = config['max_cycles']
        self.n_agents = config['n_agents']
        self.num_actions = config['num_actions']
        self.obs_shape = config['obs_shape']
        self.curr_latent = None
        self.total_episodes = config['total_episodes']
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.alpha = 0.01
        self.clip_coef = config['clip_coef']
        self.ent_coef = config['ent_coef']
        self.vf_coef = config['vf_coef']
        self.continuous = config['continuous']

        # GA hyperparameters
        self.crossover_rate = 1.0
        self.mutation_rate = 0.003
        self.n_generations = config['total_episodes']
        self.pop_size = config['n_agents'] * 2
        self.evolution_period = 500
        self.merging_period = 250
        self.merging = None

    """ TRAINING LOGIC """
    def train(self):
        
        y = []
        end_step = 0
        total_episodic_return = [0 for _ in range(self.n_agents)]
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

            # after children born, use 250 eps for merging (train the gates).
            # then, compress the big student and finetune until next evolution.
            if episode % self.merging_period == 0 and self.merging == True:
                for i, gate in enumerate(self.pop_gates):
                    important_indices = gate.important_indices()
                    self.policy.pop_actors[i] = nn.Sequential(
                        compress_first_linear(self.policy.pop_actors[i][0], important_indices),
                        compress_middle_linear(self.policy.pop_actors[i][1], important_indices),
                        compress_final_linear(self.policy.pop_actors[i][1], important_indices),
                        nn.Softmax(dim=-1)
                    )
                print(self.policy.pop_actors)
                self.merging = False # means merging is done -> start to finetune.

            # evolve every 500 eps.
            if episode % self.evolution_period == 0:
                #################################### Genetic Algorithm ####################################

                # population: policy.pop_actors -> nn.ModuleList
                # fitness: currently using rewards as fitness for each agents
                fitness = self.get_fitness(total_episodic_return)

                # select: use fitness to compute a probs dist of networks in pop. 
                # randomly choice individual by the probs dist to reform pop. 
                # Meaning -> the actor with high rewards are more likely being selected as parents (e.g. the expert)
                # Selection + crossover + mutation together = full mate selection process to make offsprings.
                self.policy.pop_actors = self.select(self.policy.pop_actors, fitness)

                # for each individual,
                for i, actor in enumerate(self.policy.pop_actors):
                    # give a probability that crossover happens by merging two actors' networks.
                    child = self.crossover(actor, self.pop_gates[i], self.policy.pop_actors)
                    #child = self.mutate(child)
                    actor = child
                ###########################################################################################
                self.merging = True
            
            

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
                    actions, logprobs, entropy, values = self.policy.act(obs) # obs: [n, obs_shape]

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


            

            # Optimizing the policy and value network
            rb_index = np.arange(rb_obs.shape[0])
            
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
                    
                if self.merging == True:
                    loss = self.train_merging_stage(
                        newlogprob,
                        rb_logprobs[batch_index, :],
                        rb_advantages[batch_index, :],
                        values,
                        rb_returns[batch_index, :],
                        rb_values[batch_index, :],
                        entropy
                    )
                    self.alpha += 0.05 * np.sqrt(self.alpha).item()
                else:
                    self.alpha = 0.01
                    loss = self.train_finetune_stage(
                        newlogprob,
                        rb_logprobs[batch_index, :],
                        rb_advantages[batch_index, :],
                        values,
                        rb_returns[batch_index, :],
                        rb_values[batch_index, :],
                        entropy
                    )

            print(f"Training episode {episode}")
            print(f"Episodic Return: {total_episodic_return}")
            print(f"Episodic Mean Return: {np.mean(total_episodic_return)}")
            print(f"Episodic Loss: {loss.item()}")
            print(f"Episode Length: {end_step}")
            print("\n-------------------------------------------\n")

            x = np.linspace(0, episode, episode+1)
            y.append(np.mean(total_episodic_return))
            if episode % 10000 == 0:
                plt.plot(x, y)
                plt.pause(0.05)
        
            
                    

            
        plt.show()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)


    def get_fitness(self, rewards: list) -> np.array:
        # shift rewards to postive value.
        min_reward = np.min(rewards)
        shift_constant = np.abs(min_reward) + 1  # Shift to make the minimum reward 1
        scaled_rewards = np.asarray([r + shift_constant for r in rewards], dtype=np.float32)
        return scaled_rewards


    def select(self, pop: torch.nn.ModuleList, fitness: np.array) -> torch.nn.ModuleList:
        idx = np.random.choice(np.arange(self.n_agents), size=self.n_agents, p=fitness/fitness.sum())
        new_pop = copy.deepcopy(pop)
        for i, actor in enumerate(new_pop):
            actor = pop[idx[i]]
        return new_pop


    def crossover(self, parent1: torch.nn.Sequential, gate: nn.Module, pop: torch.nn.ModuleList) -> torch.nn.Sequential:
        if np.random.rand() < self.crossover_rate:
            i_ = np.random.randint(0, self.n_agents, size=1)
            parent2 = pop[i_.item()]
            child = nn.Sequential(
                concat_first_linear(parent1[0], parent2[0]),
                concat_middle_linear(parent1[1], parent2[1]),
                gate,
                concat_last_linear(parent1[-2], parent2[-2]),
                nn.Softmax(dim=-1)
            )

        return child


    def mutate(self, child):
        return child
    
    def train_merging_stage(
                        self,
                        newlogprob,
                        rb_logprobs,
                        rb_advantages,
                        values,
                        rb_returns,
                        rb_values,
                        entropy
                ):
                    logratio = newlogprob - rb_logprobs
                    ratio = logratio.exp()

                    # normalize advantaegs
                    advantages = rb_advantages
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                    # Policy loss
                    pg_loss1 = -rb_advantages * ratio
                    pg_loss2 = -rb_advantages * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    values = values.squeeze(-1)
                    v_loss_unclipped = (values - rb_returns) ** 2
                    v_clipped = rb_values + torch.clamp(
                        values - rb_values,
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - rb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    l0_loss = []
                    for gate in self.pop_gates:
                        l0_loss.append(self.alpha * gate.l0_loss().item())
                    l0_loss = torch.Tensor(l0_loss).mean().to(self.device)

                    loss = loss + l0_loss

                    self.opt.zero_grad()
                    self.opt_gate.zero_grad()
                    loss.backward()
                    self.opt.step()
                    self.opt_gate.step()

                    return loss
                

    def train_finetune_stage(
                        self,
                        newlogprob,
                        rb_logprobs,
                        rb_advantages,
                        values,
                        rb_returns,
                        rb_values,
                        entropy
                ):
                    logratio = newlogprob - rb_logprobs
                    ratio = logratio.exp()

                    # normalize advantaegs
                    advantages = rb_advantages
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                    # Policy loss
                    pg_loss1 = -rb_advantages * ratio
                    pg_loss2 = -rb_advantages * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    values = values.squeeze(-1)
                    v_loss_unclipped = (values - rb_returns) ** 2
                    v_clipped = rb_values + torch.clamp(
                        values - rb_values,
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - rb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()

                    return loss
