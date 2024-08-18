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


    def get_fitness(self, rewards):
        fitness = rewards
        return fitness


    def select(self, pop, fitness):
        idx = np.random.choice(np.arange(self.n_agents), size=self.n_agents, p=fitness/fitness.sum())
        return pop[idx]


    def crossover(self, parent1, parent2, pop):
        offspring = 0
        return offspring


    def mutate(self, child):
        return child



    def train(self):
        plt.ion()
        
        for episode in range(self.n_generations):
            
            
            fitness = get_fitness(F_values)
            pop = select(pop, fitness)
            pop_copy = pop.copy()
            for parent in pop:
                child = crossover(parent, pop_copy)
                child = mutate(child)
                parent[:] = child


        plt.plot()
        plt.ioff();plt.show()

