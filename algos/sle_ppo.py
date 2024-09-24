import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import multiprocess as mp
import copy
import math
import os
from datetime import date

from processing.batching import batchify, batchify_obs, unbatchify
from processing.l0module import L0GateLayer1d, concat_first_linear, concat_middle_linear, concat_last_linear,  \
    compress_first_linear, compress_middle_linear, compress_final_linear
from policies.centralized_policy import CentralizedPolicy
from policies.multitask_policy import MultiTaskPolicy

from processing.normalizer import NormalizeObservation, NormalizeReward
from processing.mating import pairwise_scores, probability_distribution, sample_mates

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
    

class SLE_MTPPO():

    def __init__(
            self,
            env,
            config
    ):
        self.env = env
        self.seed = env.seed
        self.obs_shape = env.observation_space.shape[0]
        self.device = config['device']
        self.name = 'sle-mtppo'
        self.hidden_size = config['hidden_size']

        self.pop = [
            MultiTaskPolicy(
                env = env,
                num_tasks = len(env.tasks),
                hidden_size = config['hidden_size'],
                continuous = config['continuous'],
                device = config['device']
            ).to(config['device'])
            for _ in range(config['pop_size'])
        ]

        self.max_grad_norm = 0.5
        self.normalize_states = config['normalize_states']
        self.normalize_values = config['normalize_values']
        self.normalize_rewards = config['normalize_rewards']
        self.max_cycles = config['max_path_length']
        self.pop_size = config['pop_size']
        self.total_episodes = config['total_episodes']
        self.epoch_opt = config['epoch_opt']
        self.epoch_merging = config['epoch_merging']
        self.epoch_finetune = config['epoch_finetune']
        self.batch_size = config['batch_size']
        self.min_batch = config['min_batch']
        self.discount = config['discount']
        self.gae_lambda = config['gae_lambda']
        self.clip_coef = config['lr_clip_range']
        self.ent_coef = config['ent_coef']
        self.vf_coef = config['vf_coef']
        self.continuous = config['continuous']
        self.lr = config['lr']
        self.num_tasks = len(env.tasks)

        # GA hyperparameters
        self.mutation_rate = 0.003
        self.mutation_mean = 0.0
        self.mutation_std = 0.01
        self.fitness = None
    
    """ TRAINING LOGIC """
    def eval(self, env, policy, norm_obs, norm_rew):
        task_returns = []
        success_tracker_eval = MultiTaskSuccessTracker(self.num_tasks)
        policy.eval()
        with torch.no_grad():
            for i, task in enumerate(env.tasks): # 10
                episodic_return = []
                    
                for epoch in range(5):      

                    next_obs, infos = task.reset(self.seed)
                    if self.normalize_states:
                        next_obs = norm_obs.normalize(torch.FloatTensor(next_obs), i)
                        next_obs = torch.clip(next_obs, -10, 10)

                    one_hot_id = torch.diag(torch.ones(self.num_tasks))[i]
                    step_return = 0
                    if_success = False
                    for step in range(0, self.max_cycles): # 500
                        # rollover the observation 
                        # obs = batchify_obs(next_obs, self.device)
                        obs = torch.FloatTensor(next_obs)
                        obs = torch.concatenate((obs, one_hot_id), dim=-1).to(self.device)

                        # get actions from skills
                        action, logprob, entropy, value = policy.get_action_and_value(obs)

                        # execute the environment and log data
                        action = torch.clip(action, -1, 1)  # action clip
                        next_obs, reward, term, trunc, info = task.step(action.cpu().numpy())
                        if self.normalize_rewards:
                            reward = norm_rew.normalize(reward, term, i)
                            reward = torch.clip(reward, -10, 10)
                        step_return += reward

                        success = info.get('success', 0.0)
                        if success != 0.0:
                            if_success = True
                            break

                        if self.normalize_states:
                            next_obs = norm_obs.normalize(torch.FloatTensor(next_obs), i)
                            next_obs = torch.clip(next_obs, -10, 10)

                        
                                
                    episodic_return.append(step_return)
                    success_tracker_eval.update(i, if_success)
                task_returns.append(np.mean(episodic_return))
        return task_returns, success_tracker_eval.overall_success_rate()
            
    def get_fitness(self, pop: list, norm_obs_list: list, norm_rew_list: list):
        print("Getting fitness for each agent of population...")
        ######
        '''
        envs = [copy.deepcopy(self.env) for _ in range(self.pop_size)]
        pool = mp.Pool()
        process_inputs = [(envs[i], pop[i], norm_obs_list[i], norm_rew_list[i]) for i in range(self.pop_size)]
        results = pool.starmap(self.eval, process_inputs)
        policies_fitness = [res[0] for res in results]  # receive from multi-process
        success_rates = [res[1] for res in results]  # receive from multi-process
        '''
        ######
        policies_fitness = []
        success_rates = []
        for i, agent in enumerate(pop):
            fitness, sr = self.eval(self.env, agent, norm_obs_list[i], norm_rew_list[i])
            policies_fitness.append(fitness)
            success_rates.append(sr)
            print(f"Agent {i} evaluating complete.")
        return np.asarray(policies_fitness), success_rates


    def select(self, pop: list, fitness: np.array) -> torch.nn.ModuleList:
        score_matrix = pairwise_scores(torch.from_numpy(fitness))
        prob_matrix = probability_distribution(score_matrix)
        print("Score matrix: \n", score_matrix)
        print("Prob matrix: \n", prob_matrix)
        mates, mate_indices = sample_mates(pop, prob_matrix)
        return mates, mate_indices


    def crossover(self, parent1: torch.nn.Sequential, parent2: torch.nn.Sequential) -> torch.nn.Sequential:
        child = nn.Sequential(
            concat_first_linear(parent1[0], parent2[0]),
            nn.Tanh(),
            L0GateLayer1d(n_features=self.hidden_size*2),
            concat_middle_linear(parent1[2], parent2[2]),
            nn.Tanh(),
            L0GateLayer1d(n_features=self.hidden_size*2),
            concat_last_linear(parent1[-1], parent2[-1]),
        )
        return child


    def mutate(self, child, mean=0.0, std=0.01):
        for module in child.modules():
            if np.random.rand() < self.mutation_rate:
                if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                    with torch.no_grad():
                        module.weight.add_(torch.randn(module.weight.size()) * std + mean)
                        if module.bias is not None:
                            module.bias.add_(torch.randn(module.bias.size(), device=module.bias.device) * std + mean)
        return child
    

    def evolve(self):
        x = []
        y = []
        z = []
        sr = []
        y_pop = []
        sr_pop = []
        fitness_pop = []
        gen_mates = []
        norm_obs_list = [NormalizeObservation(self.num_tasks) for _ in range(self.pop_size)]
        norm_rew_list = [NormalizeReward(self.num_tasks) for _ in range(self.pop_size)]
        # Generations, 4000
        for episode in range(self.total_episodes):
            pop = copy.deepcopy(self.pop)

            # fitness func / evaluation: currently using rewards as fitness for each individual
            
            fitness, success_rate = self.get_fitness(pop, norm_obs_list, norm_rew_list)
            fitness_pop.append(fitness)
            sr.append(np.max(success_rate))
            sr_pop.append(success_rate)
            self.fitness = fitness
            print(f"Training episode {episode}")
            print(f"Evaluation return: {fitness}")
            print(f"Evaluation success rate: {success_rate}")

            # mate selection: for each actor in the pop, select one mate from the pop respect to the rule.
            mates, mate_indices = self.select(pop, fitness)
            print(f"Correspendent mates: {mate_indices}")
            gen_mates.append(mate_indices)

            # for each individual,
            for i, policy in enumerate(pop):
                # give a probability that crossover happens by merging two actors' networks.
                child = self.crossover(policy.shared_layers, mates[i].shared_layers)
                child = self.mutate(child, self.mutation_mean, self.mutation_std)
                self.pop[i].shared_layers = child
            print("Finished crossover and mutation.")
            ################################ Training ##################################
            '''
            envs = [copy.deepcopy(self.env) for _ in range(self.pop_size)]
            pool = mp.Pool()
            process_inputs = [(envs[i], self.pop[i]) for i in range(self.pop_size)]
            results = pool.starmap(self.train, process_inputs)

            self.pop = [res[0] for res in results] # receive from multi-process
            seeds_episodic_return = [res[1] for res in results]  # receive from multi-process
            seeds_episodic_sr = [res[2] for res in results]  # receive from multi-process
            seeds_loss = [res[3] for res in results]  # receive from multi-process
            norm_obs_list = [res[4] for res in results]
            norm_rew_list = [res[5] for res in results]
            '''
            trained_pop = []
            seeds_episodic_return = []
            seeds_episodic_sr = []
            seeds_loss = []
            norm_obs_list = []
            norm_rew_list = []
            print(f"Episode {episode} training begin...")
            for i, agent in enumerate(self.pop):
                trained_agent, episodic_return, episodic_sr, loss, nb, nr = self.train(self.env, agent)
                trained_pop.append(trained_agent)
                seeds_episodic_return.append(episodic_return)
                seeds_episodic_sr.append(episodic_sr)
                seeds_loss.append(loss)
                norm_obs_list.append(nb)
                norm_rew_list.append(nr)
                print(f"Agent {i} training complete.")
            self.pop = trained_pop
            print("New population is generated!")
            ################################ Training ##################################
            
            
            print(f"Episodic return: {seeds_episodic_return}")
            print(f"Episodic max return: {np.max(seeds_episodic_return)}")
            print(f"Episodic success rate: {seeds_episodic_sr}")
            print(f"Episodic max success rate: {np.max(seeds_episodic_sr)}")
            print(f"Episodic loss: {np.mean(seeds_loss)}")
            print("\n-------------------------------------------\n")

            x.append(episode)
            y.append(np.max(seeds_episodic_return))
            z.append(np.max(seeds_episodic_sr))
            y_pop.append(seeds_episodic_return)
            
            if episode % 10 == 0:
                self.logging(y, sr, y_pop, fitness_pop, sr_pop, gen_mates)
                plt.plot(x, z)
                plt.pause(0.05)
        plt.show()
        return x, y, sr, y_pop, fitness_pop, sr_pop, gen_mates


    def train(self, env, policy):
        print("Start merging with mates...")
        policy = self.train_merging_stage(env, policy)
        print("Done.")
        important_indices1 = policy.shared_layers[2].important_indices()
        important_indices2 = policy.shared_layers[5].important_indices()
        policy.shared_layers = nn.Sequential(
            compress_first_linear(policy.shared_layers[0], important_indices1),
            nn.Tanh(),
            compress_middle_linear(policy.shared_layers[3], important_indices1, important_indices2),
            nn.Tanh(),
            compress_final_linear(policy.shared_layers[-1], important_indices2)
        )
        print("Finished compressing big agent. Start finetuning...")
        policy, mean_episodic_return, episodic_success_rate, loss, norm_obs, norm_rew = self.train_finetune_stage(env, policy)
                
        return policy, mean_episodic_return, episodic_success_rate, loss, norm_obs, norm_rew # a tuple of 6
                

    def train_merging_stage(self, env, policy):

        policy = copy.deepcopy(policy).to(self.device)
        opt = optim.Adam(policy.non_gate_parameters(), lr=self.lr, eps=1e-5)
        opt_gate = optim.SGD(policy.gate_parameters(), lr=0.001, momentum=0.9)
        alpha = 0.01

        policy.train()
        # clear memory
        rb_obs = torch.zeros((self.batch_size, self.obs_shape + self.num_tasks)).to(self.device)
        if self.continuous == True:
            rb_actions = torch.zeros((self.batch_size, env.action_space.shape[0])).to(self.device)
        else:
            rb_actions = torch.zeros((self.batch_size, 1)).to(self.device)
        rb_logprobs = torch.zeros((self.batch_size, 1)).to(self.device)
        rb_rewards = torch.zeros((self.batch_size, 1)).to(self.device)
        rb_advantages = torch.zeros((self.batch_size, 1)).to(self.device)
        rb_terms = torch.zeros((self.batch_size, 1)).to(self.device)
        rb_values = torch.zeros((self.batch_size, 1)).to(self.device)
        
        '''
        # learning Rate annealing
        frac = (episode - 1.0) / self.total_episodes
        new_lr = self.lr * (1.0 - frac)
        new_lr = max(new_lr, 0.0)
        self.opt.param_groups[0]["lr"] = new_lr
        '''
            
        # sampling
        index = 0
            
        task_returns = []
        success_tracker = MultiTaskSuccessTracker(self.num_tasks)
        norm_obs = NormalizeObservation(self.num_tasks)
        norm_rew = NormalizeReward(self.num_tasks)
            
        with torch.no_grad():
            for i, task in enumerate(env.tasks): # 10
                episodic_return = []
                    
                for epoch in range(int((self.batch_size / self.num_tasks) / self.max_cycles)): # 10         

                    next_obs, infos = task.reset(self.seed)
                    if self.normalize_states:
                        next_obs = norm_obs.normalize(torch.FloatTensor(next_obs), i)
                        next_obs = torch.clip(next_obs, -10, 10)

                    one_hot_id = torch.diag(torch.ones(self.num_tasks))[i]
                        
                    step_return = 0
                    if_success = False
                    for step in range(0, self.max_cycles): # 500
                        # rollover the observation 
                        # obs = batchify_obs(next_obs, self.device)
                        obs = torch.FloatTensor(next_obs)
                        obs = torch.concatenate((obs, one_hot_id), dim=-1).to(self.device)

                        # get actions from skills
                        action, logprob, entropy, value = policy.get_action_and_value(obs)
                        # add to episode storage
                        rb_actions[index] = action
                        rb_logprobs[index] = logprob
                        rb_values[index] = value.flatten()

                        # execute the environment and log data
                        action = torch.clip(action, -1.0, 1.0)  # action clip
                        next_obs, reward, term, trunc, info = task.step(action.cpu().numpy())
                        step_return += reward
                            

                        success = info.get('success', 0.0)
                        if success != 0.0:
                            if_success = True
                            term = 1.0
                            next_obs, infos = task.reset(self.seed)

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
                    success_tracker.update(i, if_success)

                    # advantage
                    gae = 0
                    for t in range(index-2, (index-self.max_cycles)-1, -1):
                        delta = rb_rewards[t] + self.discount * rb_values[t + 1] * rb_terms[t + 1] - rb_values[t]
                        gae = delta + self.discount * self.gae_lambda * rb_terms[t] * gae
                        rb_advantages[t] = gae

                        
                task_returns.append(np.mean(episodic_return))

        # normalize advantaegs
        rb_returns = rb_advantages + rb_values

        # Optimizing the policy and value network
         
        rb_index = np.arange(rb_obs.shape[0])
        clip_fracs = []
        for epoch in range(self.epoch_opt): # 4
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
                _, newlogprob, entropy, values = policy.get_action_and_value(x = rb_obs[batch_index, :], action = old_actions)
                    
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

                l0_loss = alpha * policy.l0_loss()
                # print(l0_loss)

                loss = loss + l0_loss
                    
                opt.zero_grad()
                opt_gate.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                opt.step()
                opt_gate.step()


            alpha += 0.05 * math.sqrt(alpha)
        
        return policy

    def train_finetune_stage(self, env, policy):

        policy = copy.deepcopy(policy).to(self.device)
        opt = optim.Adam(policy.parameters(), lr=self.lr, eps=1e-5)

        policy.train()
        # clear memory
        rb_obs = torch.zeros((self.batch_size, self.obs_shape + self.num_tasks)).to(self.device)
        if self.continuous == True:
            rb_actions = torch.zeros((self.batch_size, env.action_space.shape[0])).to(self.device)
        else:
            rb_actions = torch.zeros((self.batch_size, 1)).to(self.device)
        rb_logprobs = torch.zeros((self.batch_size, 1)).to(self.device)
        rb_rewards = torch.zeros((self.batch_size, 1)).to(self.device)
        rb_advantages = torch.zeros((self.batch_size, 1)).to(self.device)
        rb_terms = torch.zeros((self.batch_size, 1)).to(self.device)
        rb_values = torch.zeros((self.batch_size, 1)).to(self.device)
        
        '''
        # learning Rate annealing
        frac = (episode - 1.0) / self.total_episodes
        new_lr = self.lr * (1.0 - frac)
        new_lr = max(new_lr, 0.0)
        self.opt.param_groups[0]["lr"] = new_lr
        '''
            
        # sampling
        index = 0
            
        task_returns = []
        success_tracker = MultiTaskSuccessTracker(self.num_tasks)
        norm_obs = NormalizeObservation(self.num_tasks)
        norm_rew = NormalizeReward(self.num_tasks)
            
        with torch.no_grad():
            for i, task in enumerate(env.tasks): # 10
                episodic_return = []
                    
                for epoch in range(int((self.batch_size / self.num_tasks) / self.max_cycles)): # 10         

                    next_obs, infos = task.reset(self.seed)
                    if self.normalize_states:
                        next_obs = norm_obs.normalize(torch.FloatTensor(next_obs), i)
                        next_obs = torch.clip(next_obs, -10, 10)

                    one_hot_id = torch.diag(torch.ones(self.num_tasks))[i]
                        
                    step_return = 0
                    if_success = False
                    for step in range(0, self.max_cycles): # 500
                        # rollover the observation 
                        # obs = batchify_obs(next_obs, self.device)
                        obs = torch.FloatTensor(next_obs)
                        obs = torch.concatenate((obs, one_hot_id), dim=-1).to(self.device)

                        # get actions from skills
                        action, logprob, entropy, value = policy.get_action_and_value(obs)
                        # add to episode storage
                        rb_actions[index] = action
                        rb_logprobs[index] = logprob
                        rb_values[index] = value.flatten()

                        # execute the environment and log data
                        action = torch.clip(action, -1.0, 1.0)  # action clip
                        next_obs, reward, term, trunc, info = task.step(action.cpu().numpy())
                        step_return += reward
                            

                        success = info.get('success', 0.0)
                        if success != 0.0:
                            if_success = True
                            term = 1.0
                            next_obs, infos = task.reset(self.seed)

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
                    success_tracker.update(i, if_success)

                    # advantage
                    gae = 0
                    for t in range(index-2, (index-self.max_cycles)-1, -1):
                        delta = rb_rewards[t] + self.discount * rb_values[t + 1] * rb_terms[t + 1] - rb_values[t]
                        gae = delta + self.discount * self.gae_lambda * rb_terms[t] * gae
                        rb_advantages[t] = gae

                        
                task_returns.append(np.mean(episodic_return))

        # normalize advantaegs
        rb_returns = rb_advantages + rb_values

        # Optimizing the policy and value network
         
        rb_index = np.arange(rb_obs.shape[0])
        clip_fracs = []
        for epoch in range(self.epoch_opt): # 4
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
                _, newlogprob, entropy, values = policy.get_action_and_value(x = rb_obs[batch_index, :], action = old_actions)
                    
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
                    
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                opt.step()
        
        return policy, np.mean(task_returns), success_tracker.overall_success_rate(), loss.item(), norm_obs, norm_rew


    def save(self, path):
        fitness = torch.sum(self.fitness, dim=-1)
        torch.save(self.pop[torch.argmax(fitness, dim=-1)].state_dict(), path)
    
    def logging(self, y, sr, y_pop, fitness_pop, sr_pop, gen_mates):
        
        path_to_exp = f"./logs/{self.name}_{self.num_tasks}_{self.batch_size}_{self.epoch_opt}_{self.total_episodes}_{date.today()}"
        os.makedirs(path_to_exp, exist_ok=True)
        np.save(f"{path_to_exp}/algo_returns.npy", np.array(y))
        np.save(f"{path_to_exp}/algo_sr.npy", np.array(sr))
        np.save(f"{path_to_exp}/pop_returns.npy", np.array(y_pop))
        np.save(f"{path_to_exp}/pop_sr.npy", np.array(sr_pop))
        np.save(f"{path_to_exp}/pop_fitness.npy", np.array(fitness_pop))
        np.save(f"{path_to_exp}/gen_mates.npy", np.array(gen_mates))