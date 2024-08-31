import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import multiprocess as mp
import copy

from processing.batching import batchify, batchify_obs, unbatchify
from processing.l0module import L0GateLayer1d, concat_first_linear, concat_middle_linear, concat_last_linear,  \
    compress_first_linear, compress_middle_linear, compress_final_linear
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


class SLE_MTPPO():

    def __init__(
            self,
            env,
            config
    ):
        self.env = env
        self.obs_shape = env.observation_space.shape[0]
        self.device = config['device']
        self.name = 'sle-mtppo'

        self.pop = [
            MultiTaskPolicy(
                pop_size = config['pop_size'], 
                env = env,
                num_tasks = len(env.tasks),
                continuous = config['continuous'],
                device = config['device']
            ).to(config['device'])
            for _ in range(config['pop_size'])
        ]

        

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
        self.lr = config['lr']

        # GA hyperparameters
        self.mutation_rate = 0.003
        self.evolution_period = 2000
        self.merging_period = 250
        self.fitness = None
    
    """ TRAINING LOGIC """
    def get_fitness(self, pop: list) -> torch.FloatTensor:
        policies_fitness = []
        for policy in pop:
            task_returns = []
            success_tracker_eval = MultiTaskSuccessTracker(len(self.env.tasks))
            policy.eval()
            with torch.no_grad():
                for i, task in enumerate(self.env.tasks):
                    # render 5 episodes out
                    episodic_return = []
                    for episode in range(5):
                        next_obs, infos = task.reset()
                        one_hot_id = torch.diag(torch.ones(len(self.env.tasks)))[i]
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
            policies_fitness.append(task_returns)
            
        return torch.FloatTensor(policies_fitness)


    def select(self, pop: list, fitness: np.array) -> torch.nn.ModuleList:
        return 0


    def crossover(self, parent1: torch.nn.Sequential, parent2: torch.nn.Sequential) -> torch.nn.Sequential:
        child = nn.Sequential(
            concat_first_linear(parent1[0], parent2[0]),
            nn.Tanh(),
            L0GateLayer1d(n_features=1024),
            concat_last_linear(parent1[-1], parent2[-1]),
        )
        return child


    def mutate(self, child):
        return child
    

    def evolve(self):
        y1 = []
        y2 = []
        y3 = []
        # Generations, 4000
        for episode in range(self.total_episodes):
            pop = copy.deepcopy(self.pop)

            # fitness func / evaluation: currently using rewards as fitness for each individual
            fitness, mean_success_rate = self.get_fitness(pop)
            self.fitness = fitness
            print(f"Training episode {episode}")
            print(f"Evaluation return: {fitness}")
            print(f"Evaluation success rate: {mean_success_rate}")

            # mate selection: for each actor in the pop, select one mate from the pop respect to the rule.
            mates, mate_indices = self.select(pop, fitness)
            print(f"Correspendent mates: {mate_indices}")

            # for each individual,
            for i, policy in enumerate(pop):
                # give a probability that crossover happens by merging two actors' networks.
                child = self.crossover(policy.shared_layers, mates[i].shared_layers)
                child = self.mutate(child)
                self.pop[i].shared_layers = child
                
            ################################ Training ##################################
            env = [copy.deepcopy(self.env) for _ in range(self.pop_size)]
            with mp.Pool(processes=16) as pool:
                process_inputs = [(env[i], self.pop[i]) for i in range(self.pop_size)]
                results = pool.starmap(self.train, process_inputs)

            self.pop = [res[0] for res in results] # receive from multi-process
            seeds_episodic_return = [res[1] for res in results]  # receive from multi-process
            seeds_episodic_sr = [res[2] for res in results]  # receive from multi-process
            seeds_loss = [res[3] for res in results]  # receive from multi-process

            
            print(f"Episodic return: {np.mean(seeds_episodic_return)}")
            print(f"Episodic success rate: {np.mean(seeds_episodic_sr)}")
            print(f"Episodic loss: {np.mean(seeds_loss)}")
            print("\n-------------------------------------------\n")

            x = np.linspace(0, episode, episode+1)
            y1.append(np.mean(seeds_episodic_return))
            y2.append(np.mean(seeds_episodic_sr))
            y3.append(np.mean(seeds_loss))
            if episode % 10 == 0:
                plt.plot(x, y1)
                plt.pause(0.05)
        plt.show()


    def train(self, env, policy):
        policy.train()
        policy = self.train_merging_stage(env, policy)

        important_indices = policy.shared_layers[2].important_indices()
        policy.shared_layers = nn.Sequential(
            compress_first_linear(policy.shared_layers[0], important_indices),
            nn.Tanh(),
            compress_final_linear(policy.shared_layers[-1], important_indices),
        )

        policy, mean_episodic_return, episodic_success_rate, loss = self.train_finetune_stage(env, policy)
                
        return (policy, mean_episodic_return, episodic_success_rate, loss) # a tuple of 4
                

    def train_merging_stage(self, env, policy):

        policy = copy.deepcopy(policy)
        opt = optim.Adam(policy.non_gate_parameters(), lr=self.lr, eps=1e-8)
        opt_gate = optim.SGD(policy.gate_parameters(), lr=0.001, momentum=0.9)
        alpha = 0.01

        rb_obs = torch.zeros((self.batch_size, self.obs_shape + len(env.tasks))).to(self.device)
        if self.continuous == True:
            rb_actions = torch.zeros((self.batch_size, env.action_space.shape[0])).to(self.device)
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
        success_tracker = MultiTaskSuccessTracker(len(env.tasks))
        with torch.no_grad():
            for i, task in enumerate(env.tasks): # 10
                episodic_return = []
                for epoch in range(int((self.batch_size / len(env.tasks)) / self.max_cycles)): 
                    next_obs, infos = task.reset()
                    one_hot_id = torch.diag(torch.ones(len(env.tasks)))[i]
                    step_return = 0
                    
                    for step in range(0, self.max_cycles): # 500
                        # rollover the observation 
                        #obs = batchify_obs(next_obs, self.device)
                        obs = torch.FloatTensor(next_obs)
                        obs = torch.concatenate((obs, one_hot_id), dim=-1).to(self.device)

                        # get actions from skills
                        actions, logprobs, entropy, values = policy.act(obs)

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
        for epoch in range(self.epoch_opt): # 256
            alpha += 0.05 * np.sqrt(alpha)
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
                _, newlogprob, entropy, values = policy.evaluate(
                    x = rb_obs[batch_index, :],
                    actions = old_actions
                )
                        
                logratio = newlogprob.unsqueeze(-1) - rb_logprobs
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
                v_loss_unclipped = (values - rb_returns) ** 2
                v_clipped = rb_values + torch.clamp(
                    values - rb_values,
                    -self.clip_coef,
                    self.clip_coef,
                )
                v_loss_clipped = (v_clipped - rb_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.max()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                l0_loss = alpha * policy.shared_layers[2].l0_loss()
                # print(l0_loss)

                loss = loss + l0_loss

                opt.zero_grad()
                opt_gate.zero_grad()
                loss.backward()
                opt.step()
                opt_gate.step()
        
        return policy

    def train_finetune_stage(self, env, policy):

        policy = copy.deepcopy(policy)
        opt = optim.Adam(policy.parameters(), lr=self.lr, eps=1e-8)

        rb_obs = torch.zeros((self.batch_size, self.obs_shape + len(env.tasks))).to(self.device)
        if self.continuous == True:
            rb_actions = torch.zeros((self.batch_size, env.action_space.shape[0])).to(self.device)
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
        success_tracker = MultiTaskSuccessTracker(len(env.tasks))
        with torch.no_grad():
            for i, task in enumerate(env.tasks): # 10
                episodic_return = []
                for epoch in range(int((self.batch_size / len(env.tasks)) / self.max_cycles)): 
                    next_obs, infos = task.reset()
                    one_hot_id = torch.diag(torch.ones(len(env.tasks)))[i]
                    step_return = 0
                    
                    for step in range(0, self.max_cycles):
                        # rollover the observation 
                        # obs = batchify_obs(next_obs, self.device)
                        obs = torch.FloatTensor(next_obs)
                        obs = torch.concatenate((obs, one_hot_id), dim=-1).to(self.device)

                        # get actions from skills
                        actions, logprobs, entropy, values = policy.act(obs)

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
                _, newlogprob, entropy, values = policy.evaluate(
                    x = rb_obs[batch_index, :],
                    actions = old_actions
                )
                        
                logratio = newlogprob.unsqueeze(-1) - rb_logprobs
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
                v_loss_unclipped = (values - rb_returns) ** 2
                v_clipped = rb_values + torch.clamp(
                    values - rb_values,
                    -self.clip_coef,
                    self.clip_coef,
                )
                v_loss_clipped = (v_clipped - rb_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.max()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                opt.zero_grad()
                loss.backward()
                opt.step()
        
        return policy, np.mean(task_returns), success_tracker.overall_success_rate(), loss.item()


    def save(self, path):
        fitness = torch.sum(self.fitness, dim=-1)
        torch.save(self.pop[torch.argmax(fitness, dim=-1)].state_dict(), path)