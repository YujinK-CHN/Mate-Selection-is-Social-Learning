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

class RewardsNormalizer:
    def __init__(self, num_tasks, epsilon=1e-8):
        self.mean = np.zeros(num_tasks)
        self.var = np.ones(num_tasks)
        self.count = np.zeros(num_tasks)
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
        std = np.sqrt(self.var[task_id] / (self.count[task_id] + self.epsilon))
        
        # Normalize the reward using the updated mean and standard deviation
        return (reward - self.mean[task_id]) / (std + self.epsilon)
    

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
                        actions, logprobs, entropy, values = policy.act(obs)

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
            
    def get_fitness(self, pop: list):
        pool = mp.Pool()
        results = pool.map(self.eval, pop)
        policies_fitness = [res[0] for res in results]  # receive from multi-process
        success_rates = [res[1] for res in results]  # receive from multi-process
        return np.asarray(policies_fitness), success_rates


    def select(self, pop: list, fitness: np.array) -> torch.nn.ModuleList:
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, p=np.mean(fitness, axis=-1)/np.sum(np.mean(fitness, axis=-1)))
        mates = [pop[i] for i in idx]
        return mates, idx


    def crossover(self, parent1: torch.nn.Sequential, parent2: torch.nn.Sequential) -> torch.nn.Sequential:
        child = nn.Sequential(
            concat_first_linear(parent1[0], parent2[0]),
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
                            module.bias.add_(torch.randn(module.bias.size()) * std + mean)
        return child
    

    def evolve(self):
        y = []
        sr = []
        y_pop = []
        sr_pop = []
        fitness_pop = []
        gen_mates = []
        # Generations, 4000
        for episode in range(self.total_episodes):
            pop = copy.deepcopy(self.pop)

            # fitness func / evaluation: currently using rewards as fitness for each individual
            fitness, success_rate = self.get_fitness(pop)
            fitness_pop.append(fitness)
            sr.append(np.mean(success_rate))
            sr_pop.append(success_rate)
            self.fitness = fitness
            print(f"Training episode {episode}")
            print(f"Evaluation return: {fitness}")
            print(f"Evaluation success rate: {np.mean(success_rate)}")

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
                
            ################################ Training ##################################
            env = [copy.deepcopy(self.env) for _ in range(self.pop_size)]
            pool = mp.Pool()
            process_inputs = [(env[i], self.pop[i]) for i in range(self.pop_size)]
            results = pool.starmap(self.train, process_inputs)

            self.pop = [res[0] for res in results] # receive from multi-process
            seeds_episodic_return = [res[1] for res in results]  # receive from multi-process
            seeds_episodic_sr = [res[2] for res in results]  # receive from multi-process
            seeds_actor_loss = [res[3] for res in results]  # receive from multi-process
            seeds_v_loss = [res[4] for res in results]  # receive from multi-process

            
            print(f"Episodic return: {np.mean(seeds_episodic_return)}")
            print(f"Episodic success rate: {np.mean(seeds_episodic_sr)}")
            print(f"Episodic actor loss: {np.mean(seeds_actor_loss)}")
            print(f"Episodic critic loss: {np.mean(seeds_v_loss)}")
            print("\n-------------------------------------------\n")

            x = np.linspace(0, episode, episode+1)
            y.append(np.mean(seeds_episodic_return))
            y_pop.append(seeds_episodic_return)
            
            if episode % 10 == 0:
                plt.plot(x, y)
                plt.pause(0.05)
        plt.show()
        return x, y, sr, y_pop, fitness_pop, sr_pop, gen_mates


    def train(self, env, policy):

        policy = self.train_merging_stage(env, policy)
        important_indices = policy.shared_layers[2].important_indices()
        policy.shared_layers = nn.Sequential(
            compress_first_linear(policy.shared_layers[0], important_indices),
            nn.Tanh(),
            compress_final_linear(policy.shared_layers[-1], important_indices),
        )

        policy, mean_episodic_return, episodic_success_rate, loss = self.train_finetune_stage(env, policy)
                
        return policy, mean_episodic_return, episodic_success_rate, loss # a tuple of 4
                

    def train_merging_stage(self, env, policy):

        policy = copy.deepcopy(policy).to(self.device)
        actor_opt = optim.Adam(policy.non_gate_parameters(), lr=self.lr, eps=1e-8)
        critic_opt = optim.Adam(policy.critic.parameters(), lr=self.lr, eps=1e-8)
        opt_gate = optim.SGD(policy.gate_parameters(), lr=0.001, momentum=0.9)
        alpha = 0.01

        policy.train()
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


        # sampling
        index = 0
            
        task_returns = []
        success_tracker = MultiTaskSuccessTracker(self.num_tasks)
        with torch.no_grad():
            for i, task in enumerate(env.tasks): # 10
                episodic_return = []
                for epoch in range(int((self.batch_size / self.num_tasks) / self.max_cycles)): # 10
                    next_obs, infos = task.reset(self.seed)
                    one_hot_id = torch.diag(torch.ones(self.num_tasks))[i]
                    step_return = 0
                    for step in range(0, self.max_cycles): # 500
                        # rollover the observation 
                        # obs = batchify_obs(next_obs, self.device)
                        obs = torch.FloatTensor(next_obs)
                        obs = torch.concatenate((obs, one_hot_id), dim=-1).to(self.device)

                        # get actions from skills
                        actions, logprobs, entropy, values = policy.act(obs, i)

                        # execute the environment and log data
                        next_obs, rewards, terms, truncs, infos = task.step(actions.cpu().numpy())
                        success = infos.get('success', 0.0)
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

                    # advantage
                    gae = 0
                    # normalize rewards
                    # rb_rewards[(index-self.max_cycles):(index-1)] = (rb_rewards[(index-self.max_cycles):(index-1)] - rb_rewards[(index-self.max_cycles):(index-1)].mean()) / (torch.std(rb_rewards[(index-self.max_cycles):(index-1)]) + 1e-8)
                    for t in range(index-2, (index-self.max_cycles)-1, -1):
                        delta = rb_rewards[t] + self.discount * rb_values[t + 1] * rb_terms[t + 1] - rb_values[t]
                        gae = delta + self.discount * self.gae_lambda * rb_terms[t] * gae
                        rb_advantages[t] = gae

                        
                task_returns.append(np.mean(episodic_return))                   

        # normalize advantaegs
        rb_returns = rb_advantages + rb_values

        # Optimizing the policy and value network
         
        rb_index = np.arange(rb_obs.shape[0])
        for epoch in range(self.epoch_opt): # 16
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
                    actions = old_actions,
                    task_id=i
                )
                    
                logratio = newlogprob.unsqueeze(-1) - rb_logprobs[batch_index, :]
                ratio = logratio.exp()

                    
                advantages = rb_advantages[batch_index, :]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)    

                # Policy loss
                pg_loss1 = advantages * ratio
                pg_loss2 = advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.min(pg_loss1, pg_loss2)
                pg_loss += self.ent_coef * entropy.unsqueeze(-1)

                actor_loss = -pg_loss.mean()

                v_loss = nn.MSELoss()(values, rb_returns[batch_index, :])

                l0_loss = alpha * policy.shared_layers[2].l0_loss()
                # print(l0_loss)

                actor_loss = actor_loss + l0_loss

                # Calculate gradients and perform backward propagation for actor and gate network
                actor_opt.zero_grad()
                opt_gate.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(policy.non_gate_parameters(), self.max_grad_norm)
                actor_opt.step()
                opt_gate.step()

                # Calculate gradients and perform backward propagation for critic network
                critic_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(policy.critic.parameters(), self.max_grad_norm)
                critic_opt.step()
        
        return policy

    def train_finetune_stage(self, env, policy):

        policy = copy.deepcopy(policy).to(self.device)
        actor_opt = optim.Adam(policy.shared_layers.parameters(), lr=self.lr, eps=1e-8)
        critic_opt = optim.Adam(policy.critic.parameters(), lr=self.lr, eps=1e-8)

        policy.train()
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


        # sampling
        index = 0
            
        task_returns = []
        success_tracker = MultiTaskSuccessTracker(self.num_tasks)
        with torch.no_grad():
            for i, task in enumerate(env.tasks): # 10
                episodic_return = []
                for epoch in range(int((self.batch_size / self.num_tasks) / self.max_cycles)): # 10
                    next_obs, infos = task.reset(self.seed)
                    one_hot_id = torch.diag(torch.ones(self.num_tasks))[i]
                    step_return = 0
                    for step in range(0, self.max_cycles): # 500
                        # rollover the observation 
                        # obs = batchify_obs(next_obs, self.device)
                        obs = torch.FloatTensor(next_obs)
                        obs = torch.concatenate((obs, one_hot_id), dim=-1).to(self.device)

                        # get actions from skills
                        actions, logprobs, entropy, values = policy.act(obs, i)

                        # execute the environment and log data
                        next_obs, rewards, terms, truncs, infos = task.step(actions.cpu().numpy())
                        success = infos.get('success', 0.0)
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

                    # advantage
                    gae = 0
                    # normalize rewards
                    # rb_rewards[(index-self.max_cycles):(index-1)] = (rb_rewards[(index-self.max_cycles):(index-1)] - rb_rewards[(index-self.max_cycles):(index-1)].mean()) / (torch.std(rb_rewards[(index-self.max_cycles):(index-1)]) + 1e-8)
                    for t in range(index-2, (index-self.max_cycles)-1, -1):
                        delta = rb_rewards[t] + self.discount * rb_values[t + 1] * rb_terms[t + 1] - rb_values[t]
                        gae = delta + self.discount * self.gae_lambda * rb_terms[t] * gae
                        rb_advantages[t] = gae

                        
                task_returns.append(np.mean(episodic_return))                   

        # normalize advantaegs
        rb_returns = rb_advantages + rb_values

        # Optimizing the policy and value network

        rb_index = np.arange(rb_obs.shape[0])
        for epoch in range(self.epoch_opt): # 16
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
                    actions = old_actions,
                    task_id=i
                )
                    
                logratio = newlogprob.unsqueeze(-1) - rb_logprobs[batch_index, :]
                ratio = logratio.exp()

                    
                advantages = rb_advantages[batch_index, :]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)    

                # Policy loss
                pg_loss1 = advantages * ratio
                pg_loss2 = advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.min(pg_loss1, pg_loss2)
                pg_loss += self.ent_coef * entropy.unsqueeze(-1)

                actor_loss = -pg_loss.mean()

                v_loss = nn.MSELoss()(values, rb_returns[batch_index, :])


                '''
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                '''
                # Calculate gradients and perform backward propagation for actor network
                actor_opt.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(policy.shared_layers.parameters(), self.max_grad_norm)
                actor_opt.step()

                # Calculate gradients and perform backward propagation for critic network
                critic_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(policy.critic.parameters(), self.max_grad_norm)
                critic_opt.step()
        
        return policy, np.mean(task_returns), success_tracker.overall_success_rate(), actor_loss.item(), v_loss.item()


    def save(self, path):
        fitness = torch.sum(self.fitness, dim=-1)
        torch.save(self.pop[torch.argmax(fitness, dim=-1)].state_dict(), path)