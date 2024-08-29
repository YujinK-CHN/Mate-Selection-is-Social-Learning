import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
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
        self.policy = MultiTaskPolicy(
            pop_size = config['pop_size'], 
            env = env,
            num_tasks = len(env.tasks),
            continuous = config['continuous'],
            device = config['device']
        ).to(config['device'])
        self.gate = L0GateLayer1d(n_features=1024) # reset at each time evolve.
        self.opt = optim.Adam(self.policy.parameters(), lr=config['lr'], eps=1e-5)
        self.opt_gate = optim.SGD(self.pop_gates.parameters(), lr=0.001, momentum=0.9)

        self.pop = nn.ModuleList([
                    copy.deepcopy(self.policy.shared_layers)
                    for _ in range(config['pop_size'])
                ])

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

        # GA hyperparameters
        self.alpha = 0.01
        self.mutation_rate = 0.003
        self.evolution_period = 2000
        self.merging_period = 250
        self.merging = None
    
    """ TRAINING LOGIC """

    def train(self):

        y1 = []
        y2 = []
        y3 = []
        
        # train for n number of episodes
        for episode in range(self.total_episodes): # 4000
            self.policy.train()

            # after children born, use 250 eps for merging (train the gates).
            # then, compress the big student and finetune until next evolution.
            if episode % self.merging_period == 0 and self.merging == True:
                important_indices = self.gate.important_indices()
                self.policy.shared_layers = nn.Sequential(
                    compress_first_linear(self.policy.shared_layers[0], important_indices),
                    compress_middle_linear(self.policy.shared_layers[1], torch.arange(512), important_indices),
                    nn.Tanh(),
                    compress_final_linear(self.policy.shared_layers[-2], important_indices),
                    nn.Tanh()
                )
                
                self.merging = False # means merging is done -> start to finetune.

            #################################### Genetic Algorithm ####################################
            # evolve every _ eps. 
            if episode % self.evolution_period == 0:
                

                # reset gates.
                self.gate = L0GateLayer1d(n_features=1024)

                # make kid: randomly mutate the main_charactor _ times and append those kids to population
                self.pop = self.make_kid(self.policy.shared_layers)
                
                # fitness func / evaluation: currently using rewards as fitness for each agents
                fitness = self.get_fitness(0)

                # kill bad: only keep half pop (e.g. 6 -> pop_size=3) top fitness
                self.pop = self.kill_bad(self.pop)

                # mate selection
                partner = self.select(self.pop, fitness)

                # crossover
                self.policy.shared_layers = self.crossover(self.policy.shared_layers, self.gate, partner)

                # merging button: ON
                self.merging = True
            ###########################################################################################

            # clear memory
            rb_obs = torch.zeros((self.batch_size, self.obs_shape)).to(self.device)
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
            episodic_return = []
            success_tracker = MultiTaskSuccessTracker(len(self.env.tasks))
            
            for epoch in range(int(self.batch_size / self.max_cycles)): # 5000 / 500 = 10
                # collect an episode
                with torch.no_grad():
                    # collect observations and convert to batch of torch tensors
                    next_obs, info = self.env.reset()
                    task_id = self.env.tasks.index(self.env.current_task)

                    step_return = 0
                    
                    # each episode has num_steps
                    for step in range(0, self.max_cycles):
                        # rollover the observation 
                        #obs = batchify_obs(next_obs, self.device)
                        obs = torch.FloatTensor(next_obs).to(self.device)

                        # get actions from skills
                        actions, logprobs, entropy, values = self.policy.act(obs, task_id)

                        # execute the environment and log data
                        next_obs, rewards, terms, truncs, infos = self.env.step(actions.cpu().numpy())
                        success = infos.get('success', False)
                        success_tracker.update(task_id, success)

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
                        if terms or truncs:
                            break

                        index += 1
                    episodic_return.append(step_return)

                    index += 1

                    
                    # skills advantage
                    with torch.no_grad():
                        gae = 0
                        for t in range(index-2, (index-self.max_cycles)-1, -1):
                            delta = rb_rewards[t] + self.discount * rb_values[t + 1] * rb_terms[t + 1] - rb_values[t]
                            gae = delta + self.discount * self.gae_lambda * rb_terms[t] * gae
                            rb_advantages[t] = gae
                            
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
                    _, newlogprob, entropy, values = self.policy.evaluate(
                        x = rb_obs[batch_index, :],
                        task_id = task_id,
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

            mean_eval_return, mean_success_rate = self.eval()
            
            print(f"Training episode {episode}")
            print(f"Episodic Return: {np.mean(episodic_return)}")
            print(f"Episodic success rate: {success_tracker.overall_success_rate()}")
            print(f"Evaluation Return: {mean_eval_return}")
            print(f"Evaluation success rate: {mean_success_rate}")
            print(f"Episodic Loss: {loss.item()}")
            #print(f"overall success rate: {success_tracker.overall_success_rate() * 100:.2f}")
            print("\n-------------------------------------------\n")

            x = np.linspace(0, episode, episode+1)
            y1.append(np.mean(episodic_return))
            y2.append(mean_eval_return)
            y3.append(success_tracker.overall_success_rate())
            if episode % 10 == 0:
                plt.plot(x, y1)
                plt.plot(x, y2)
                plt.plot(x, y3)
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
                terms = False
                truncs = False
                step_return = 0
                while not terms and not truncs:
                    # rollover the observation 
                    #obs = batchify_obs(next_obs, self.device)
                    obs = torch.FloatTensor(next_obs).to(self.device)

                    # get actions from skills
                    actions, logprobs, entropy, values = self.policy.act(obs, task_id)

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


    def make_kid(self, main_ch: torch.nn.Sequential) -> nn.ModuleList:
        # mutation: randomly mutate the main_charactor _ times and append those kids to population
        return self.pop
    

    def get_fitness(self, rewards: list) -> np.array:
        # shift rewards to postive value.
        min_reward = np.min(rewards)
        shift_constant = np.abs(min_reward) + 1  # Shift to make the minimum reward 1
        scaled_rewards = np.asarray([r + shift_constant for r in rewards], dtype=np.float32)
        return scaled_rewards
    
    def kill_bad(self, pop: torch.nn.ModuleList):
        return pop


    def select(self, pop: torch.nn.ModuleList, fitness: np.array) -> torch.nn.ModuleList:
        return pop


    def crossover(self, parent1: torch.nn.Sequential, gate: nn.Module, parent2: torch.nn.Sequential) -> torch.nn.Sequential:
        child = nn.Sequential(
            concat_first_linear(parent1[0], parent2[0]),
            concat_middle_linear(parent1[1], parent2[1]),
            nn.Tanh(),
            gate,
            concat_last_linear(parent1[-2], parent2[-2]),
            nn.Tanh(),
        )
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

                        l0_loss = self.gate.l0_loss()
                        # print(l0_loss)

                        loss = loss + l0_loss.mean()

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

                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()

                        return loss