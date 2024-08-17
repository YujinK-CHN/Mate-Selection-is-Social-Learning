import torch
import torch.nn as nn
import torch.optim as optim
from .independent_policy import IndependentPolicy

    
class HierPolicy_prompt(nn.Module):
    def __init__(
            self,
            env,
            config
    ):
        super().__init__()
        self.env = env
        self.name = 'HierPolicy_prompt'
        self.obs_shape = config['obs_shape']
        self.num_actions = config['num_actions']
        self.n_agents = config['n_agents']
        self.n_skills = config['n_skills']
        self.device = config['device']

        self.h_policy = IndependentPolicy(
            n_agents = config['n_agents'], 
            input_dim = config['obs_shape'],
            output_dim = config['n_skills'],
            continuous = config['continuous'],
            device = config['device']
        )

        self.l_policy = IndependentPolicy(
            n_agents = config['n_agents'], 
            input_dim = config['obs_shape'] + config['n_skills'],
            output_dim = config['num_actions'],
            continuous = config['continuous'],
            device = config['device']
        )

        self.h_opt = optim.Adam(self.h_policy.parameters(), lr=config['manager_lr'], eps=1e-5)
        self.l_opt = optim.Adam(self.l_policy.parameters(), lr=config['skill_lr'], eps=1e-5)

    def optimize(self, loss):
        self.h_opt.zero_grad()
        self.l_opt.zero_grad()
        loss.backward()
        self.h_opt.step()
        self.l_opt.step()

    def optimize_h(self, h_loss):
        self.h_opt.zero_grad()
        h_loss.backward()
        self.h_opt.step()

    def optimize_l(self, l_loss):
        self.l_opt.zero_grad()
        l_loss.backward()
        self.l_opt.step()


    def get_num_h_policy(self):
        return self.n_agents
    
    def get_num_l_policy(self):
        return self.n_agents
    
    def run(self, obs):
        latents, _, _, _ = self.h_policy.get_action_and_value(obs)
        actions, _, _, _ = self.l_policy.get_action_and_value(torch.concatenate((obs, latents), dim=-1))
        return actions
    