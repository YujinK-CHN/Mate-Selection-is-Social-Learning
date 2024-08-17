import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HierAgent(nn.Module):
    def __init__(self, obs_shape, num_agents, num_skills, num_actions, obs_type):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_type = obs_type
        self.num_agents = num_agents
        self.num_skills = num_skills
        self.num_actions = num_actions

        self.encoder_rgb = nn.Sequential(
            self._layer_init(nn.Conv2d(self.obs_shape[-1], 32, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(64, 128, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            self._layer_init(nn.Linear(128 * 8 * 8, 512)),
            nn.ReLU(),
        )

        self.encoder_vector = nn.Sequential(
            self._layer_init(nn.Linear(self.obs_shape[0], 512)),
            nn.ReLU()
        )

        self.manager = nn.Sequential(
            self._layer_init(nn.Linear(512, num_skills), std=0.01),
            nn.ReLU()
        )

        self.skill = nn.Sequential(
            self._layer_init(nn.Linear(512, num_actions), std=0.01),
        )

        self.pop_managers = nn.ModuleList(
            [
                self.manager
                for _ in range(num_agents)
            ]
        )

        self.pop_skills = nn.ModuleList(
            [
                self.skill
                for _ in range(num_skills*num_agents)
            ]
        )
        
        self.critic = self._layer_init(nn.Linear(512, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def get_value(self, x):
        if self.obs_type == "rgb":
            # hidden shape [B, 512], B could be n_agents (sampling) or batch_size (training)
            return self.critic(self.encoder_rgb(x / 255.0))
        else:
            return self.critic(self.encoder_vector(x))

    def get_action_and_value(self, x, sampled_skill=None, action=None):
        if self.obs_type == "rgb":
            # hidden shape [B, 512], B could be n_agents (sampling) or batch_size (training)
            # transpose to be (num_agents, channel, height, width)
            x = x.permute(0, -1, 1, 2)
            hidden = self.encoder_rgb(x / 255.0)
        else:
            hidden = self.encoder_vector(x)

        # [5, 2]
        z = torch.stack(
            [
                manager(hidden[i, :])
                for i, manager in enumerate(self.pop_managers)
            ],
            dim=-2,
        )
        selections = torch.argmax(z, dim=-1) 

        # manager log probs
        manager_logits = torch.zeros((self.num_agents, self.num_skills)).to(device) # want [B, 2]
        manager_probs = Categorical(logits=manager_logits)
        if sampled_skill is None:
            sampled_skill = manager_probs.sample()


        # [5, 2, 3]
        a = torch.zeros((self.num_agents, self.num_skills, self.num_actions))
        i=0
        j=0
        for skill in self.pop_skills:
            if j == self.num_skills:
                i+=1
                j=0
            a[i, j, :] = skill(hidden[i, :])
            j+=1

        
        # skill log probs, each manager select one skill.
        skill_logits = torch.zeros((self.num_agents, self.num_actions)).to(device) # want [B, 3]
        for i in range(a.shape[0]):
            skill_logits[i, :] = a[i, selections[i], :]
            
        skill_probs = Categorical(logits=skill_logits)
        if action is None:
            action = skill_probs.sample()

        return action, sampled_skill, manager_probs.log_prob(sampled_skill), skill_probs.log_prob(action), skill_probs.entropy(), self.critic(hidden)