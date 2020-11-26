# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 10:20:48 2020

@author: Aamod Save
"""

import os
import copy
import numpy as np
import random
import torch
from torch.nn.functional import mse_loss
from networks import Actor, Critic
from experience_replay import ReplayBuffer


class Agents:
    """
    Agents which will learn and act in the environment.
    """
    def __init__(self, state_space, action_space, num_agents, **kwargs):
        """
        Initialize all the agents
        
        Args:
            state_space: state dimensions
            action space: action dimensions
            num_agents: number of agents in te environment
            kwargs: additional parameters
        """
        self.state_space = state_space
        self.action_space = action_space
        self.num_agents = num_agents
        
        self.actor_optim = kwargs['actor_optim']
        self.critic_optim = kwargs['critic_optim']
        self.lr_actor = kwargs['lr_actor']
        self.lr_critic = kwargs['lr_critic']
        self.tau = kwargs['tau']
        self.seed = random.seed(kwargs['seed'])
        self.weight_decay = kwargs['weight_decay']
        self.batch_size = kwargs['batch_size']
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        for i in range(self.num_agents):
            self.actor_locals.update({str(i):Actor(self.state_space, 
                                                   self.action_space, 
                                                   kwargs['seed']).to(self.device)})
            
            self.actor_targets.update({str(i):Actor(self.state_space, 
                                                    self.action_space, 
                                                    kwargs['seed']).to(self.device)})
            
            self.actor_optimizers.update({str(i):self.actor_optim(self.actor_locals[str(i)].parameters(), 
                                                                  lr=self.lr_actor)})
    
            self.buffer.update({str(i): ReplayBuffer(kwargs['buffer_size'], 
                                                     kwargs['batch_size'], 
                                                     kwargs['seed'])})
            
        self.critic_local = Critic(self.state_space, 
                                   self.action_space, 
                                   kwargs['seed']).to(self.device)
        self.critic_target = Critic(self.state_space, 
                                    self.action_space, 
                                    kwargs['seed']).to(self.device)
        self.critic_optimizer = self.critic_optim(self.critic_local.parameters(), 
                                                  lr=self.lr_critic, 
                                                  weight_decay=self.weight_decay)
        
        self.noise = Noise(self.action_space, self.seed)
        
    def train(self, gamma=1.0):
        """
        One training iteration for all agents. Performed every UPDATE_EVERY time steps.
        The loss function used to train the critic networks is the temporal difference.
        
        Equation:
            Q = reward + gamma * (next_state_Q)
        
        Loss used for the actor networks is the output of the critic networks.
        
        After the local networks are updated, a soft update is performed on target networks.
        
        Args:
            gamma: Discount factor, type(float)
        """
        experiences = {}
        for i in range(self.num_agents):
            experiences.update({str(i): self.buffer[str(i)].sample_batch()})
        
        states = {}
        actions = {}
        rewards = {}
        next_states = {}
        dones = {}
        for i in range(self.num_agents):
            states.update({str(i): experiences[str(i)][0]})
            actions.update({str(i): experiences[str(i)][1]})
            rewards.update({str(i): experiences[str(i)][2]})
            next_states.update({str(i): experiences[str(i)][3]})
            dones.update({str(i): experiences[str(i)][4]})
        
        next_actions = {}
        for i in range(self.num_agents):
            next_actions.update({str(i): self.actor_targets[str(i)](next_states[str(i)])})

        #Update critic network
        for i in range(self.num_agents):
            q_next = self.critic_target(next_states[str(i)], next_actions[str(i)])
            q_targets = rewards[str(i)] + gamma * (q_next) * (1 - dones[str(i)])
            
            preds = self.critic_local(states[str(i)], actions[str(i)])
            
            critic_loss = mse_loss(preds, q_targets)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        
        #Update actor networks
        for i in range(self.num_agents):
            actor_preds = self.actor_locals[str(i)](states[str(i)])
            actor_loss = -self.critic_local(states[str(i)], actor_preds).mean()
            
            self.actor_optimizers[str(i)].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[str(i)].step()
            
        #Update target networks
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        
        for i in range(self.num_agents):
            self.soft_update(self.actor_locals[str(i)], self.actor_targets[str(i)], self.tau)
        
    def act(self, state, i, add_noise=False):
        """
        Perform action in the current state for all agents.
        
        Args:
            state: state of the agent in the environment
            i: agent number, type(int)
        Returns:
            Actions: actions performed by the agent
        """
        state = torch.from_numpy(state).float().to(self.device)
        
        self.actor_locals[str(i)].eval()
        
        with torch.no_grad():
            action = self.actor_locals[str(i)](state).unsqueeze(0).cpu().data.numpy()
            
        self.actor_locals[str(i)].train()
    
        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1)
    
    def step(self, experiences):
        """
        Store experiences in the buffer.
        
        Args:
            experience: type(list)
        """
        for i in range(len(experiences)):
            self.buffer[str(i)].store(experiences[i])
    
    def soft_update(self, local, target, tau=1e-3):
        """
        Perform soft update for the target networks.
        
        Equation:
            target_value = tau*(local_value) + (1-tau)*(target_value)
        
        Args:
            local: local network
            target: target network
        """
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(tau*(local_param.data) + (1.0-tau)*(target_param.data))
            
    def save(self, path):
        """
        Save the parameters for trained agents.
        
        Args:
            path: path to save directory, type(str)
        """
        for i in range(self.num_agents):
            torch.save(self.actor_locals[str(i)].state_dict(), os.path.join(path, 'actor_{}'.format(i)))
        
        torch.save(self.critic_local.state_dict(), os.path.join(path, 'critic'))
        
    def load(self, path):
        """
        Load the saved agents' parameters.
        
        Args:
            path: path to saved model, type(str)
        """
        for i in range(self.num_agents):
            self.actor_locals[str(i)].load_state_dict(torch.load(os.path.join(path, 'actor_{}'.format(i))))
            
        self.critic_local.load_state_dict(torch.load(os.path.join(path, 'critic')))

        
class Noise:
    """
    Add noise to the actions according to the Ornstein-Uhlenbeck process
    """
    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """
        Initialize the noise process
        
        Args:
            size: action length, type(ndarray)
            seed: random seed, type(int)
            mu: mean, type(float)
            theta: type(float)
            sigma: type(float)
        """
        self.mu = mu * np.ones(size)
        self.size = size
        self.seed = random.seed(seed)
        self.theta = theta
        self.sigma=sigma
        self.reset()
        
    def reset(self):
        """
        Reset the internal state of the noise.
        """
        self.state = copy.copy(self.mu)
        
    def sample(self):
        """
        Draw a noise sample.
        
        Returns:
            noise: noise signal
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
        
