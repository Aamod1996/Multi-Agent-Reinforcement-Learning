# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 10:20:48 2020

@author: Aamod Save
"""

from networks import Actor, Critic
import torch
import copy
from torch.nn.functional import mse_loss
import numpy as np
import os
import random
from experience_replay import ReplayBuffer

class Agent:
    
    def __init__(self, state_space, action_space, num_agents, **kwargs):
        
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
        
        #Set the device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        #Actor network
        self.actor_locals = {str(i): Actor(self.state_space, self.action_space, kwargs['seed']).to(self.device) for i in range(num_agents)}
        self.actor_targets = {str(i): Actor(self.state_space, self.action_space, kwargs['seed']).to(self.device) for i in range(num_agents)}
        self.actor_optimizers = {str(i): self.actor_optim(self.actor_locals[str(i)].parameters(), lr=self.lr_actor) for i in range(num_agents)}
        
        #Critic network
        self.critic_local = Critic(self.state_space, self.action_space, kwargs['seed']).to(self.device)
        self.critic_target = Critic(self.state_space, self.action_space, kwargs['seed']).to(self.device)
        self.critic_optimizer = self.critic_optim(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)
        
        #Make a replay buffer for every agent
        self.buffer = {str(i): ReplayBuffer(kwargs['buffer_size'], kwargs['batch_size'], kwargs['seed']) for i in range(num_agents)}
        
        #Noise
        self.noise = Noise(self.action_space, self.seed)
        
    def train(self, gamma=1.0):
        
        #Get observations from the memory
        experiences = {str(i): self.buffer[str(i)].sample_batch() for i in range(self.num_agents)}
        
        #Get the observations for all agents
        states = {str(i): experiences[str(i)][0] for i in range(self.num_agents)}
        actions= {str(i): experiences[str(i)][1] for i in range(self.num_agents)}
        rewards = {str(i): experiences[str(i)][2] for i in range(self.num_agents)}
        next_states = {str(i): experiences[str(i)][3] for i in range(self.num_agents)}
        dones = {str(i): experiences[str(i)][4] for i in range(self.num_agents)}
        
        #Get the next_actions for all agents using target actors
        next_actions = {str(i): self.actor_targets[str(i)](next_states[str(i)]) for i in range(self.num_agents)}

        #Choose experiences from all agents
        for i in range(self.num_agents):
            
            #Get target Q values for target critic nextwork
            targets = self.critic_target(next_states[str(i)], next_actions[str(i)])
            targets = rewards[str(i)] + gamma * (targets) * (1 - dones[str(i)])
            
            #Get predicted Q values for local critic
            preds = self.critic_local(states[str(i)], actions[str(i)])
            
            #Calculate the loss and train the critic
            critic_loss = mse_loss(preds, targets)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        
        #Calculate the actor loss for every agent        
        for i in range(self.num_agents):
            actor_preds = self.actor_locals[str(i)](states[str(i)])
            actor_loss = -self.critic_local(states[str(i)], actor_preds).mean()
            
            self.actor_optimizers[str(i)].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[str(i)].step()
            
        #Perform soft update for critic
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        
        #Perform soft update for every actor
        for i in range(self.num_agents):
            self.soft_update(self.actor_locals[str(i)], self.actor_targets[str(i)], self.tau)
        
    def act(self, state, i, add_noise=False):
        
        state = torch.from_numpy(state).float().to(self.device)
        
        #Set the actor to eval mode
        self.actor_locals[str(i)].eval()
        
        with torch.no_grad():
            action = self.actor_locals[str(i)](state).unsqueeze(0).cpu().data.numpy()
            
        #Set the actor in training mode
        self.actor_locals[str(i)].train()
    
        #Add noise to the action
        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1)
    
    def step(self, experiences, gamma=1.0):
        
        for i in range(len(experiences)):
            self.buffer[str(i)].store(experiences[i])
    
    def soft_update(self, local, target, tau=1e-3):
    
        #Perform a soft update of the network
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(tau*(local_param.data) + (1.0-tau)*(target_param.data))
            
    def save(self, path):
        for i in range(self.num_agents):
            torch.save(self.actor_locals[str(i)].state_dict(), os.path.join(path, 'actor_{}'.format(i)))
        
        torch.save(self.critic_local.state_dict(), os.path.join(path, 'critic'))
        
    def load(self, path):
        
        for i in range(self.num_agents):
            self.actor_locals[str(i)].load_state_dict(torch.load(os.path.join(path, 'actor_{}'.format(i))))
            
        self.critic_local.load_state_dict(torch.load(os.path.join(path, 'critic')))
        
#Add a class for Noise 
class Noise:
    
    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        
        self.mu = mu * np.ones(size)
        self.size = size
        self.seed = random.seed(seed)
        self.theta = theta
        self.sigma=sigma
        self.reset()
        
    def reset(self):
        #Reset the internal state
        self.state = copy.copy(self.mu)
        
    def sample(self):
        #Add noise according to Ornstein-Uhlenbeck process
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
        