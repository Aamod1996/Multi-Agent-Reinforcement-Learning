# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 00:12:53 2020

@author: Aamod Save
"""

import numpy as np
import torch
import random

#Define the class replay buffer
class ReplayBuffer():
    
    """
    buffer: [(state, action, reward, next_state, done), ...]
    """
    
    def __init__(self, buffer_size, batch_size, seed):
        
        #Initialize buffer
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = random.seed(seed)
        
    def store(self, experience):
        
        #If buffer is full delete the oldest experience
        if len(self.memory) >= self.buffer_size:
            self.memory.pop(0)
        
        self.memory.append(experience)
        
    def sample_batch(self):
        
        #Sample a batch with batch_size number of observations
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.from_numpy(np.vstack([e[0] for e in batch if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in batch if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in batch if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in batch if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in batch if e is not None]).astype(np.uint8)).float().to(self.device)
    
        return (states, actions, rewards, next_states, dones)
        
    def clear_buffer(self):
        self.memory = []
        
    def __len__(self):
        return len(self.memory)