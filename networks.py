    # -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 00:12:53 2020

@author: Aamod Save
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

#Define the class for network
class Actor(nn.Module):
    
    def __init__(self, input_shape, output_shape, seed):
        super().__init__()
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.seed = torch.manual_seed(seed)
        
        self.input = nn.Linear(self.input_shape, 600)
        self.fc1 = nn.Linear(600, 400)
        self.fc2 = nn.Linear(400, 200)
        self.output = nn.Linear(200, self.output_shape)
        
        #Initialize
        self.initialize()
        
    def forward(self, x):
        
        #Forward pass
        x = F.relu(self.input(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.output(x))
        return x
    
    #Initialize the network with our weight initializer
    def initialize(self):
        self.input.weight.data.uniform_(*weight_init(self.input))
        self.fc1.weight.data.uniform_(*weight_init(self.fc1))
        self.fc2.weight.data.uniform_(*weight_init(self.fc2))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    
class Critic(nn.Module):
    
    def __init__(self, input_shape, output_shape, seed):
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.seed = torch.manual_seed(seed)
        
        self.input = nn.Linear(self.input_shape, 400)
        self.fc1 = nn.Linear(400+self.output_shape, 300)
        self.output = nn.Linear(300, 1)
        
        #Initialize
        self.initialize()
        
    def forward(self, state, action):
        
        #Forward pass
        x = F.relu(self.input(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x
    
    #Initialize the network with our weight initializer
    def initialize(self):
        self.input.weight.data.uniform_(*weight_init(self.input))
        self.fc1.weight.data.uniform_(*weight_init(self.fc1))
        self.output.weight.data.uniform_(-3e-3, 3e-3)
    
def weight_init(layer):
    #Initialize the weights
    size = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(size)
    return (-lim, lim)