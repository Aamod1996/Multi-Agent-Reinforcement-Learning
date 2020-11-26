    # -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 00:12:53 2020

@author: Aamod Save
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Actor network for the actor-critic model.
    """
    def __init__(self, input_shape, output_shape, seed):
        """
        Initialize the actor network.
        
        Args:
            input_shape: input dimensions for the model
            output_shape: output dimensions for the model
            seed: random seed, type(int) 
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.seed = torch.manual_seed(seed)
        
        self.input = nn.Linear(self.input_shape, 600)
        self.fc1 = nn.Linear(600, 400)
        self.fc2 = nn.Linear(400, 200)
        self.output = nn.Linear(200, self.output_shape)
        
        self.initialize()
        
    def forward(self, x):
        """
        Perform the forward pass for the model.
        
        Args:
            x: input data samples
        """
        x = F.relu(self.input(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.output(x))
        return x
    
    def initialize(self):
        """
        Initialize the parameters in the network with uniform distribution.
        """
        self.input.weight.data.uniform_(*weight_init(self.input))
        self.fc1.weight.data.uniform_(*weight_init(self.fc1))
        self.fc2.weight.data.uniform_(*weight_init(self.fc2))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    
class Critic(nn.Module):
    """
    Critic network for the actor-critic model.
    """
    def __init__(self, input_shape, output_shape, seed):
        """
        Initialize the critic network.
        
        Args:
            input_shape: input dimensions for the model
            output_shape: output dimensions for the model
            seed: random seed, type(int) 
        """
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.seed = torch.manual_seed(seed)
                
        self.input = nn.Linear(self.input_shape+self.output_shape, 400)
        self.fc1 = nn.Linear(400, 300)
        self.output = nn.Linear(300, 1)
        
        self.initialize()
        
    def forward(self, state, action):
        """
        Perform the forward pass for the model.
        
        Args:
            x: input data samples
        """
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.input(x))
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x
    
    def initialize(self):
        """
        Initialize the parameters in the network with uniform distribution.
        """
        self.input.weight.data.uniform_(*weight_init(self.input))
        self.fc1.weight.data.uniform_(*weight_init(self.fc1))
        self.output.weight.data.uniform_(-3e-3, 3e-3)
    
    
def weight_init(layer):
    size = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(size)
    return (-lim, lim)
