# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 23:51:09 2020

@author: Aamod Save
"""

from unityagents import UnityEnvironment
import numpy as np
from networks import MLP
import torch

#Path to environment
path_to_env = "banana_env/Banana.exe"

#Path to saved model
model_path = "trained_models/model"

#Number of test episodes
n_test = 5
t_max = 1000

#Main function
if __name__ == '__main__':
    
    #Load the environment
    env = UnityEnvironment(path_to_env)
    
    #Get the banana brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    #Reset the env to get state and action space
    env_info = env.reset()[brain_name]

    #Get the state an actions spaces
    state_space = len(env_info.vector_observations[0])
    action_space = brain.vector_action_space_size
    
    #Create the network to load saved model parameters
    agent = MLP(state_space, action_space)
    
    #Load the saved model
    agent.load_state_dict(torch.load(model_path))
    
    #Watch the smart agent play
    for n in range(n_test):
        
        rewards = 0
        
        #Reset the env
        env_info = env.reset()[brain_name]
        state = env_info.vector_observations[0]
        
        done = False
        
        for t in range(t_max):
        
            #Choose the best action 
            Q_values = agent(torch.from_numpy(state).float())
            action = np.argmax(list(Q_values.detach())).astype(int)
            
            #Perform the action
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            
            state = next_state
            rewards += reward
            
            if done:
                print("Episode finished in {} timesteps!".format(t))
                print("Rewards earned: {}".format(rewards))
                break
            
    #Close the environment
    env.close()