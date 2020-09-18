# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 23:51:09 2020

@author: Aamod Save
"""

from unityagents import UnityEnvironment
import numpy as np

path_to_env = 'Reacher_Windows_x86_64/Reacher.exe'

env = UnityEnvironment(file_name=path_to_env)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

# reset the environment
env_info = env.reset(train_mode=True)[brain_name] 

# get the current state
state = env_info.vector_observations[0]            

# initialize the score
score = 0          
                                
while True:
    # select an action
    actions = np.random.randn(num_agents, action_size)
    actions = np.clip(actions, -1, 1)

    #Perform the action     
    env_info = env.step(actions)[brain_name]  

    #Get observations      
    next_state = env_info.vector_observations[0]   
    reward = env_info.rewards[0]                   
    done = env_info.local_done[0]                  
    score += reward                                
    state = next_state   

    #If episode is finished then break                          
    if done:                                       
        break
    
print("Score: {}".format(score))

env.close()