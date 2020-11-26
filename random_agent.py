# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 23:51:09 2020

@author: Aamod Save
"""

import numpy as np
from unityagents import UnityEnvironment

PATH_TO_ENV = 'Reacher_Windows_x86_64/Reacher.exe'

if __name__ == "__main__":

    env = UnityEnvironment(file_name=PATH_TO_ENV)
    
    brain_name = env.brain_names[0]  # Get the default brain
    brain = env.brains[brain_name]
    
    env_info = env.reset(train_mode=True)[brain_name]
    
    num_agents = len(env_info.agents)    
    action_size = brain.vector_action_space_size    
    state = env_info.vector_observations[0]
    state_size = len(state)
    
    print('Number of agents:', num_agents)
    print('Number of actions:', action_size)
    print('States look like:', state)
    print('States have length:', state_size)
    
    env_info = env.reset(train_mode=True)[brain_name] 
    state = env_info.vector_observations[0]            
    
    score = 0          
                                    
    while True:
        actions = np.random.randn(num_agents, action_size)
        actions = np.clip(actions, -1, 1)
    
        env_info = env.step(actions)[brain_name]  
    
        next_state = env_info.vector_observations[0]   
        reward = env_info.rewards[0]                   
        done = env_info.local_done[0]                  
        score += reward                                
        state = next_state   
    
        if done:                                       
            break
        
    print("Score: {}".format(score))
    
    env.close()