# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 00:13:55 2020

@author: Aamod Save
"""

#Import necessary libraries
import numpy as np
from unityagents import UnityEnvironment
from maddpg_agent import Agent
import matplotlib.pyplot as plt
from collections import deque
from torch.optim import Adam

#Initialize hyperparameters
n_test = 4
t_max = 100000

#Make a kwargs dictionary
kwargs = {'actor_optim': Adam, 'critic_optim': Adam, 'lr_actor': 0,
          'lr_critic': 0, 'tau': 0, 'seed': 0, 'weight_decay': 0,
          'buffer_size': 0, 'batch_size': 0}

#Specify the path to the environment
path_to_env = 'Tennis_Windows_x86_64/Tennis.exe'

#Specify model save path
path_to_model = 'trained_models/'
    
def plot_rewards(rewards, episodes):
    
    plt.figure()
    plt.plot(range(1, episodes+1), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title("Agent")
    plt.show()
    
#Main function
if __name__ == '__main__':
        
    #Load the environment
    env = UnityEnvironment(file_name=path_to_env)
    
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    
    #Get state space and action space
    state_space = len(env_info.vector_observations[0])
    action_space = brain.vector_action_space_size
    num_agents = env_info.vector_observations.shape[0]
                
    #Create the agent
    agent = Agent(state_space, action_space, num_agents, **kwargs)
    agent.load(path_to_model)
    
    #Track rewards
    total_rewards = []
    rewards_window = deque(maxlen=100)
    
    print("Watch a smart agent play...")
            
    #Start the training 
    for episode in range(1, n_test+1):
        
        #Reset the rewards every episode
        episodic_rewards = 0
        
        #Get the initial state
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        
        done = False   
        t = 0
                        
        for t in range(t_max):
            
            actions = []
            
            for i in range(num_agents):
                #Choose an action for both agents
                actions.append(agent.act(state[i], i))
            
            #Concatenate every action into one
            actions = np.concatenate(actions, axis=0)
                        
            #Perform the action
            env_info = env.step(actions)[brain_name]
                        
            #Get agent observations
            next_state = env_info.vector_observations   
            reward = env_info.rewards                  
            done = env_info.local_done 
            
            #Update the state
            state = next_state
            
            episodic_rewards += max(reward)
            
            if done[0] or done[1]:
                break
            
        #Track rewards
        rewards_window.append(episodic_rewards)
        total_rewards.append(episodic_rewards)
            
    #Plot the rewards
    plot_rewards(total_rewards, episode)
    
    #Close the environment
    env.close()