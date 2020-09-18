# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 00:13:55 2020

@author: Aamod Save
"""

#Import necessary libraries
import numpy as np
from experience_replay import ReplayBuffer
from unityagents import UnityEnvironment
from torch.optim import Adam
import os

from maddpg_agent import Agent

import matplotlib.pyplot as plt

from collections import deque

#Initialize hyperparameters
lr_actor = 1.5e-4
lr_critic = 1.5e-4
weight_decay = 0.0001
batch_size = 128
buffer_size = int(1e5)
episodes = 400
tau = 1e-3
gamma = 0.99
t_max = 1000
update_every = 1
seed = 3

#Make a kwargs dictionary
kwargs = {'actor_optim': Adam, 'critic_optim': Adam, 'lr_actor': lr_actor,
          'lr_critic': lr_critic, 'tau': tau, 'seed': seed, 'weight_decay': weight_decay,
          'buffer_size': buffer_size, 'batch_size': batch_size}

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
    
    #Track rewards
    total_rewards = []
    rewards_window = deque(maxlen=100)
    
    print("Starting training...")
        
    env_solved = False
    
    #Start the training 
    for episode in range(1, episodes+1):
        
        #Reset the rewards every episode
        episodic_rewards = np.zeros(2)
        
        #Get the initial state
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        
        done = False   
        t = 0
                        
        for t in range(t_max):
                                                
            #Choose an action for both agents
            actions = agent.act(state)
            
            #Perform the action
            env_info = env.step(actions)[brain_name]
                        
            #Get agent observations
            next_state = env_info.vector_observations   
            reward = env_info.rewards                  
            done = env_info.local_done 
            
            experiences = []
            #Store agent1 obs
            for i in range(num_agents):
                experiences.append([state[i], actions[i], reward[i], next_state[i], done[i]])
            
            agent.step(experiences, gamma)
            
            #Update the state
            state = next_state
            
            episodic_rewards += reward
            
            #Update the agent every n time steps
            if t % update_every == 0 and len(agent.buffer[str(0)]) >= batch_size:
                
                #Train the agent
                agent.train(gamma)
            
            if done[0] or done[1]:
                break
            
        #Track rewards
        rewards_window.append(episodic_rewards)
        total_rewards.append(episodic_rewards)
        
        if episode % 100 == 0:
            print("Episode: {}, Rewards: {}, {}".format(episode, np.mean(rewards_window), np.mean(rewards_window)))
            agent.save(os.path.join(path_to_model, 'agent'))
    
            if np.mean(rewards_window) >= 0.5:
                print("Environment solved in {} number of episodes...".format(episode))
                env_solved = True
                
        if env_solved:
            break
            
    #Plot the rewards
    plot_rewards(total_rewards, episode)
    
    #Close the environment
    env.close()