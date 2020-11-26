# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 00:13:55 2020

@author: Aamod Save
"""

import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from unityagents import UnityEnvironment
from maddpg_agent import Agent

N_TEST = 4
T_MAX = 100000
PATH_TO_ENV = 'Tennis_Windows_x86_64/Tennis.exe'
PATH_TO_MODEL = 'trained_models/'

kwargs = {'actor_optim': Adam, 'critic_optim': Adam, 'lr_actor': 0,
          'lr_critic': 0, 'tau': 0, 'seed': 0, 'weight_decay': 0,
          'buffer_size': 0, 'batch_size': 0}
    
def plot_rewards(rewards, episodes):
    """
    Plot the rewards gained vs number of episodes.
    
    Args:
        rewards: rewards gained per episode, type(list), len(episodes)
        episodes: total number of episodes, type(int)
    """
    plt.figure()
    plt.plot(range(1, episodes+1), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title("Agent")
    plt.show()
    
if __name__ == '__main__':
        
    env = UnityEnvironment(file_name=PATH_TO_ENV)
    brain_name = env.brain_names[0]  # Get the default brain
    brain = env.brains[brain_name]
    
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    state_space = len(state)
    action_space = brain.vector_action_space_size
    num_agents = env_info.vector_observations.shape[0]
    
    print('Number of agents:', num_agents)
    print('Number of actions:', action_space)
    print('States look like:', state)
    print('States have length:', state_space)
                
    agent = Agent(state_space, action_space, num_agents, **kwargs)
    agent.load(PATH_TO_MODEL)
    
    total_rewards = []
    
    print("Watch a smart agent play...")
            
    for episode in range(1, N_TEST+1):
        
        episodic_rewards = 0
        
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
                        
        for t in range(T_MAX):
            
            actions = []
            
            for i in range(num_agents):
                actions.append(agent.act(state[i], i))
            
            actions = np.concatenate(actions, axis=0)
                        
            env_info = env.step(actions)[brain_name]
                        
            next_state = env_info.vector_observations   
            reward = env_info.rewards                  
            done = env_info.local_done 
            state = next_state
            
            episodic_rewards += max(reward)
            
            if done[0] or done[1]:
                break
            
        total_rewards.append(episodic_rewards)
            
    plot_rewards(total_rewards, N_TEST)
    env.close()