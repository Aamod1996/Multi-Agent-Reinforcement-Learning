# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 00:13:55 2020

@author: Aamod Save
"""

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from torch.optim import Adam
from maddpg_agent import Agents
from unityagents import UnityEnvironment

LR_ACTOR = 1e-4
LR_CRITIC = 1e-4
WEIGHT_DECAY = 0.00001
BATCH_SIZE = 256
BUFFER_SIZE = int(1e5)
EPISODES = 10000
TAU = 1e-3
GAMMA = 0.995
T_MAX = 1000
UPDATE_EVERY = 4
SEED = 10

kwargs = {'actor_optim': Adam, 'critic_optim': Adam, 'lr_actor': LR_ACTOR,
          'lr_critic': LR_CRITIC, 'tau': TAU, 'seed': SEED, 'weight_decay': WEIGHT_DECAY,
          'buffer_size': BUFFER_SIZE, 'batch_size': BATCH_SIZE}

PATH_TO_ENV = 'Tennis_Windows_x86_64/Tennis.exe'
PATH_TO_MODEL = 'trained_models/'
    
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
    state_space = len(env_info.vector_observations[0])
    action_space = brain.vector_action_space_size
    num_agents = env_info.vector_observations.shape[0]
                
    agent = Agents(state_space, action_space, num_agents, **kwargs)
    
    total_rewards = []
    rewards_window = deque(maxlen=100)
    
    print("Starting training...")
        
    env_solved = False
    
    for episode in range(1, EPISODES+1):
        
        episodic_rewards = 0
        
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        
        done = False   
        t = 0
                        
        for t in range(T_MAX):
            
            actions = []
            
            for i in range(num_agents):
                actions.append(agent.act(state[i], i))
            
            actions = np.concatenate(actions, axis=0)
                        
            env_info = env.step(actions)[brain_name]
            next_state = env_info.vector_observations   
            reward = env_info.rewards                  
            done = env_info.local_done 
            
            experiences = []
            for i in range(num_agents):
                experiences.append([state[i], actions[i], reward[i], next_state[i], done[i]])
            
            agent.step(experiences)
            
            state = next_state
            
            episodic_rewards += max(reward)
            
            if t % UPDATE_EVERY == 0 and len(agent.buffer[str(0)]) >= BATCH_SIZE:
                agent.train(GAMMA)
            
            if done[0] or done[1]:
                break
            
        rewards_window.append(episodic_rewards)
        total_rewards.append(episodic_rewards)
        
        if episode % 100 == 0:
            print("Episode: {}, Rewards: {}".format(episode, np.mean(rewards_window)))
            agent.save(PATH_TO_MODEL)
    
            if np.mean(rewards_window) >= 0.5:
                print("Environment solved in {} number of episodes...".format(episode))
                env_solved = True
                
        if env_solved:
            break
            
    plot_rewards(total_rewards, episode)
    
    env.close()