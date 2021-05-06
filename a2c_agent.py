
import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gridenv import GridWorld

hidden_layer = 256
num_episodes = 1500
num_step = 300
gamma = 0.90
lr = 3e-4

class Grid(torch.nn.Module):      #create Neural Network
    def __init__(self,i,j):
        super(Grid, self). __init__()
        self.num_states = i*j
        self.actions = 4
        self.criticl1 = nn.Linear(self.num_states, hidden_layer)#create input critic layer
        nn.Dropout(p=0.2)#Dropingout the unused connections
        self.criticl2 = nn.Linear(hidden_layer, 1)#critic output layer
        
        self.alayer1 = nn.Linear(self.num_states, hidden_layer)#creating actor input layer
        nn.Dropout(p=0.2)
        self.alayer2 = nn.Linear(hidden_layer, self.actions)#creating output actor layer
        
    def forward(self, state):#forward function to find the action values and policy distribution
        state = state.float()
        value = fn.relu(self.criticl1(state))
        value = self.criticl2(value)
        
        policy_dis = fn.relu(self.alayer1(state))
        actionval = self.alayer2(policy_dis)
        policy_distribution = fn.softmax(actionval, dim = -1)
        
        return value, actionval, policy_distribution

if __name__ == '__main__':    #main function start
    env = GridWorld(5,5)       # performing with (5*5 grid) You can try untill 50*50
    agent = Grid(5, 5)
    optim = torch.optim.Adam(agent.parameters(), lr =lr) # optimiser used is Adam optimiser
    entropy_term = 0   # to reduce the degree of uncertanity 
    actionpossibilities = ['U', 'D', 'L' , 'R'] # total 4 actions possible
    all_rewards = []
    all_lengths = []
    total_loss = []
    average_lengths = []
    for episode in range(num_episodes):
        print("episode number", episode)
        obs = env.reset()
        rewards= torch.zeros(num_step,1)
        values = torch.zeros(num_step,1)
        log_probs = torch.zeros(num_step,1)
        for steps in range(num_step):
            obs = obs.long()
            obs = fn.one_hot(obs, num_classes = 25)
            value, actionval, policy_distribution = agent.forward(obs)
            pdist = policy_distribution.detach().numpy()
            value = value.reshape(-1)
            action = np.random.choice(4, p=np.squeeze(pdist))
            log_prob = torch.log(policy_distribution.squeeze(0)[action])
            entropy = -np.sum(np.mean(pdist) * np.log(pdist))
            action = actionpossibilities[action]
            new_state, reward, done, _ = env.step(action)
            reward = torch.tensor(reward)
            rewards[steps].copy_(reward)
            values[steps].copy_(value)
            log_probs[steps].copy_(log_prob)
            entropy_term += entropy
            obs = torch.tensor(new_state)
            if done or steps == num_step-1:
                obs = obs.long()
                obs = fn.one_hot(obs, num_classes = 25)
                Qval, _,_ = agent.forward(obs)
                Qval = Qval.detach()
                all_rewards.append(np.sum(rewards.numpy()))
                all_lengths.append(steps)
                print("Episode reward", all_rewards[-1])
                print("number of steps",all_lengths[-1])
                break
        
        # To find the Q value
        Qvals = np.zeros_like(values.detach().numpy())
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + gamma * Qval
            Qvals[t] = Qval
  
        #To update advantage function
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)

        
        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term
        print("total loss", ac_loss)
        total_loss.append(ac_loss)
        optim.zero_grad()
        ac_loss.backward()
        optim.step()

          
    smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    plt.plot(all_rewards)
    plt.plot(smoothed_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
    
    smoothed_loss = pd.Series.rolling(pd.Series(total_loss), 10).mean()
    smoothed_loss = [elem for elem in smoothed_loss]
    plt.plot(total_loss)
    plt.plot(smoothed_loss)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('loss')
    plt.show()


    plt.plot(all_lengths)
    plt.plot(average_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
        
