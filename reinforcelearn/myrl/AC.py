# 这里我们定义一个AC类，包含策略网络和价值网络

import torch.nn as nn
import torch.nn.functional as F
import torch
import gym
from tqdm import tqdm
import numpy as np
# 定义一个策略网络
class PolicyNet(nn.Module):
    """
    three parameters:state_dim,hidden_dim,action_dim
    state_dim:state的维度
    hidden_dim:隐藏层的维度
    action_dim:动作的维度
    """
    def __init__(self, state_dim, hidden_dim,action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.softmax(self.fc2(x), dim=1) # 返回一个概率分布，其中是每个动作的概率，动作数量为action_dim
        return x

# 定义一个价值网络
class ValueNet(nn.Module):
    """
    three parameters:state_dim,hidden_dim
    state_dim:state的维度
    hidden_dim:隐藏层的维度
    """
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

# 下面开始定义 Actor-Critic 算法
class AC:
    def __init__(self, state_dim, hidden_dim, action_dim, gamma,device,lr=1e-3):
        self.policynet  = PolicyNet(state_dim, hidden_dim, action_dim)
        self.valuenet = ValueNet(state_dim, hidden_dim)
        self.actor_optimizer = torch.optim.Adam(self.policynet.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.policynet.parameters(), lr=lr)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.gamma = gamma
        self.device = device

    def take_action(self, state): # 根据当前状态采样动作
        state = torch.FloatTensor(state).to(self.device)
        probs = self.policynet(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.item()
    
    def update(self, transition_dict): # 根据经验更新策略网络和价值网络
        states = torch.FloatTensor(transition_dict['states'],dtype = torch.float).to(self.device)
        actions = torch.LongTensor(transition_dict['actions'],dtype = torch.float).view(-1,1).to(self.device)
        rewards = torch.FloatTensor(transition_dict['rewards'],dtype = torch.float).to(self.device)
        next_states = torch.FloatTensor(transition_dict['next_states'],dtype = torch.float).to(self.device)
        dones = torch.FloatTensor(transition_dict['dones']).view(-1,1).to(self.device)

        # 计算TD目标
        td_target = rewards+self.gamma*self.valuenet(next_states)-self.valuenet(states)