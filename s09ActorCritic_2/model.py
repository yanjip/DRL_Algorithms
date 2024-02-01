# time: 2023/10/21 21:17
# author: YanJP
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


class Actor(nn.Module):
    '''
    演员Actor网络
    '''
    def __init__(self, action_dim, state_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 300)
        self.fc2 = nn.Linear(300, action_dim)

        self.ln = nn.LayerNorm(300)

    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)
        x = self.ln(F.relu(self.fc1(s)))
        out = F.softmax(self.fc2(x), dim=-1)

        return out


class Critic(nn.Module):
    '''
    评论家Critic网络
    '''
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 300)
        self.fc2 = nn.Linear(300, 1)

        self.ln = nn.LayerNorm(300)

    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)
        x = self.ln(F.relu(self.fc1(s)))
        out = self.fc2(x)

        return out


class Actor_Critic:
    def __init__(self, env):
        self.gamma = 0.99
        self.lr_a = 3e-4
        self.lr_c = 5e-4

        self.env = env
        self.action_dim = self.env.action_space.n             #获取描述行动的数据维度
        self.state_dim = self.env.observation_space.shape[0]  #获取描述环境的数据维度

        self.actor = Actor(self.action_dim, self.state_dim)   #创建演员网络
        self.critic = Critic(self.state_dim)                  #创建评论家网络

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.loss = nn.MSELoss()

    def get_action(self, s):
        a = self.actor(s)
        dist = Categorical(a)
        action = dist.sample()             #可采取的action
        log_prob = dist.log_prob(action)   #每种action的概率

        return action.detach().numpy(), log_prob

    def learn(self, log_prob, s, s_, rew):
        #使用Critic网络估计状态值
        v = self.critic(s)
        v_ = self.critic(s_)

        critic_loss = self.loss(self.gamma * v_ + rew, v)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        td = self.gamma * v_ + rew - v          #计算TD误差
        loss_actor = -log_prob * td.detach()
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()
