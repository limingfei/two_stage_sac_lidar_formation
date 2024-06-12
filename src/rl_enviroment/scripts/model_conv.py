#!/usr/bin/env python
import torch
import parl
import torch.nn as nn
from torch.nn import MultiheadAttention
import numpy as np

LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0
class GazeboModel(parl.Model):
    def __init__(self,obs_dim,act_dim):
        super(GazeboModel,self).__init__()
        self.actor_model = Actor(obs_dim,act_dim)
        self.critic_model = Critic(obs_dim,act_dim)

        self.apply(weights_init)
    
    def policy(self,obs):
        return self.actor_model(obs[1])
    
    def value(self,obs,act):
        return self.critic_model(obs[0],act)
    
    def get_actor_params(self):
        return self.actor_model.parameters()
    
    def get_critic_params(self):
        return self.critic_model.parameters()

    
class Actor(parl.Model):
    def __init__(self,feature_dim,act_dim):
        super(Actor,self).__init__()

        self.conv1 = nn.Conv2d(1,32,8,stride=4,padding=1)
        self.maxp1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,4,stride=2,padding=2)
        self.maxp2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(64,64,3,stride=1,padding=0)
        self.maxp3 = nn.MaxPool2d(2,2)
        self.flatten = nn.Flatten()

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        # self.l2 = nn.Linear(128,256)
        # self.l3 = nn.Linear(256,256)

        self.mean_linear1 = nn.Linear(64*4*9,512)
        self.mean_linear2 = nn.Linear(512,512)
        self.mean_linear3 = nn.Linear(512,act_dim)

        self.std_linear1 = nn.Linear(64*4*9,512)
        self.std_linear2 = nn.Linear(512,512)
        self.std_linear3 = nn.Linear(512,act_dim)

        # self.l1.weight.data = norm_col_init(self.l1.weight.data, 0.01)
        # self.l1.bias.data.fill_(0)
        # self.l2.weight.data = norm_col_init(self.l2.weight.data, 0.01)
        # self.l2.bias.data.fill_(0)
        # self.l3.weight.data = norm_col_init(self.l3.weight.data, 0.01)
        # self.l3.bias.data.fill_(0)

        self.mean_linear1.weight.data = norm_col_init(self.mean_linear1.weight.data, 0.01)
        self.mean_linear1.bias.data.fill_(0)
        self.mean_linear2.weight.data = norm_col_init(self.mean_linear2.weight.data, 0.01)
        self.mean_linear2.bias.data.fill_(0)
        self.mean_linear3.weight.data = norm_col_init(self.mean_linear3.weight.data, 0.01)
        self.mean_linear3.bias.data.fill_(0)

        self.std_linear1.weight.data = norm_col_init(self.std_linear1.weight.data, 0.01)
        self.std_linear1.bias.data.fill_(0)
        self.std_linear2.weight.data = norm_col_init(self.std_linear2.weight.data, 0.01)
        self.std_linear2.bias.data.fill_(0)
        self.std_linear3.weight.data = norm_col_init(self.std_linear3.weight.data, 0.01)
        self.std_linear3.bias.data.fill_(0)

        

    def forward(self,obs):
    
        x = torch.relu(self.maxp1(self.conv1(obs)))
        x = torch.relu(self.maxp2(self.conv2(x)))
        x = torch.relu(self.maxp3(self.conv3(x)))
        x = self.flatten(x)

        act_mean = torch.relu(self.mean_linear1(x))
        act_mean = torch.relu(self.mean_linear2(act_mean))
        act_mean = self.mean_linear3(act_mean)

        act_std = torch.relu(self.std_linear1(x))
        act_std = torch.relu(self.std_linear2(act_std))
        act_std = self.std_linear3(act_std)

        act_log_std = torch.clip(act_std,min=LOG_SIG_MIN,max=LOG_SIG_MAX)
        return act_mean,act_log_std
    

class Critic(parl.Model):
    def __init__(self,feature_dim,act_dim):
        super(Critic,self).__init__()

        # Q1 network
        self.l1 = nn.Linear(feature_dim + act_dim,128)

        self.l2 = nn.Linear(128,256)
        self.l3 = nn.Linear(256,256)
        self.l4 = nn.Linear(256,256)
        self.l5 = nn.Linear(256,1)

        self.l1.weight.data = norm_col_init(self.l1.weight.data, 0.01)
        self.l1.bias.data.fill_(0)
        self.l2.weight.data = norm_col_init(self.l2.weight.data, 0.01)
        self.l2.bias.data.fill_(0)
        self.l3.weight.data = norm_col_init(self.l3.weight.data, 0.01)
        self.l3.bias.data.fill_(0)
        self.l4.weight.data = norm_col_init(self.l4.weight.data, 0.01)
        self.l4.bias.data.fill_(0)
        self.l5.weight.data = norm_col_init(self.l5.weight.data, 0.01)
        self.l5.bias.data.fill_(0)

        # Q2 network

        self.l6 = nn.Linear(feature_dim + act_dim,128)

        self.l7 = nn.Linear(128,256)
        self.l8 = nn.Linear(256,256)
        self.l9 = nn.Linear(256,256)
        self.l10 = nn.Linear(256,1)

        self.l6.weight.data = norm_col_init(self.l6.weight.data, 0.01)
        self.l6.bias.data.fill_(0)
        self.l7.weight.data = norm_col_init(self.l7.weight.data, 0.01)
        self.l7.bias.data.fill_(0)
        self.l8.weight.data = norm_col_init(self.l8.weight.data, 0.01)
        self.l8.bias.data.fill_(0)
        self.l9.weight.data = norm_col_init(self.l9.weight.data, 0.01)
        self.l9.bias.data.fill_(0)
        self.l10.weight.data = norm_col_init(self.l10.weight.data, 0.01)
        self.l10.bias.data.fill_(0)
        


    
    def forward(self,obs,act):
        
        
        x = torch.concat([obs,act],1)

        # Q1

        q1 = torch.relu(self.l1(x))
        q1 = torch.relu(self.l2(q1))
        q1 = torch.relu(self.l3(q1))
        q1 = torch.relu(self.l4(q1))
        q1 = self.l5(q1)

        # Q2

        q2 = torch.relu(self.l6(x))
        q2 = torch.relu(self.l7(q2))
        q2 = torch.relu(self.l8(q2))
        q2 = torch.relu(self.l9(q2))
        q2 = self.l10(q2)

        return q1,q2

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x
# model = GazeboModel(9,3)
# obs2 = torch.randn(32,1,360,640)
# obs1 = torch.randn(9,)
# out = model.policy([obs1,obs2])
# print(out[0].shape,out[1].shape)
