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
        return self.actor_model(obs)
    
    def value(self,obs,act):
        return self.critic_model(obs,act)
    
    def get_actor_params(self):
        return self.actor_model.parameters()
    
    def get_critic_params(self):
        return self.critic_model.parameters()
    def init_lstm_state(self,batch_size):
        self.actor_model.init_lstm_state(batch_size)
        self.critic_model.init_lstm_state(batch_size)
    def init_lstm_state_train(self):
        self.actor_model.init_lstm_state_train()
        self.critic_model.init_lstm_state_train()

    
class Actor(parl.Model):
    def __init__(self,feature_dim,act_dim):
        super(Actor,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hidden_size = 128
        self.first = False
        self.act_dim = act_dim

        self.fc1 = nn.Sequential(
                            nn.Linear(feature_dim,128),
                            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
                            nn.Linear(self.hidden_size,256),
                            nn.ReLU())
        self.mean_linear = nn.Linear(256,self.act_dim)
        self.std_linear = nn.Linear(256,self.act_dim)

        self.fc1[0].weight.data = norm_col_init(self.fc1[0].weight.data, 0.01)
        self.fc1[0].bias.data.fill_(0)
        self.fc2[0].weight.data = norm_col_init(self.fc2[0].weight.data, 0.01)
        self.fc2[0].bias.data.fill_(0)
        self.mean_linear.weight.data = norm_col_init(self.mean_linear.weight.data, 0.01)
        self.mean_linear.bias.data.fill_(0)
        self.std_linear.weight.data = norm_col_init(self.std_linear.weight.data, 0.01)
        self.std_linear.bias.data.fill_(0)        
    def init_lstm_state(self,batch_size):
            self.h = torch.zeros((1,batch_size,self.hidden_size),dtype=torch.float32).to(self.device)
            self.c = torch.zeros((1,batch_size,self.hidden_size),dtype=torch.float32).to(self.device)
            self.first = True
    def init_lstm_state_train(self):
        self.h = torch.zeros((1,self.hidden_size),dtype=torch.float32).to(self.device)
        self.c = torch.zeros((1,self.hidden_size),dtype=torch.float32).to(self.device)
        self.first = False
    def forward(self,obs):
        
                
        batch_size = obs.shape[0]
        x = self.fc1(obs)
        x = self.fc2(x)
        mean = self.mean_linear(x)
        log_std = self.std_linear(x)
        act_log_std = torch.clip(log_std,min=LOG_SIG_MIN,max=LOG_SIG_MAX)
        
        return mean,act_log_std
    

class Critic(parl.Model):
    def __init__(self,feature_dim,act_dim):
        super(Critic,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hidden_size = 128
        self.first = False
        self.fc1 = nn.Sequential(
                   nn.Linear(feature_dim+act_dim,128),
                   nn.ReLU()                   
        )
        
        self.fc2 = nn.Sequential(
                            nn.Linear(128,256),
                            nn.ReLU())

        # Q1 network
        self.q1 = nn.Linear(256,1)
        # Q1 network
        self.q2 = nn.Linear(256,1)

        self.fc1[0].weight.data = norm_col_init(self.fc1[0].weight.data, 0.01)
        self.fc1[0].bias.data.fill_(0)
        
        self.fc2[0].weight.data = norm_col_init(self.fc2[0].weight.data, 0.01)
        self.fc2[0].bias.data.fill_(0)
        self.q1.weight.data = norm_col_init(self.q1.weight.data, 0.01)
        self.q1.bias.data.fill_(0)
        self.q2.weight.data = norm_col_init(self.q2.weight.data, 0.01)
        self.q2.bias.data.fill_(0)

    def init_lstm_state(self,batch_size):
            self.h = torch.zeros((1,batch_size,self.hidden_size),dtype=torch.float32).to(self.device)
            self.c = torch.zeros((1,batch_size,self.hidden_size),dtype=torch.float32).to(self.device)
            self.first = True
    def init_lstm_state_train(self):
        self.h = torch.zeros((1,1,self.hidden_size),dtype=torch.float32).to(self.device)
        self.c = torch.zeros((1,1,self.hidden_size),dtype=torch.float32).to(self.device)  
        self.first = False
  
    def forward(self,obs,act):
              
        obs_act = torch.concat([obs,act],-1)
        x = self.fc1(obs_act)     
      
        x = self.fc2(x)
        q1 = self.q1(x)
        q2 = self.q2(x)
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

