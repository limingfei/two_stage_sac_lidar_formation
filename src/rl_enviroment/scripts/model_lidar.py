#!/usr/bin/env python
import torch
import parl
import torch.nn as nn
from torch.nn import MultiheadAttention
import numpy as np
from config import *

LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0
class GazeboModel(parl.Model):
    def __init__(self):
        super(GazeboModel,self).__init__()
        self.actor_model = Actor()
        if HIHGH_CRITIC:
            self.critic_model = HighCritic()
        else:
            self.critic_model = Critic()


        self.apply(weights_init)
    
    def policy(self,src,tgt,vel):
        return self.actor_model(src,tgt,vel)
    
    def value(self,*args):
        if HIHGH_CRITIC:
            assert len(args) == 2
            obs,act = args
            return self.critic_model(obs,act)
        else:
            assert len(args) == 4
            src,tgt,vel,act = args
            return self.critic_model(src,tgt,vel,act)
    
    def get_actor_params(self):
        return self.actor_model.parameters()
    
    def get_critic_params(self):
        return self.critic_model.parameters()

class TeacherModel(parl.Model):
    def __init__(self):
        super(TeacherModel,self).__init__()
        self.actor_model = TeacherActor()
        self.critic_model = HighCritic()

        self.apply(weights_init)
    
    def policy(self,obs):
        return self.actor_model(obs)
    
    def value(self,obs,act):
        return self.critic_model(obs,act)
    
    def get_actor_params(self):
        return self.actor_model.parameters()
    
    def get_critic_params(self):
        return self.critic_model.parameters()


class Actor(parl.Model):
    def __init__(self):
        super(Actor,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed = StateEmbed()
        self.emb_t_r = nn.Sequential(
                        nn.Linear(STATE_DIM,HEAD_DIM*2),
                        nn.ReLU(),
                        nn.Linear(HEAD_DIM*2,HEAD_DIM),
                        nn.ReLU())
        
        self.action_mean = nn.Linear(HEAD_DIM+ACT_DIM,ACT_DIM)
        self.action_std = nn.Linear(HEAD_DIM+ACT_DIM,ACT_DIM)

        self.emb_t_r[0].weight.data = norm_col_init(self.emb_t_r[0].weight.data, 0.01)
        self.emb_t_r[0].bias.data.fill_(0)
        self.emb_t_r[2].weight.data = norm_col_init(self.emb_t_r[2].weight.data, 0.01)
        self.emb_t_r[2].bias.data.fill_(0)
        self.action_mean.weight.data = norm_col_init(self.action_mean.weight.data, 0.01)
        self.action_mean.bias.data.fill_(0)
        self.action_std.weight.data = norm_col_init(self.action_std.weight.data, 0.01)
        self.action_std.bias.data.fill_(0)

        
        
    def forward(self,src,tgt,vel):
                      
        state,em_tgt = self.embed(src,tgt)
        em_state = self.emb_t_r(state) 
        em_state = torch.cat([em_state,vel],dim=1)     

        mean = self.action_mean(em_state)
        log_std = self.action_std(em_state)
        act_log_std = torch.clip(log_std,min=LOG_SIG_MIN,max=LOG_SIG_MAX)
        
        return mean,act_log_std



class TeacherActor(parl.Model):
    def __init__(self):
        super(TeacherActor,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed = nn.Sequential(
                        nn.Linear(HIGH_DIM,128),
                        nn.ReLU(),
                        nn.Linear(128,256),
                        nn.ReLU(),
                        nn.Linear(256,128),
                        nn.ReLU())
        self.action_mean = nn.Linear(128,ACT_DIM)
        self.action_std = nn.Linear(128,ACT_DIM)
        self.embed[0].weight.data = norm_col_init(self.embed[0].weight.data, 0.01)
        self.embed[0].bias.data.fill_(0)
        self.embed[2].weight.data = norm_col_init(self.embed[2].weight.data, 0.01)
        self.embed[2].bias.data.fill_(0)
        self.embed[4].weight.data = norm_col_init(self.embed[4].weight.data, 0.01)
        self.embed[4].bias.data.fill_(0)

        self.action_mean.weight.data = norm_col_init(self.action_mean.weight.data, 0.01)
        self.action_mean.bias.data.fill_(0)
        self.action_std.weight.data = norm_col_init(self.action_std.weight.data, 0.01)
        self.action_std.bias.data.fill_(0)
    def forward(self,obs):
        state = self.embed(obs) 
        mean = self.action_mean(state)
        log_std = self.action_std(state)
        act_log_std = torch.clip(log_std,min=LOG_SIG_MIN,max=LOG_SIG_MAX)
        return mean,act_log_std

class HighCritic(parl.Model):
    def __init__(self):
        super(HighCritic,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embed = nn.Sequential(
                        nn.Linear(HIGH_DIM+ACT_DIM,128),
                        nn.ReLU(),
                        nn.Linear(128,256),
                        nn.ReLU(),
                        nn.Linear(256,128),
                        nn.ReLU())   
        self.q1 = nn.Linear(128,1)
        self.q2 = nn.Linear(128,1)
        self.embed[0].weight.data = norm_col_init(self.embed[0].weight.data, 0.01)
        self.embed[0].bias.data.fill_(0)
        self.embed[2].weight.data = norm_col_init(self.embed[2].weight.data, 0.01)
        self.embed[2].bias.data.fill_(0)
        self.embed[4].weight.data = norm_col_init(self.embed[4].weight.data, 0.01)
        self.embed[4].bias.data.fill_(0)
        self.q1.weight.data = norm_col_init(self.q1.weight.data, 0.01)
        self.q1.bias.data.fill_(0)
        self.q2.weight.data = norm_col_init(self.q2.weight.data, 0.01)
        self.q2.bias.data.fill_(0)
    def forward(self,obs,act):
        obs_act = torch.cat((obs,act),dim=-1)
        state = self.embed(obs_act)
        q1 = self.q1(state)
        q2 = self.q2(state)
        return q1,q2

class Critic(parl.Model):
    def __init__(self):
        super(Critic,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embed = StateEmbed()
        self.emb_v = nn.Sequential(
                        nn.Linear(STATE_DIM,HEAD_DIM*2),
                        nn.ReLU(),
                        nn.Linear(HEAD_DIM*2,HEAD_DIM),
                        nn.ReLU())
        
        self.q1 = nn.Linear(HEAD_DIM+ACT_DIM+ACT_DIM,1)
        self.q2 = nn.Linear(HEAD_DIM+ACT_DIM+ACT_DIM,1)

        self.emb_v[0].weight.data = norm_col_init(self.emb_v[0].weight.data, 0.01)
        self.emb_v[0].bias.data.fill_(0)
        self.emb_v[2].weight.data = norm_col_init(self.emb_v[2].weight.data, 0.01)
        self.emb_v[2].bias.data.fill_(0)
        self.q1.weight.data = norm_col_init(self.q1.weight.data, 0.01)
        self.q1.bias.data.fill_(0)
        self.q2.weight.data = norm_col_init(self.q2.weight.data, 0.01)
        self.q2.bias.data.fill_(0)
    
    def forward(self,src,tgt,vel,act):
       
        
        embed,_ = self.embed(src,tgt)   
        embed_v = self.emb_v(embed)
        state = torch.cat([embed_v,vel,act],dim=1)
        q1 = self.q1(state)
        q2 = self.q2(state)
        return q1,q2

class StateEmbed(parl.Model):
    def __init__(self):
        super(StateEmbed,self).__init__()
        self.embed = nn.Sequential(
                        nn.Conv1d(IN_CHANNELS,64,1),
                        nn.ReLU(),
                        nn.Conv1d(64,128,1),
                        nn.ReLU(),
                        nn.Conv1d(128,1024,1))        
    def forward(self,src,tgt):
        B,N,D = src.shape
        emb_src = self.embed_state(src.transpose(2,1))
        emb_tgt = self.embed_state(tgt.transpose(2,1))
        state = torch.cat((emb_src,emb_tgt),dim=-1)
        sate = state.view(B,-1)
        return state,emb_tgt
    def embed_state(self,x):
        B,D,N = x.shape
        x = self.embed(x)
        x_pooled = torch.max(x,2,keepdim=True)[0]
        return x_pooled.view(B,-1) 

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







# model = GazeboModel()

# src = torch.randn((1024,3))
# tgt = torch.randn((1024,3))
# act = torch.randn((3))

# # mean,std = model.policy(src,tgt)
# # q1,q2 = model.value(src,tgt,act)
# # print(mean.shape,std.shape)
# # print(q1.shape,q2.shape)

