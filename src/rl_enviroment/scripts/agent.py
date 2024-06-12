#!/usr/bin/env python
import parl
import torch
import numpy as np
from control_pid import PidControl
# from control_agv2_fuzzy import FuzzyPidControl

class GazeboAgent(parl.Agent):
    def __init__(self, algorithm,env):
        super(GazeboAgent,self).__init__(algorithm)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        # self.pidcontrol = PidControl(self.env.high_bound,self.env.robot2_high_bound,self.env.low_bound,self.env.robot2_low_bound)
        # self.fuzzypidcontrol = FuzzyPidControl(self.env.high_bound,self.env.robot2_high_bound,self.env.low_bound,self.env.robot2_low_bound)

        self.alg.sync_target(decay=0)

    def predict(self,obs):

        obs = torch.tensor(obs.reshape(1,-1),dtype=torch.float32).to(self.device)
        action= self.alg.predict(obs)
        action_numpy = action.cpu().detach().numpy().flatten()
        return action_numpy
    
    def sample(self,obs):

        obs = torch.tensor(obs.reshape(1,-1),dtype=torch.float32).to(self.device)
        action,_ = self.alg.sample(obs)
        action_numpy = action.cpu().detach().numpy().flatten()
        return action_numpy

    def learn(self,obs,action,reward,next_obs,terminal):
        terminal = np.expand_dims(terminal,-1)
        reward = np.expand_dims(reward,-1)

        obs = torch.tensor(obs,dtype=torch.float32).to(self.device)
        action = torch.tensor(action,dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward,dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(next_obs,dtype=torch.float32).to(self.device)
        terminal = torch.tensor(terminal,dtype=torch.float32).to(self.device)
        critic_loss, actor_loss,alpha_loss = self.alg.learn(obs,action,reward,next_obs,terminal)
        return critic_loss,actor_loss,alpha_loss
    # def use_pid(self,env,noise):
        # action = self.pidcontrol.pid_get_follower_cmd(self.env._relative_pos_cmd,noise)
        # action = self.fuzzypidcontrol.fuzzly_get_follower_cmd(self.env._relative_pos_cmd,noise)
        # return action

