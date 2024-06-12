#!/usr/bin/env python
import parl
import torch
import numpy as np
import os
class GazeboAgent(parl.Agent):
    def __init__(self, algorithm):
        super(GazeboAgent,self).__init__(algorithm)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.alg.sync_target(decay=0)
        self.pid_sac_flag = 0

    def predict(self,src,tgt,vel,obs,student_mode):
        # obs0 = torch.tensor(obs[0].reshape(1,-1),dtype=torch.float32).to(self.device)
        # obs1 = torch.tensor(obs[1],dtype=torch.float32).unsqueeze(0).to(self.device)
        src = torch.tensor(src,dtype=torch.float32).unsqueeze(0).to(self.device)
        tgt = torch.tensor(tgt,dtype=torch.float32).unsqueeze(0).to(self.device)
        vel = torch.tensor(vel.reshape(1,-1),dtype=torch.float32).to(self.device)
        obs = torch.tensor(obs.reshape(1,-1),dtype=torch.float32).to(self.device)
        # obs = [obs0,obs1]
        action= self.alg.predict(src,tgt,vel,obs,student_mode)
        action_numpy = action.cpu().detach().numpy().flatten()
        return action_numpy
    
    def sample(self,obs):

        # src = torch.tensor(src,dtype=torch.float32).unsqueeze(0).to(self.device)
        # tgt = torch.tensor(tgt,dtype=torch.float32).unsqueeze(0).to(self.device)
        # vel = torch.tensor(vel.reshape(1,-1),dtype=torch.float32).to(self.device)
        obs = torch.tensor(obs.reshape(1,-1),dtype=torch.float32).to(self.device)
        action,_ = self.alg.sample(obs)
        action_numpy = action.cpu().detach().numpy().flatten()
        return action_numpy
    def student_sample(self,src,tgt,vel):

        src = torch.tensor(src,dtype=torch.float32).unsqueeze(0).to(self.device)
        tgt = torch.tensor(tgt,dtype=torch.float32).unsqueeze(0).to(self.device)
        vel = torch.tensor(vel.reshape(1,-1),dtype=torch.float32).to(self.device)
        # obs = torch.tensor(obs.reshape(1,-1),dtype=torch.float32).to(self.device)
        action,_ = self.alg.student_sample(src,tgt,vel)
        action_numpy = action.cpu().detach().numpy().flatten()
        return action_numpy

    def learn(self,src,tgt,vel,obs,action,reward,next_obs,terminal):
        # terminal = np.expand_dims(terminal,-1)
        # reward = np.expand_dims(reward,-1)


        src = torch.as_tensor(src,dtype=torch.float32).to(self.device)
        tgt = torch.as_tensor(tgt,dtype=torch.float32).to(self.device)
        vel = torch.as_tensor(vel,dtype=torch.float32).to(self.device)
        obs = torch.as_tensor(obs,dtype=torch.float32).to(self.device)
        
        action = torch.as_tensor(action,dtype=torch.float32).to(self.device)
        # expert_act = torch.as_tensor(expert_act,dtype=torch.float32).to(self.device)
        reward = torch.as_tensor(reward,dtype=torch.float32).to(self.device)
        # next_src = torch.as_tensor(next_src,dtype=torch.float32).to(self.device)
        # next_vel = torch.as_tensor(next_vel,dtype=torch.float32).to(self.device)
        next_obs = torch.as_tensor(next_obs,dtype=torch.float32).to(self.device)
        terminal = torch.as_tensor(terminal,dtype=torch.float32).to(self.device)

        critic_loss, actor_loss,alpha_loss,bc_loss = self.alg.learn(src,tgt,vel,obs,action,reward,next_obs,terminal)
        return critic_loss, actor_loss,alpha_loss,bc_loss
    def save(self, model_save_path, opt_save_path=None,model=None,opt=None):
        """Save parameters.

        Args:
            save_path(str): where to save the parameters.
            model(parl.Model): model that describes the neural network structure. If None, will use self.alg.model.

        Raises:
            ValueError: if model is None and self.alg.model does not exist.

        Example:

        .. code-block:: python

            agent = AtariAgent()
            agent.save('./model.ckpt')

        """
        if model is None:
            model = self.alg.model
            student_model = self.alg.student_model
        if opt is None:
            opt_act = self.alg.actor_optimizer
            opt_cri = self.alg.critic_optimizer
        sep = os.sep
        model_dirname = sep.join(model_save_path.split(sep)[:-1])
        if model_dirname != '' and not os.path.exists(model_dirname):
            os.makedirs(model_dirname)
        if opt_save_path is not None:
            actor_opt_dirname = sep.join(opt_save_path[0].split(sep)[:-1])
            critic_opt_dirname = sep.join(opt_save_path[1].split(sep)[:-1])
            if actor_opt_dirname != '' and not os.path.exists(actor_opt_dirname):
                os.makedirs(actor_opt_dirname)
            if critic_opt_dirname != '' and not os.path.exists(critic_opt_dirname):
                os.makedirs(critic_opt_dirname)
            torch.save(opt_act.state_dict(), opt_save_path[0])
            torch.save(opt_cri.state_dict(), opt_save_path[1])
        torch.save(model.state_dict(), model_save_path)
        torch.save(student_model.state_dict(), model_save_path.replace('best_model.pth','best_student_model.pth'))
    def restore(self, model_save_path, opt_save_path=None,model=None,opt=None,map_location=None):
        """Restore previously saved parameters.
        This method requires a model that describes the network structure.
        The save_path argument is typically a value previously passed to ``save()``.

        Args:
            save_path(str): path where parameters were previously saved.
            model(parl.Model): model that describes the neural network structure. If None, will use self.alg.model.
            map_location: a function, torch.device, string or a dict specifying how to remap storage locations

        Raises:
            ValueError: if model is None and self.alg does not exist.

        Example:

        .. code-block:: python

            agent = AtariAgent()
            agent.save('./model.ckpt')
            agent.restore('./model.ckpt')
            
            agent.restore('./model.ckpt', map_location=torch.device('cpu')) # load gpu-trained model in cpu machine
        """

        if model is None:
            student_model = self.alg.student_model
        if opt is None:
            actor_opt = self.alg.actor_optimizer
            critic_opt = self.alg.critic_optimizer
        if opt_save_path is not None:        
            actor_checkpoint = torch.load(opt_save_path[0], map_location=map_location)
            critic_checkpoint = torch.load(opt_save_path[1], map_location=map_location)
            actor_opt.load_state_dict(actor_checkpoint)
            critic_opt.load_state_dict(critic_checkpoint)
        model_checkpoint = torch.load(model_save_path, map_location=map_location)
        student_model.load_state_dict(model_checkpoint)
