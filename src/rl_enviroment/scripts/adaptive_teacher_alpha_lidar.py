from base_teacher_sac_lidar import SAC
import torch
import torch.nn.functional as F
from parl.utils.utils import check_model_method
from copy import deepcopy
class GazeboSAC(SAC):
    def __init__(self,
                 model,
                 student_model,
                 gamma=None,
                 tau=None,
                 alpha=None,
                 actor_lr=None,
                 critic_lr=None,
                 alpha_lr = None,
                 bc_lr = None,
                 action_dim = None,
                 adaptive_alpha=True):
        """ SAC algorithm
            Args:
                model(parl.Model): forward network of actor and critic.
                gamma(float): discounted factor for reward computation
                tau (float): decay coefficient when updating the weights of self.target_model with self.model
                alpha (float): Temperature parameter determines the relative importance of the entropy against the reward
                actor_lr (float): learning rate of the actor model
                critic_lr (float): learning rate of the critic model
        """
        # checks
        check_model_method(model, 'value', self.__class__.__name__)
        check_model_method(model, 'policy', self.__class__.__name__)
        check_model_method(model, 'get_actor_params', self.__class__.__name__)
        check_model_method(model, 'get_critic_params', self.__class__.__name__)
        check_model_method(student_model, 'value', self.__class__.__name__)
        check_model_method(student_model, 'policy', self.__class__.__name__)
        check_model_method(student_model, 'get_actor_params', self.__class__.__name__)
        check_model_method(student_model, 'get_critic_params', self.__class__.__name__)
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(alpha, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)
        assert isinstance(alpha_lr, float)
        assert isinstance(bc_lr, float)
        assert isinstance(action_dim, int)
        assert isinstance(adaptive_alpha, bool)
        self.adaptive_alpha = adaptive_alpha
        self.gamma = gamma
        self.tau = tau
        
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.bc_lr = bc_lr

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.student_model = student_model.to(device)
        self.target_model = deepcopy(self.model)
        self.actor_optimizer = torch.optim.Adam(
            self.model.get_actor_params(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.model.get_critic_params(), lr=critic_lr)
        self.bc_optimizer = torch.optim.Adam(self.student_model.get_actor_params(),lr=bc_lr)
        if self.adaptive_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1,requires_grad=True,device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha],lr=alpha_lr)
        else:
            self.alpha = alpha
    def _alpha_learn(self,obs):
        act, log_pi = self.sample(obs)
        alpha_loss = -(self.log_alpha.exp()*(log_pi+self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        return alpha_loss
    def bc_learn(self,src,tgt,vel,obs):
    
        teacher_act,teacher_log_pi = self.sample(obs)
        student_act,student_log_pi = self.student_sample(src,tgt,vel)
        loss_bc = F.mse_loss(student_act,teacher_act).mean()
        self.bc_optimizer.zero_grad()
        loss_bc.backward()
        self.bc_optimizer.step()
        return loss_bc
        

    def learn(self,src,tgt,vel,obs,action,reward,next_obs,terminal):
        critic_loss = self._critic_learn( obs, action, reward, next_obs,
                                         terminal)
        actor_loss = self._actor_learn(obs)
        
        alpha_loss = self._alpha_learn(obs)
        bc_loss = self.bc_learn(src,tgt,vel,obs)
        # bc_loss = 0.0
        # critic_loss = 0.0
        # actor_loss = 0.0
        # alpha_loss = 0.0
        self.sync_target()
        self.critci_sync_target()
        return critic_loss, actor_loss, alpha_loss,bc_loss
    def critci_sync_target(self):
        for param, student_param in zip(self.model.get_critic_params(),
                                       self.student_model.get_critic_params()):
            student_param.data.copy_(param.data)
    