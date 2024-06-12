from base_sac_lidar import SAC
import torch
import torch.nn.functional as F
from parl.utils.utils import check_model_method
from copy import deepcopy
class GazeboSAC(SAC):
    def __init__(self,
                 model,
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
        self.pre_model = deepcopy(self.model)
        self.actor_weight = self.pre_model.get_actor_params()
        self.target_model = deepcopy(self.model)
        self.actor_optimizer = torch.optim.Adam(
            self.model.get_actor_params(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.model.get_critic_params(), lr=critic_lr)
        self.bc_optimizer = torch.optim.Adam(self.model.get_actor_params(),lr=bc_lr)
        if self.adaptive_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1,requires_grad=True,device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha],lr=alpha_lr)
        else:
            self.alpha = alpha
    def _alpha_learn(self,src,tgt,vel):
        act, log_pi = self.sample(src,tgt,vel)
        alpha_loss = -(self.log_alpha.exp()*(log_pi+self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        return alpha_loss
    def bc_learn(self,src,tgt,vel,expert_act):
    
        act,log_pi = self.sample(src,tgt,vel)
        loss_bc = F.mse_loss(act,expert_act).mean()
        self.bc_optimizer.zero_grad()
        loss_bc.backward()
        self.bc_optimizer.step()
        return loss_bc
        

    def learn(self,src,tgt,vel,obs,action,expert_act,reward,next_src,next_vel,next_obs,terminal):
        # critic_loss = self._critic_learn( src,tgt,vel, action, reward, next_src,next_vel,
                                        #  terminal)
        critic_loss,critic_reg_loss = self._critic_learn(src,tgt,vel,obs, action, reward, next_src,next_vel,next_obs, terminal)
        actor_loss,actor_reg_loss = self._actor_learn(src,tgt,vel,obs)
        
        # alpha_loss = self._alpha_learn(src,tgt,vel)
        # bc_loss = self.bc_learn(src,tgt,vel,expert_act)
        bc_loss = 0.0
        # critic_loss = 0.0
        # actor_loss = 0.0
        # critic_reg_loss = 0.0
        alpha_loss = 0.0
        self.sync_target()
        return critic_loss, actor_loss, alpha_loss,bc_loss,actor_reg_loss,critic_reg_loss
    
    