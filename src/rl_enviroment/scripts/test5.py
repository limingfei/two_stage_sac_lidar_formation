import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from rich.progress import Progress
from model_lidar import GazeboModel
from adaptive_alpha_lidar import GazeboSAC
from agent_lidar import GazeboAgent
from tensorboardX import SummaryWriter

WARMUP_STEPS = 1e3
# WARMUP_STEPS = 10

EVAL_EPISODES = 3
MEMORY_SIZE = int(2e4)
# MEMORY_SIZE = int(1e2)
BATCH_SIZE = 64
# BATCH_SIZE = 8
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.01
# ACTOR_LR = 3e-4
ACTOR_LR = 1e-4
BC_LR = 1e-3
# CRITIC_LR = 3e-4
CRITIC_LR = 1e-4
ALPHA_LR = 3e-3
NUM_STEPS = 50
MAX_EPOCH = 10

class CustomDataset(Dataset):
    def __init__(self):
        root_path = 'collect_dataset'
        dir = os.listdir(root_path)
        self.src = np.empty((0,768,3))
        self.tgt = np.empty((0,768,3))
        self.vel = np.empty((0,3))
        self.act = np.empty((0,3))
        self.expert_act = np.empty((0,3))
        self.reward = np.empty((0,1))
        self.next_src = np.empty((0,768,3))
        self.next_vel = np.empty((0,3))
        self.done = np.empty((0,1))
        with Progress() as progress:
            task = progress.add_task("[cyan]loading dataset...", total=len(dir))

            for k in dir:
                file = os.path.join(root_path,k)
                data = np.load('collect_dataset/0.npz',allow_pickle=True)
                data = data['data'].tolist()
                data = dict(data)
                self.src = np.concatenate((self.src,data['src']),axis=0)
                self.tgt = np.concatenate((self.tgt,data['tgt']),axis=0)
                self.vel = np.concatenate((self.vel,data['vel']),axis=0)
                self.act = np.concatenate((self.act,data['act']),axis=0)
                self.expert_act = np.concatenate((self.expert_act,data['expert_act']),axis=0)
                self.reward = np.concatenate((self.reward,data['reward']),axis=0)
                self.next_src = np.concatenate((self.next_src,data['next_src']),axis=0)
                self.next_vel = np.concatenate((self.next_vel,data['next_vel']),axis=0)
                self.done = np.concatenate((self.done,data['done']),axis=0)
                progress.update(task,advance=1)
        assert self.src.shape[0] == self.tgt.shape[0] == self.vel.shape[0] == self.act.shape[0] == self.expert_act.shape[0] == self.reward.shape[0] == self.next_src.shape[0] == self.next_vel.shape[0] == self.done.shape[0] 
        array_list = [self.src,self.tgt,self.vel,self.act,self.expert_act,self.reward,self.next_src,self.next_vel,self.done]
        has_nan = any(np.isnan(array).any() for array in array_list)
        has_inf = any(np.isinf(array).any() for array in array_list)
        # has_inf = np.isinf([self.src,self.tgt,self.vel,self.act,self.expert_act,self.reward,self.next_src,self.next_vel,self.done])
        if has_nan:
            print("数组中存在 NaN 值。")
        else:
            print("数组中不存在 NaN 值。")

        if has_inf:
            print("数组中存在 Inf 值。")
        else:
            print("数组中不存在 Inf 值。")
            
    def __len__(self):
        return self.src.shape[0]
 
    def __getitem__(self, idx):
        return self.src[idx],self.tgt[idx],self.vel[idx],self.act[idx],self.expert_act[idx],self.reward[idx],self.next_src[idx],self.next_vel[idx],self.done[idx]
    


writer = SummaryWriter('./bc_logs')

 
dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

model = GazeboModel()
alg = GazeboSAC(model=model,gamma=GAMMA,tau=TAU,alpha=ALPHA,actor_lr=ACTOR_LR,critic_lr=CRITIC_LR,alpha_lr=ALPHA_LR,bc_lr=BC_LR,action_dim=3,adaptive_alpha=False)
agent = GazeboAgent(algorithm=alg)
step = 0
# 使用dataloader进行迭代
with Progress() as progress:
    task = progress.add_task("[cyan]training...", total=int(MAX_EPOCH*len(dataset)/32))
    for k in range(MAX_EPOCH):
        for batch_src,batch_tgt,batch_vel,batch_act,batch_expert_act,batch_reward,batch_next_src,batch_next_vel,batch_done in dataloader:
            step += 1
            critic_loss, actor_loss, alpha_loss,bc_loss = agent.learn(batch_src,batch_tgt,batch_vel,batch_act,batch_expert_act,batch_reward,batch_next_src,batch_next_vel,batch_done)
            if torch.isnan(critic_loss.cpu()).any():
                print('critic_loss is nan')
            if torch.isnan(actor_loss.cpu()).any():
                print('actor_loss is nan')
            # if torch.isnan(alpha_loss.cpu()).any():
            #     print('alpha_loss is nan')
            if torch.isnan(bc_loss.cpu()).any():
                print('bc_loss is nan')
            if torch.isinf(critic_loss.cpu()).any():
                print('critic_loss is inf')
            if torch.isinf(actor_loss.cpu()).any():
                print('actor_loss is inf')
            # if torch.isinf(alpha_loss.cpu()).any():
            #     print('alpha_loss is inf')
            if torch.isinf(bc_loss.cpu()).any():
                print('bc_loss is inf')
            
            if torch.isnan(critic_loss.cpu()).any() or torch.isnan(actor_loss.cpu()).any() or torch.isnan(bc_loss.cpu()).any() or torch.isinf(critic_loss.cpu()).any() or torch.isinf(actor_loss.cpu()).any() or torch.isinf(bc_loss.cpu()).any():
                break
            else:
                writer.add_scalar(tag='critic_loss',scalar_value=critic_loss.cpu().item(),global_step=step)
                writer.add_scalar(tag='actor_loss',scalar_value=actor_loss.cpu().item(),global_step=step)
                # writer.add_scalar(tag='alpha_loss',scalar_value=alpha_loss.cpu().item(),global_step=step)
                writer.add_scalar(tag='bc_loss',scalar_value=bc_loss.cpu().item(),global_step=step)
            if (step % 5000) == 0:
                agent.save('./bc_logs/step{}_model.pth'.format(step))
            progress.update(task,advance=1)

agent.save('./bc_logs/final_model.pth')
        # 在这里处理批数据

# a = {'src':np.array([1,2,3]),'act':np.array([4,5,6])}
# c = {'src':np.array([10,20,30]),'act':np.array([40,50,60])}

# src = np.concatenate((a['src'],c['src']),axis=-1)
# print(src)