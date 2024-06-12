import collections
import numpy as np
import random
class EpisodeMemory(object):
    def __init__(self,episode_size,num_step):
        self.buffer = collections.deque(maxlen=episode_size)
        self.num_step = num_step
    
    def put(self,episode):
        self.buffer.append(episode)
    
    def sample(self,batch_size):
        mini_batch = random.sample(self.buffer,batch_size)
        obs_batch,action_batch,reward_batch,next_obs_batch,done_batch = [],[],[],[],[]
        for experience in mini_batch:
            self.num_step = min(self.num_step,len(experience))
        for experience in mini_batch:
            idx = np.random.randint(0,len(experience)-self.num_step + 1)
            s,a,r,s_p,done = [],[],[],[],[]
            for i in range(idx,idx+self.num_step):
                e1,e2,e3,e4,e5 = experience[i][0]
                s.append(e1),a.append(e2),r.append(e3),s_p.append(e4),done.append(e5)
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)
        obs_batch = np.array(obs_batch).astype('float32')
        action_batch = np.array(action_batch).astype('float32')
        reward_batch = np.array(reward_batch).astype('float32')
        next_obs_batch = np.array(next_obs_batch).astype('float32')
        done_batch = np.array(done_batch).astype('float32')
        return obs_batch,action_batch,reward_batch,next_obs_batch,done_batch
    def __len__(self):
        return len(self.buffer)

class ReplayMemory(object):
    def __init__(self,episode_size,num_step):
        self.e_rpm = EpisodeMemory(episode_size,num_step)
        self.buff = []

    def append(self,exp,done):
        self.buff.append([exp])
        if done:
            self.e_rpm.put(self.buff)
            self.buff = []
    def sample(self,batch_size):
        return self.e_rpm.sample(batch_size)
    def __len__(self):
        return len(self.e_rpm)

