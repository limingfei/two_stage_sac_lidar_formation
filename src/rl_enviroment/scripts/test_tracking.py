#!/usr/bin/env python
import numpy as np
from agent import GazeboAgent
from model import GazeboModel
import argparse
from parl.utils import logger,ReplayMemory
from parl.env import ActionMappingWrapper
from parl.algorithms import SAC
from trancking_agv import DoubleEscapeEnv
import datetime
import os

from tensorboardX import SummaryWriter

WARMUP_STEPS = 2e3
EVAL_EPISODES = 3
MEMORY_SIZE = int(1e4)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4


import warnings
warnings.simplefilter("ignore", ResourceWarning)


def run_eval_episodes(agent,env,eval_episodes):
    avg_reward = 0.0
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False

        while not done:
            action = agent.predict(obs)
            obs,reward,done,info = env.step([action])
            # logger.info('obs:{:.3f} done:{}, Reward:{:.3f}'.format(obs[0],done,avg_reward))

            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward,obs,done


def main():
    env = DoubleEscapeEnv()
    obs_dim = int(9)
    act_dim = int(3)
    model = GazeboModel(obs_dim,act_dim)
    algorithm = SAC(model,gamma=GAMMA,tau=TAU,alpha=ALPHA,actor_lr=ACTOR_LR,critic_lr=CRITIC_LR)
    agent = GazeboAgent(algorithm)
    agent.restore('/home/lmf/Documents/catkin_test/logs/sac_2023-10-18 14_44_49.618323/best_model.pth')
    rpm = ReplayMemory(MEMORY_SIZE,obs_dim=obs_dim,act_dim=act_dim)
    total_steps = 0
    test_flag = 0
    root_log_path = os.path.join(args.log_path,'sac_{}'.format(datetime.datetime.now().__str__().replace(':','_')))
    save_best_model_path = os.path.join(root_log_path,'best_model.pth')
    save_final_model_path = os.path.join(root_log_path,'final_model.pth')
    writer = SummaryWriter(root_log_path)
    save_reward = -100
    for _ in range(100):
        avg_reward,obs,done = run_eval_episodes(agent,env,EVAL_EPISODES)
        writer.add_scalar('eval/episode_reward',avg_reward,total_steps)
        # logger.info('obs:{} done:{}, Reward:{}'.format(obs,done,avg_reward))
        save_every_model_path = os.path.join(root_log_path,'every_{}_steps_model.pth'.format(total_steps))
        agent.save(save_every_model_path)
        if avg_reward > save_reward:
            agent.save(save_best_model_path)
            save_reward = avg_reward
    agent.save(save_final_model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_path',default='./logs/test',type=str
    )
    parser.add_argument(
        '--train_total_steps',default=1e6,type=int
    )
    parser.add_argument(
        '--test_every_steps',default=1e3,type=int
    )
    args = parser.parse_args()
    main()
    





    

    
