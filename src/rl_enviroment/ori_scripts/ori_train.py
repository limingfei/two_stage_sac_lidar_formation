#!/usr/bin/env python
import numpy as np
from ori_agent import GazeboAgent
from ori_model_without_lstm import GazeboModel
import argparse
from parl.utils import logger,ReplayMemory
from parl.env import ActionMappingWrapper
from ori_adaptive_alpha import GazeboSAC
from ori_trancking_agv import DoubleEscapeEnv
import datetime
import os
from rich.progress import track

from tensorboardX import SummaryWriter

WARMUP_STEPS = 1e3
EVAL_EPISODES = 3
MEMORY_SIZE = int(1.5e4)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.01
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4
import warnings
warnings.simplefilter("ignore", ResourceWarning)
def run_train_episode(agent,env,rpm):
    action_dim = 3
    obs = env.reset()
    done = False
    episode_reward,episode_steps = 0,0
    while not done:
        episode_steps += 1
        if rpm.size() < WARMUP_STEPS:
            action = np.random.uniform(-1,1,size=action_dim)
            print('rpm size:',rpm.size())
        else:
            action = agent.sample(obs)
        
        next_obs,reward,done,info = env.step(action)
        rpm.append(obs,action,reward,next_obs,done)
        obs = next_obs
        episode_reward += reward

        if rpm.size() > WARMUP_STEPS:
            batch_obs,batch_action,batch_reward,batch_next_obs,batch_terminal = rpm.sample_batch(BATCH_SIZE)
            agent.learn(batch_obs,batch_action,batch_reward,batch_next_obs,batch_terminal)

    
    return episode_reward,episode_steps


def run_eval_episodes(agent,env,eval_episodes):
    avg_reward = 0.0
    episode_steps = 0
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False

        while not done:
            episode_steps += 1
            action = agent.predict(obs)
            obs,reward,done,info = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    episode_steps /= eval_episodes
    return avg_reward,int(episode_steps)


def main():
    env = DoubleEscapeEnv(args.env_parms)
    obs_dim = int(9)
    act_dim = int(3)
    model = GazeboModel(obs_dim,act_dim)
    algorithm = GazeboSAC(model,gamma=GAMMA,tau=TAU,alpha=ALPHA,actor_lr=ACTOR_LR,critic_lr=CRITIC_LR,alpha_lr=ALPHA_LR,action_dim=act_dim)

    agent = GazeboAgent(algorithm)
    rpm = ReplayMemory(MEMORY_SIZE,obs_dim=obs_dim,act_dim=act_dim)
    total_steps = 0
    test_flag = 0
    root_log_path = os.path.join(args.log_path,'sac_{}'.format(datetime.datetime.now().__str__().replace(':','_')))
    save_best_model_path = os.path.join(root_log_path,'best_model.pth')
    save_final_model_path = os.path.join(root_log_path,'final_model.pth')
    writer = SummaryWriter(root_log_path)
    save_reward = -100
    while len(rpm) < WARMUP_STEPS:
        run_train_episode(agent,env,rpm)
    while total_steps < args.train_total_steps:
        episode_reward,episode_steps = run_train_episode(agent,env,rpm)
        total_steps += episode_steps
        writer.add_scalar('train/episode_reward',episode_reward,total_steps)
        writer.add_scalar('train/episode_steps',episode_steps,total_steps)
        writer.add_scalar('train/alpha',algorithm.alpha.detach().cpu(),total_steps)
        logger.info('Total Steps:{}, Reward:{}'.format(total_steps,episode_reward))

        if (total_steps+1) // args.test_every_steps >= test_flag:
            while (total_steps+1) // args.test_every_steps >= test_flag:
                test_flag += 1
            avg_reward,eval_episode_steps = run_eval_episodes(agent,env,EVAL_EPISODES)
            writer.add_scalar('eval/episode_reward',avg_reward,total_steps)
            writer.add_scalar('eval/episode_steps',eval_episode_steps,total_steps)
            logger.info('Eval over:{} episodes, Reward:{}'.format(EVAL_EPISODES,avg_reward))
            # save_every_model_path = os.path.join(root_log_path,'every_{}_steps_model.pth'.format(total_steps))
            # agent.save(save_every_model_path)
            if avg_reward > save_reward:
                agent.save(save_best_model_path)
                save_reward = avg_reward
    agent.save(save_final_model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_path',default='./logs_without_lstm',type=str
    )
    parser.add_argument(
        '--train_total_steps',default=1e6,type=int
    )
    parser.add_argument(
        '--test_every_steps',default=1e3,type=int
    )
    parser.add_argument(
        '--env_parms',default='src/rl_enviroment/config/default_test_pid_config.yaml',type=str
    )
    args = parser.parse_args()
    main()
    





    

    
