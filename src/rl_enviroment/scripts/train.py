#!/usr/bin/env python
import numpy as np
from agent import GazeboAgent
# from model_without_lstm import GazeboModel
from model import GazeboModel
import argparse
from parl.utils import logger
from rpm import ReplayMemory
from adaptive_alpha import GazeboSAC
from trancking_agv_lidar_regis import DoubleEscapeEnv
import datetime
import os
from rich.progress import track
from control_pid import PidControl
import torch
import rospy
import  time
import random


from tensorboardX import SummaryWriter
WARMUP_STEPS = 1e3
# WARMUP_STEPS = 10

EVAL_EPISODES = 3
MEMORY_SIZE = int(1e4)
BATCH_SIZE = 64
# BATCH_SIZE = 8
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.01
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4
NUM_STEPS = 50
import warnings
warnings.simplefilter("ignore", ResourceWarning)
def run_train_episode(agent,env,rpm,pid_control,use_pid):
    if use_pid:
        print('use pid:')
    else:
        print('use model:')
    
    action_dim = 3
    obs = env.reset()
    done = False
    episode_reward,episode_steps = 0,0
    agent.alg.model.init_lstm_state_train()
    agent.alg.target_model.init_lstm_state_train()
    while True:
        episode_steps += 1
        if len(rpm) < WARMUP_STEPS:
                # action = np.random.uniform(-1,1,size=action_dim)
            # pid_control = PidControl(env.parms)
            action = pid_control.pid_get_follower_cmd(env._relative_pos_cmd)

        else:
            # pid_control = PidControl(env.parms)

            if use_pid:
                action = pid_control.pid_get_follower_cmd(env._relative_pos_cmd)
            else:
                action = pid_control.pid_get_follower_cmd(env._relative_pos_cmd)

                # action = agent.sample(obs)
        
        next_obs,reward,done,info = env.step(action)
        rpm.append((obs,action,reward,next_obs,done),done)
        obs = next_obs
        episode_reward += reward
        if env.motion_done:
            break

    if len(rpm) >= WARMUP_STEPS:
        agent.alg.model.init_lstm_state(BATCH_SIZE)
        agent.alg.target_model.init_lstm_state(BATCH_SIZE)
        (batch_obs,batch_action,batch_reward,batch_next_obs,batch_terminal) = rpm.sample(BATCH_SIZE)
        agent.learn(batch_obs,batch_action,batch_reward,batch_next_obs,batch_terminal)

    
    return episode_reward,episode_steps


def run_eval_episodes(agent,env,eval_episodes,writer=None):
    eval_reward = []
    episode_steps = 0
    for _ in range(eval_episodes):
        obs = env.reset()
        episode_reward = 0.0
        agent.alg.model.init_lstm_state_train()
        agent.alg.target_model.init_lstm_state_train()
        step = 0

        while True:
            if writer is not None:
                x = env.observation['pose']['x']
                y = env.observation['pose']['y']
                yaw = env.observation['pose']['yaw']
                writer.add_scalar('eval/x',x,step)
                writer.add_scalar('eval/y',y,step)
                writer.add_scalar('eval/yaw',yaw,step)
                step += 1
            episode_steps += 1
            
            if args.use_pid:
                action = agent.use_pid(env,True)
            else:
                action = agent.predict(obs)
            obs,reward,done,info = env.step(action)
            episode_reward += reward
            if args.test:
                if step == env.max_step:
                    break
            else:
                if done:
                    break
        eval_reward.append(episode_reward)
    
    episode_steps /= eval_episodes
    return np.mean(eval_reward),int(episode_steps)


def main():
    env_parms_path = args.env_parms
    env = DoubleEscapeEnv(env_parms_path)
    pid_control = PidControl(env.parms)
    # np.random.seed(1)
    # torch.manual_seed(1)
    # random.seed(1)
    env_parms = env.parms
    obs_dim = int(4)
    act_dim = int(3)
    model = GazeboModel(obs_dim,act_dim)
    ######################################
  
    ######################################
    algorithm = GazeboSAC(model,gamma=GAMMA,tau=TAU,alpha=ALPHA,actor_lr=ACTOR_LR,critic_lr=CRITIC_LR,alpha_lr=ALPHA_LR,action_dim=act_dim)
    agent = GazeboAgent(algorithm,env)
    rpm = ReplayMemory(MEMORY_SIZE,NUM_STEPS)
    total_step = 0
    train_num = 0
    test_flag = 0

    
    
    save_reward = -100
    if args.test:
        agent.restore(args.model_parms)
        args.log_path = './logs_test_fuzzy_fuzzy'
        root_log_path = os.path.join(args.log_path,'use_pid_{}_noise_{}_tn_{}_pn_{}_speed_{}'.format(args.use_pid,env_parms['noise'],env_parms['time_noise'],env_parms['pos_noise'],env_parms['high_bound'][0]))
        save_best_model_path = os.path.join(root_log_path,'best_model.pth')
        save_final_model_path = os.path.join(root_log_path,'final_model.pth')
        writer = SummaryWriter(root_log_path)
        for i in range(1):
            avg_reward,eval_episode_steps = run_eval_episodes(agent,env,1,writer)
            writer.add_scalar('eval/episode_reward',avg_reward,i)
            writer.add_scalar('eval/episode_steps',eval_episode_steps,i)
            
            logger.info('Eval over:{} episodes, Reward:{}, episode_steps:{}'.format(i,avg_reward,eval_episode_steps))
    else:
        root_log_path = os.path.join(args.log_path,'sac_{}'.format(datetime.datetime.now().__str__().replace(':','_')))
        save_best_model_path = os.path.join(root_log_path,'best_model.pth')
        save_final_model_path = os.path.join(root_log_path,'final_model.pth')
        writer = SummaryWriter(root_log_path)
        while len(rpm) < WARMUP_STEPS:
            total_step += 1
            if total_step % 3 == 0:
                use_pid = False
            else:
                use_pid = False
            run_train_episode(agent,env,rpm,pid_control,use_pid)
            print('len rpm',len(rpm))
        while train_num < args.max_train_num:
            
            for i in range(0,50):
                total_step += 1
                if total_step % 3 == 0:
                    use_pid = False
                else:
                    use_pid = False
                episode_reward,episode_steps = run_train_episode(agent,env,rpm,pid_control,use_pid)
                train_num += 1
                writer.add_scalar('train/episode_reward',episode_reward,train_num)
                writer.add_scalar('train/episode_steps',episode_steps,train_num)
                writer.add_scalar('train/alpha',algorithm.alpha.detach().cpu(),train_num)
                logger.info('train_num:{}, time noise:{}, Reward:{}'.format(train_num,env.time_noise,episode_reward))

            
            avg_reward,eval_episode_steps = run_eval_episodes(agent,env,EVAL_EPISODES)
            writer.add_scalar('eval/episode_reward',avg_reward,train_num)
            writer.add_scalar('eval/episode_steps',eval_episode_steps,train_num)
            logger.info('Eval time noise:{} episodes, Reward:{}'.format(env.time_noise,avg_reward))
            if avg_reward > save_reward:
                agent.save(save_best_model_path)
                save_reward = avg_reward
        agent.save(save_final_model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_path',default='./logs',type=str
    )
    parser.add_argument(
        '--max_train_num',default=1e6,type=int
    )
    parser.add_argument(
        '--test_every_steps',default=1e3,type=int
    )
    parser.add_argument(
        '--test',default=False,type=bool
    )
    # parser.add_argument(
    #     '--model_parms',default='logs_noise_time/sac_2024-03-11 22_26_09.171120/best_model.pth',type=str
    # )
    parser.add_argument(
        '--model_parms',default='logs/sac_2024-04-19 03_44_11.047963/best_model.pth',type=str
    )
    # parser.add_argument(
    #     '--model_parms',default='logs_without_lstm/sac_2024-03-13 21_07_10.371530/best_model.pth',type=str
    # )
    parser.add_argument(
        '--env_parms',default='src/rl_enviroment/config/default_lidar.yaml',type=str
    )
    
    parser.add_argument(
        '--use_pid',default=False,type=bool
    )
    args = parser.parse_args()
    main()
    





    

    
