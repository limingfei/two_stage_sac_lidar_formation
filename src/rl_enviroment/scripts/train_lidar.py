#!/usr/bin/env python
import numpy as np
from agent_lidar import GazeboAgent
# from model_without_lstm import GazeboModel
from model_lidar import GazeboModel,TeacherModel
import argparse
from parl.utils import logger
from cpprb import ReplayBuffer
from adaptive_teacher_alpha_lidar import GazeboSAC
from trancking_agv_lidar_regis import DoubleEscapeEnv
import datetime
import os
from rich.progress import track
from control_pid import PidControl

import torch
import rospy
import  time
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


from tensorboardX import SummaryWriter
WARMUP_STEPS = 1e3
# WARMUP_STEPS = 10

EVAL_EPISODES = 3
MEMORY_SIZE = int(1e4)
# MEMORY_SIZE = int(1e2)
BATCH_SIZE = 64
# BATCH_SIZE = 8
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.01
# ACTOR_LR = 3e-4
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4
BC_LR = 3e-4
NUM_STEPS = 50
import warnings
warnings.simplefilter("ignore", ResourceWarning)

def square_distance(pcd1, pcd2):
    """
    Squared distance between any two points in the two point clouds.
    """
    return np.sum((pcd1[:, None, :] - pcd2[None, :, :]) ** 2,axis=-1)


def run_train_episode(agent,env,rpm,pid_control,total_step,writer):
    action_dim = 3
    src,vel,obs = env.reset()
    tgt = src
    done = False
    episode_reward,episode_steps = 0,0
   
    while True:
        total_step += 1
        episode_steps += 1
        # pid_control = PidControl(env.parms)
    
        if rpm.get_stored_size() < WARMUP_STEPS:
            # student_act = pid_control.pid_get_follower_cmd(env._relative_pos_cmd)
            student_act = agent.student_sample(src,tgt,vel)
            # action = agent.sample(obs)
            # if len(os.listdir('./collect_dataset_teacher')) > 0:
            #     action = agent.sample(obs)
            # else:
            action = np.random.uniform(-1,1,size=action_dim)            
        else:
            student_act = agent.student_sample(src,tgt,vel)
            # expert_act = pid_control.pid_get_follower_cmd(env._relative_pos_cmd)
            action = agent.sample(obs)
        
        (next_src,next_vel,next_obs),reward,done,info = env.step(action)
        # dist = square_distance(next_src,tgt)
        # dist = np.min(dist,axis=-1)
        # dist = np.mean(dist,axis=-1)

        rpm.add(src=src,tgt=tgt,vel=vel,obs=obs,act=action,student_act=student_act,reward=reward,next_src=next_src,next_vel=next_vel,next_obs=next_obs,done=done)
        src= next_src
        vel = next_vel
        obs = next_obs
        episode_reward += reward

        # if rpm.get_stored_size()  == MEMORY_SIZE:
        #     rpm.save_transitions('./collect_dataset_teacher/{}'.format(len(os.listdir('./collect_dataset_teacher'))))
        #     rpm.clear()
            

        if rpm.get_stored_size() >= WARMUP_STEPS:
            samples = rpm.sample(BATCH_SIZE)
            batch_src = samples['src']
            batch_tgt = samples['tgt']
            batch_vel = samples['vel']
            batch_obs = samples['obs']
            batch_act = samples['act']
            # batch_student_act = samples['student_act']
            batch_reward = samples['reward']
            # batch_next_src = samples['next_src']
            # batch_next_vel = samples['next_vel']
            batch_next_obs = samples['next_obs']
            batch_done= samples['done']
            
            c_l,a_l,alpha_l,bc_loss = agent.learn(batch_src,batch_tgt,batch_vel,batch_obs,batch_act,batch_reward,batch_next_obs,batch_done)
            writer.add_scalar('bc_loss',bc_loss.cpu().item(),total_step)
            writer.add_scalar('c_l',c_l.cpu().item(),total_step)
            writer.add_scalar('a_l',a_l.cpu().item(),total_step)
            writer.add_scalar('alpha_l',alpha_l.cpu().item(),total_step)
            # writer.add_scalar('alpha',agent.alpha.cpu().item(),total_step)
        if done:
            break

    
    return episode_reward,episode_steps


def run_eval_episodes(agent,env,eval_episodes,student_mode,writer=None):
    eval_reward = []
    episode_steps = 0
    for _ in range(eval_episodes):
        src,vel,obs = env.reset()
        tgt = src
        episode_reward = 0.0
        
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
                action = agent.predict(src,tgt,vel,obs,student_mode)
            (src,vel,obs),reward,done,info = env.step(action)
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
    env_parms = env.parms
    obs_dim = int(360+9)
    act_dim = int(3)
    student_model = GazeboModel()
    model = TeacherModel()
    
    ######################################
  
    ######################################
    algorithm = GazeboSAC(model,student_model=student_model,gamma=GAMMA,tau=TAU,alpha=ALPHA,actor_lr=ACTOR_LR,critic_lr=CRITIC_LR,alpha_lr=ALPHA_LR,action_dim=act_dim,bc_lr=BC_LR,adaptive_alpha=True)
    agent = GazeboAgent(algorithm)
    rpm = ReplayBuffer(MEMORY_SIZE,
                       env_dict = {"src":{"shape":(768,3)},
                                   "tgt":{"shape":(768,3)},
                                   "vel":{'shape':3},
                                   "obs":{'shape':9},
                                   "act":{'shape':3},
                                   "student_act":{'shape':3},
                                   "reward":{'shape':1},
                                   "next_src":{"shape":(768,3)},
                                   "next_vel":{'shape':3},
                                   "next_obs":{'shape':9},
                                   "done":{'shape':1}})
    train_num = 0
    total_step = 0

    pid_control = PidControl(env.parms)
    
    
    save_reward = -100
    if args.test:
        agent.restore(args.model_parms)
        args.log_path = './logs_test_fuzzy_fuzzy'
        root_log_path = os.path.join(args.log_path,'use_pid_{}_noise_{}_tn_{}_pn_{}_speed_{}'.format(args.use_pid,env_parms['noise'],env_parms['time_noise'],env_parms['pos_noise'],env_parms['high_bound'][0]))
        save_best_model_path = os.path.join(root_log_path,'best_model.pth')
        save_final_model_path = os.path.join(root_log_path,'final_model.pth')
        writer = SummaryWriter(root_log_path)
        for i in range(1):
            avg_reward,eval_episode_steps = run_eval_episodes(agent,env,1,student_mode=True,writer=writer)
            writer.add_scalar('eval/episode_reward',avg_reward,i)
            writer.add_scalar('eval/episode_steps',eval_episode_steps,i)
            
            logger.info('Eval over:{} episodes, Reward:{}, episode_steps:{}'.format(i,avg_reward,eval_episode_steps))
    else:
        # agent.restore(args.model_parms)
        root_log_path = os.path.join(args.log_path,'sac_{}'.format(datetime.datetime.now().__str__().replace(':','_')))
        save_best_model_path = os.path.join(root_log_path,'best_model.pth')
        save_final_model_path = os.path.join(root_log_path,'final_model.pth')
        writer = SummaryWriter(root_log_path)
        while rpm.get_stored_size() < WARMUP_STEPS:
            run_train_episode(agent,env,rpm,pid_control,total_step,writer)
            print('len rpm',rpm.get_stored_size())
        while train_num < args.max_train_num:
            
            for i in range(0,25):
                episode_reward,episode_steps = run_train_episode(agent,env,rpm,pid_control,total_step,writer)
                train_num += 1
                total_step += episode_steps
                writer.add_scalar('train/episode_reward',episode_reward,total_step)
                writer.add_scalar('train/episode_steps',episode_steps,total_step)
                writer.add_scalar('train/alpha',algorithm.alpha.detach().cpu(),total_step)
                logger.info('train_num:{}, time noise:{}, Reward:{}'.format(train_num,env.time_noise,episode_reward))

            print('memory size:',rpm.get_stored_size())
            student_avg_reward,student_eval_episode_steps = run_eval_episodes(agent,env,EVAL_EPISODES,student_mode=True)
            logger.info('Eval student:{} steps, Reward:{}'.format(student_eval_episode_steps,student_avg_reward))
            writer.add_scalar('eval_student/episode_reward',student_avg_reward,total_step)
            writer.add_scalar('eval_student/episode_steps',student_eval_episode_steps,total_step)
            teacher_avg_reward,teacher_eval_episode_steps = run_eval_episodes(agent,env,EVAL_EPISODES,student_mode=False)
            logger.info('Eval teacher:{} steps, Reward:{}'.format(teacher_eval_episode_steps,teacher_avg_reward))
            writer.add_scalar('eval_teacher/episode_reward',teacher_avg_reward,total_step)
            writer.add_scalar('eval_teacher/episode_steps',teacher_eval_episode_steps,total_step)
            if student_avg_reward > save_reward:
                agent.save(save_best_model_path)
                save_reward = student_avg_reward
        agent.save(save_final_model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_path',default='./logs_teacher_student',type=str
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
        '--model_parms',default='logs_teacher_student/sac_2024-05-07 22_49_38.400247/best_student_model.pth',type=str
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
    





    

    
