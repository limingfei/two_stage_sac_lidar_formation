#!/usr/bin/env python
import sys
from os import sys, path
import numpy as np
import math
import time
import random

import rospy
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose, Twist

from trancking_agv import DoubleEscapeEnv
# from .utils.double_utils import create_pose_buffer, obs_to_state
import pdb
# pdb.set_trace()

if __name__ == "__main__" and __package__ is None:
    # sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    # from utils import double_utils
    num_episodes = 1
    num_steps = 20
    print('ttttttttt')

    escaper = DoubleEscapeEnv()
    print('ttttttttt')

    escaper.reset()
    print('ttttttttt')
    for ep in range(num_episodes):
        obs= escaper.reset()
        action_1 = np.random.uniform(-1,1,(3,))
        # action_1 = np.random.rand(3)
        # rospy.loginfo("Logger was reset with observation: {} \nwith information: {}".format(obs, info))
        for st in range(num_steps):
            rospy.loginfo('********* {} ********'.format(st))
            obs, rew, done, info = escaper.step(action_1)
            
            rospy.loginfo('obs: {}'.format(obs))
            rospy.loginfo('reward: {}, done: {}'.format(rew,done))
            print(escaper.steps)
            # rospy.loginfo("Episode: {}, Step: {}, action: {}, status: {}".format(ep, st, (action_0, action_1), info))
    escaper.pausePhysics()
    rospy.logwarn("test finished")

    # actions = np.array([[1,-1],[1,1]])
    # escaper.reset() # the first reset always set the model at (0,0)
    # pose_buffer = double_utils.random_pose_buffer(num_episodes)
    # for ep in range(num_episodes):
    #     obs, info = escaper.reset()
    #     action_0 = random.choice(actions)
    #     action_1 = random.choice(actions)
    #     # state_logger0 = double_utils.obs_to_state(obs, "logger_0")
    #     rospy.loginfo("Loggers were reset with observation: {} \nwith information: {}".format(obs, info))
    #     for st in range(num_steps):
    #         obs, rew, done, info = escaper.step(action_0, action_1)
    #         state_logger0 = double_utils.obs_to_state(obs, "logger_0")
    #         state_logger1 = double_utils.obs_to_state(obs, "logger_1")
    #         state_all = double_utils.obs_to_state(obs, "all")
    #         if info["status"] == "escaped":
    #             done = True
    #         elif info["status"] == "tdoor":
    #             done = False
    #         elif info["status"] == "sdoor":
    #             done = True
    #         elif info["status"] == "blew":
    #             done = True
    #         elif info["status"] == "trapped":
    #             done = False
    #         else:
    #             done = True
    #         rospy.loginfo("Episode: {}, Step: {}, \nlogger_0 state: {}, logger_1 state: {}, state_all: {} \naction_0: {}, action_1: {} \ninfo: {}".format(ep, st, state_logger0, state_logger1, state_all, action_0, action_1, info))
    #         if done:
    #              break
    # # stop loggers in the end
    # escaper.step(np.zeros(2), np.zeros(2))
