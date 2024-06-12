#!/usr/bin/env python
"""
Task environment for two loggers escaping from the walled cell, cooperatively.
"""
from __future__ import absolute_import, division, print_function

import sys
from os import sys, path

import numpy as np
from numpy import pi
from numpy import random
import math
import time

import rospy
import tf
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, SetLinkState
from gazebo_msgs.msg import ModelState, LinkState, ModelStates, LinkStates
from geometry_msgs.msg import Pose, Twist,WrenchStamped,Vector3
import message_filters


# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
# import utils
# from utils import double_utils


def generate_random_pose():
    """
    generate a random rod pose in the room
    with center at (0, 0), width 10 meters and depth 10 meters.
    The robot has 0.2 meters as radius
    Returns:
        random_pose
    """
    def angleRange(x, y, room, L):
        """
        Compute rod angle based on a given robot position
        """
        min = 0
        max = 0
        dMinX = abs(x-room[0])
        dMaxX = abs(x-room[1])
        dMinY = abs(y-room[2])
        dMaxY = abs(y-room[3])
        if dMinX < L:
            if dMinY < L:
                min = -0.5*math.pi+math.acos(dMinY/L)
                max = math.pi-math.acos(dMinX/L)
            elif dMaxY < L:
                min = -math.pi+math.acos(dMinX/L)
                max = 0.5*math.pi-math.acos(dMaxY/L)
            else:
                min = -math.pi + math.acos(dMinX/L)
                max = math.pi-math.acos(dMinX/L)
        elif dMaxX < L:
            if dMinY < L:
                min = math.acos(dMaxX/L)
                max = 1.5*math.pi-math.acos(dMinY/L)
            elif dMaxY < L:
                min = 0.5*math.pi+math.acos(dMaxY/L)
                max = 2*math.pi-math.acos(dMaxX/L)
            else:
                min = math.acos(dMaxX/L)
                max = 2*math.pi-math.acos(dMaxX/L)
        else:
            if dMinY < L:
                min = -0.5*math.pi+math.acos(dMinY/L)
                max = 1.5*math.pi-math.acos(dMinY/L)
            elif dMaxY < L:
                min = 0.5*math.pi+math.acos(dMaxY/L)
                max = 2.5*math.pi-math.acos(dMaxY/L)
            else:
                min = -math.pi
                max = math.pi
        return min, max

    random_pose = []
    mag = 4.5
    len_rod = 2
    room = [-mag, mag, -mag, mag] # create a room with boundary
    # randomize robot position
    rx = random.uniform(-mag, mag)
    ry = random.uniform(-mag, mag)
    # randomize rod pose
    min_angle, max_angle = angleRange(rx, ry, room, len_rod)
    angle = random.uniform(min_angle, max_angle)
    x = rx + 0.5*len_rod*math.cos(angle)
    y = ry + 0.5*len_rod*math.sin(angle)
    # randomize robots orientation
    th_0 = random.uniform(-math.pi, math.pi)
    th_1 = random.uniform(-math.pi, math.pi)
    random_pose = [x, y, angle, th_0, th_1]

    return random_pose

class DoubleEscapeEnv(object):
    """
    DoubleEscape Class
    """
    def __init__(self):
        rospy.init_node("double_escape_task_env", anonymous=True, log_level=rospy.INFO)
        # init simulation parameters
        self.rate = rospy.Rate(100)
        # init environment parameters
        self.observation = dict(
            rod=dict(
                force=Vector3(),
                torque=Vector3()),
            robot1=dict(
                pose=Pose(),
                twist=Twist()),
            robot2=dict(
                pose=Pose(),
                twist=Twist())
        )
        self.action_1 = np.zeros(2)
        self.action_2 = np.zeros(2)
        self.info = dict(status="")
        self.reward = 0
        self._episode_done = False
        self.success_count = 0
        self.max_step = 2000
        self.steps = 0
        self.status = ['trapped', 'trapped']
        # self.model_states = ModelStates()
        # self.link_states = LinkStates()
        self.robot1_force_data = WrenchStamped()
        self.robot2_force_data = WrenchStamped()
        self.robot_position_data = ModelStates()
        # services
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.unpause_physics_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        # self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        # self.set_link_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetLinkState)
        # topic publisher
        self.robot1_cmd_vel_pub = rospy.Publisher("/robot1_cmd_vel", Twist, queue_size=1)
        self.robot2_cmd_vel_pub = rospy.Publisher("/robot2_cmd_vel", Twist, queue_size=1)
        # topic subscriber
        
        # rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_states_callback2)
        # # rospy.Subscriber("/gazebo/link_states", LinkStates, self._link_states_callback)
        # robot1_force_sub = message_filters.Subscriber('/robot1_ft_sensor', WrenchStamped, queue_size=1)
        # robot2_force_sub = message_filters.Subscriber('/robot2_ft_sensor', WrenchStamped, queue_size=1)
        # robot_postion_sub = message_filters.Subscriber('/gazebo/model_states', ModelStates, queue_size=1)
        # sync_robot = message_filters.ApproximateTimeSynchronizer([robot1_force_sub, robot2_force_sub,robot_postion_sub],10,0.1,allow_headerless=False)
        # sync_robot.registerCallback(self._model_states_callback)
    def pausePhysics(self):
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause_physics_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/pause_physics service call failed")

    def unpausePhysics(self):
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause_physics_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/unpause_physics service call failed")

    def resetSimulation(self):
        rospy.wait_for_service("/gazebo/reset_simulation")
        try:
            self.reset_simulation_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/reset_simulation service call failed")

    def resetWorld(self):
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_world_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/reset_world service call failed")

    def setModelState(self, model_state):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.set_model_state_proxy(model_state)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: {}".format(e))

    def setLinkState(self, link_state):
        rospy.wait_for_service('/gazebo/set_link_state')
        try:
            self.set_link_state_proxy(link_state)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: {}".format(e))

    def reset(self, init_pose=[]):
        """
        Reset environment function
        obs, info = env.reset()
        """
        rospy.logdebug("\nStart Environment Reset")
        # self._set_init(init_pose)
        self.resetSimulation()
        obs = self._get_observation()
        info = self._post_information()
        self.steps = 0
        rospy.logwarn("\nEnd Environment Reset!!!\n")

        return obs, info

    def step(self, action_1, action_2):
        """
        Manipulate logger_0 with action_0, logger_1 with action_1
        obs, rew, done, info = env.step(action_0, action_1)
        """
        print('action action action')
        rospy.logdebug("\nStart Environment Step")
        self._take_action(action_1, action_2)

        obs = self._get_observation()
        reward, done = self._compute_reward()
        print('tatatataaction action action')
        info = self._post_information()
        self.steps += 1
        rospy.logdebug("End Environment Step\n")

        return obs, reward, done, info

    def _set_init(self, init_pose):
        """
        Set initial condition of two_loggers to a specific pose
        Args:
            init_pose: [rod::x, y, angle, robot_0::angle, robot_1::angle], set to a random pose if empty
        """
        rospy.logdebug("\nStart Initializing Robots")
        # prepare
        self._take_action(np.zeros(2), np.zeros(2))
        self.pausePhysics()
        self.resetWorld()
        # set rod pose
        rod_state = ModelState()
        rod_state.model_name = "two_loggers"
        rod_state.reference_frame = "world"
        rod_state.pose.position.z = 0.245
        if init_pose: # random initialize
            assert -pi<=init_pose[2]<= pi # theta within [-pi,pi]
            assert -pi<=init_pose[3]<= pi
            assert -pi<=init_pose[4]<= pi
        else:
            init_pose = generate_random_pose()
        rod_state.pose.position.x = init_pose[0]
        rod_state.pose.position.y = init_pose[1]
        quat = tf.transformations.quaternion_from_euler(0, 0, init_pose[2])
        rod_state.pose.orientation.z = quat[2]
        rod_state.pose.orientation.w = quat[3]
        # call '/gazebo/set_model_state' service
        self.setModelState(model_state=rod_state)
        rospy.logdebug("two-logges was initialized at {}".format(rod_state))
        self.unpausePhysics()
        link_states = self.link_states
        self.pausePhysics()
        # set robot orientation
        q0 = tf.transformations.quaternion_from_euler(0, 0, init_pose[3])
        q1 = tf.transformations.quaternion_from_euler(0, 0, init_pose[4])
        # set white robot orientation
        robot0_name = 'two_loggers::link_chassis_0'
        robot0_state = LinkState()
        robot0_state.link_name = robot0_name
        robot0_state.reference_frame = 'world'
        robot0_state.pose = link_states.pose[link_states.name.index(robot0_name)]
        robot0_state.pose.orientation.z = q0[2]
        robot0_state.pose.orientation.z = q0[3]
        # set black robot orientation
        robot1_name = 'two_loggers::link_chassis_1'
        robot1_state = LinkState()
        robot1_state.link_name = robot1_name
        robot1_state.reference_frame = 'world'
        robot1_state.pose = link_states.pose[link_states.name.index(robot1_name)]
        robot1_state.pose.orientation.z = q1[2]
        robot1_state.pose.orientation.z = q1[3]
        # call '/gazebo/set_link_state' service
        self.setLinkState(link_state=robot0_state)
        self.setLinkState(link_state=robot1_state)
        self.unpausePhysics()
        self._take_action(np.zeros(2), np.zeros(2))
        rospy.logdebug("\ntwo_loggers initialized at {} \nlogger_0 orientation: {} \nlogger_1 orientation".format(rod_state, init_pose[3], init_pose[4]))
        # episode should not be done
        self._episode_done = False
        rospy.logdebug("End Initializing Robots\n")

    def _get_observation(self):
        """
        Get observation from link_states
        Return:
            observation: {"log{"pose", "twist"}", logger0{"pose", "twist"}", logger1{"pose", "twist"}"}
        """
        rospy.logdebug("\nStart Getting Observation")
        # link_states = self.link_states
        self.robot_position_data = rospy.wait_for_message('/gazebo/model_states',ModelStates)
        self.robot1_force_data = rospy.wait_for_message('/robot1_ft_sensor',WrenchStamped)
        self.robot2_force_data = rospy.wait_for_message('/robot2_ft_sensor',WrenchStamped)
        robot1_force_data = self.robot1_force_data
        robot2_force_data = self.robot2_force_data
        robot_position_data = self.robot_position_data
        self.pausePhysics()
        # robot1 position
        id_robot1 = robot_position_data.name.index('mycar1')
        self.observation['robot1']['pose'] = robot_position_data.pose[id_robot1]
        self.observation['robot1']['twist'] = robot_position_data.twist[id_robot1]
        # robot2 position
        id_robot2 = robot_position_data.name.index('mycar2')
        self.observation['robot2']['pose'] = robot_position_data.pose[id_robot2]
        self.observation['robot2']['twist'] = robot_position_data.twist[id_robot2]
        # the rod
        self.observation['rod']['force'] = robot1_force_data.wrench.force
        self.observation['rod']['torque'] = robot1_force_data.wrench.torque
        # compute logger_0's status
        '''
        if self.observation['logger_0']['pose'].position.x > 4.79:
            self.status[0] = 'east'
        elif self.observation['logger_0']['pose'].position.x < -4.79:
            self.status[0] = 'west'
        elif self.observation['logger_0']['pose'].position.y > 4.79:
            self.status[0] = 'north'
        elif -6<=self.observation['logger_0']['pose'].position.y < -4.79:
            # if np.absolute(self.observation['logger_0']['pose'].position.x) > 1:
            if np.absolute(self.observation['logger_0']['pose'].position.x) > 0.5:
                self.status[0] = 'south'
            else:
                # if np.absolute(self.observation['logger_0']['pose'].position.x) > 0.79:
                if np.absolute(self.observation['logger_0']['pose'].position.x) > 0.295:
                    self.status[0] = 'door' # stuck at door
                else:
                    self.status[0] = 'tunnel' # through door
        elif self.observation['logger_0']['pose'].position.y < -6:
            self.status[0] = 'escaped'
        elif self.observation['logger_0']['pose'].position.z > 0.1 or self.observation['logger_0']['pose'].position.z < 0.08:
            self.status[0] = 'blew'
        else:
            self.status[0] = 'trapped'
        # compute logger_1's status
        if self.observation['logger_1']['pose'].position.x > 4.79:
            self.status[1] = 'east'
        elif self.observation['logger_1']['pose'].position.x < -4.79:
            self.status[1] = 'west'
        elif self.observation['logger_1']['pose'].position.y > 4.79:
            self.status[1] = 'north'
        elif -6<=self.observation['logger_1']['pose'].position.y < -4.79:
            # if np.absolute(self.observation['logger_1']['pose'].position.x) > 1:
            if np.absolute(self.observation['logger_1']['pose'].position.x) > 0.5:
                self.status[1] = 'south'
            else:
                # if np.absolute(self.observation['logger_1']['pose'].position.x) > 0.79:
                if np.absolute(self.observation['logger_1']['pose'].position.x) > 0.295:
                    self.status[1] = 'door' # stuck at door
                else:
                    self.status[1] = 'tunnel' # through door
        elif self.observation['logger_1']['pose'].position.y < -6:
            self.status[1] = 'escaped'
        elif self.observation['logger_1']['pose'].position.z > 0.1 or  self.observation['logger_1']['pose'].position.z < 0.08:
            self.status[1] = 'blew'
        else:
            self.status[1] = 'trapped'
        self.unpausePhysics()
        # logging
        rospy.logdebug("Observation Get ==> {}".format(self.observation))
        rospy.logdebug("End Getting Observation\n")
        '''
        self.unpausePhysics()

        return self.observation

    def _take_action(self, action_1, action_2):
        """
        Set linear and angular speed for logger_0 and logger_1 to execute.
        Args:
            action: 2x np.array([v_lin,v_ang]).
        """
        rospy.logdebug("\nStart Taking Actions")
        cmd_vel_1 = Twist()
        cmd_vel_1.linear.x = action_1[0]
        cmd_vel_1.linear.y = action_1[1]
        cmd_vel_1.angular.z = action_1[2]
        cmd_vel_2 = Twist()
        cmd_vel_2.linear.x = action_2[0]
        cmd_vel_2.linear.y = action_2[1]
        cmd_vel_2.angular.z = action_2[2]
        for _ in range(15):
            self.robot1_cmd_vel_pub.publish(cmd_vel_1)
            self.robot1_cmd_vel_pub.publish(cmd_vel_2)
            self.rate.sleep()
        self.action_1 = action_1
        self.action_2 = action_2
        rospy.logdebug("\nrobot_1 take action ===> {}\nrobot_2 take action ===> {}".format(cmd_vel_1, cmd_vel_2))
        rospy.logdebug("End Taking Actions\n")

    def _compute_reward(self):
        """
        Return:
            reward: reward in current step
        """
        # rospy.logdebug("\nStart Computing Reward")
        # if self.status[0] == "escaped" and self.status[1] == "escaped":
        #     self.reward = 1
        #     self.success_count += 1
        #     self._episode_done = True
        #     rospy.logerr("\nDouble Escape Succeed!\n")
        # else:
        #     self.reward = -0.
        #     self._episode_done = False
        #     rospy.loginfo("The loggers are trapped in the cell...")
        # rospy.logdebug("Stepwise Reward: {}, Success Count: {}".format(self.reward, self.success_count))
        # # check if steps out of range
        # if self.steps > self.max_step:
        #     self._episode_done = True
        #     rospy.logwarn("Step: {}, \nMax step reached, env will reset...".format(self.steps))
        # rospy.logdebug("End Computing Reward\n")
        self.reward = 1
        self._episode_done = False
        return self.reward, self._episode_done

    def _post_information(self):
        """
        Return:
            info: {"system status"}
        """
        rospy.logdebug("\nStart Posting Information")
        self.info["status"] = self.status
        rospy.logdebug("End Posting Information\n")

        return self.info
'''
    def _model_states_callback(self,robot1_force_data,robot2_force_data,robot_position_data):
        self.robot1_force_data = robot1_force_data
        self.robot2_force_data = robot2_force_data
        self.robot_position_data = robot_position_data
        rospy.loginfo('nxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    def _model_states_callback2(self,robot_position_data):
        
        self.robot_position_data = robot_position_data
        rospy.loginfo('n*****************************')

    def _link_states_callback(self, data):
        self.link_states = data
        '''
