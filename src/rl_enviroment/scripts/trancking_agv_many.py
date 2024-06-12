#!/usr/bin/env python
"""
Task environment for two loggers escaping from the walled cell, cooperatively.
"""
from __future__ import absolute_import, division, print_function

import sys
from os import sys, path
import copy
import numpy as np
from numpy import pi
from numpy import random
import math
import time

from scipy.spatial.transform import Rotation
import rospy
import tf
import tf2_ros
from geometry_msgs.msg import TransformStamped

from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, SetLinkState
from gazebo_msgs.msg import ModelState, LinkState, ModelStates, LinkStates
from geometry_msgs.msg import Pose, Twist,WrenchStamped,Vector3
import message_filters
# from rl_enviroment.msg import RelativePose
from gazebo_msgs.srv import GetModelState,GetModelStateRequest
import tf
import yaml
from sensor_msgs.msg import PointCloud2,PointField
from sensor_msgs import point_cloud2 as pc2

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
    rx = np.random.uniform(-mag, mag)
    ry = np.random.uniform(-mag, mag)
    # randomize rod pose
    min_angle, max_angle = angleRange(rx, ry, room, len_rod)
    angle = np.random.uniform(min_angle, max_angle)
    x = rx + 0.5*len_rod*math.cos(angle)
    y = ry + 0.5*len_rod*math.sin(angle)
    # randomize robots orientation
    th_0 = np.random.uniform(-math.pi, math.pi)
    th_1 = np.random.uniform(-math.pi, math.pi)
    random_pose = [x, y, angle, th_0, th_1]

    return random_pose

class DoubleEscapeEnv(object):
    """
    DoubleEscape Class
    """
    def __init__(self,parms_path):
        rospy.init_node("double_escape_task_env", anonymous=True, log_level=rospy.INFO)
        # init simulation parameters

        with open(parms_path,'r') as file:
            self.parms = yaml.safe_load(file)
        self.obs_with_velocity = self.parms['obs_with_velocity']
        self.obs_with_time = self.parms['obs_with_time']
        self.exp_x_distance = np.array(self.parms['exp_x_distance'])
        self.x_max_distance = np.array(self.parms['x_max_distance'])
        self.x_min_distance = np.array(self.parms['x_min_distance'])
        self.pub_leader_rate = self.parms['pub_leader_rate']
        self.x_max_distance_err = self.x_max_distance - self.exp_x_distance
        self.y_max_distance = np.array(self.parms['y_max_distance'])
        self.exp_y_distance = np.array(self.parms['exp_y_distance'])
        self.y_max_distance_err = self.y_max_distance - self.exp_y_distance
        self.y_min_distance = np.array(self.parms['y_min_distance'])
        self.max_angle = self.parms['max_angle']
        self.max_angle_err = self.max_angle
        self.low_bound = np.array(self.parms['low_bound'])
        self.robot2_low_bound = np.array(self.parms['robot2_low_bound'])
        self.high_bound = np.array(self.parms['high_bound'])
        self.robot2_high_bound = np.array(self.parms['robot2_high_bound'])
        self.rate = rospy.Rate(100)
        self.noise = self.parms['noise']
        self.time_noise = self.parms['time_noise']
        self.pos_noise = self.parms['pos_noise']
        self.obs_with_relative_velocity = self.parms['obs_with_relative_velocity']
        if self.parms['set_random_seed']:
            np.random.seed(1)

        # init environment parameters
        self.observation = []
        self.action_1 = np.zeros(2)
        self.info = dict(status="")
        self.reward = 0
        self._episode_done = False
        self.success_count = 0
        self.max_step = self.parms['max_step']
        self.steps = 0
        self.status = ['trapped', 'trapped']
        self.behivor = [0,1,2,3,4]  # 0 x直行，1 y直行，2 斜行， 3 旋转，4 混合
        # self.buffer = tf2_ros.Buffer()
        # self.listener = tf2_ros.TransformListener(self.buffer)
        self.robot_position_data = ModelStates()
        # services
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.unpause_physics_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.client = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
        
        # topic publisher
        self.robot2_cmd_vel_pub = rospy.Publisher("/robot2_cmd_vel", Twist, queue_size=1)
        self.robot1_cmd_vel_pub = []
        self.get_relative_requset = []
        self.get_robot1_requset = []
        self.get_robot2_request = []
        self.leader_mode = self.parms['leader_mode']
        followers = self.parms['followers']
        self.point_clound_topics = np.array(self.parms['point_clound_topics'])
        for follower in followers:      

            self.robot1_cmd_vel_pub.append(rospy.Publisher(follower['follower_cmd'], Twist, queue_size=1))
            self.get_relative_requset.append(GetModelStateRequest())
            self.get_relative_requset[-1].model_name = follower['leader'] # mycar2
            self.get_relative_requset[-1].relative_entity_name = follower['follower'] # mycar1

            self.get_robot1_requset.append(GetModelStateRequest())
            self.get_robot1_requset[-1].model_name = follower['follower'] #mycar1
            self.get_robot2_request.append(GetModelStateRequest())
            self.get_robot2_request[-1].model_name = follower['leader'] #mycar2

            self.observation.append(dict(
            pose = dict(x=0.0,
            y=0.0,
            yaw=0.0),
            robot1 = dict(
            v_x = 0.0,
            v_y = 0.0,
            v_ang = 0.0),
            relative=dict(
            v_x = 0.0,
            v_y = 0.0,
            v_ang = 0.0),
            follower = dict(
            x=0.0,
            y=0.0,
            ),
            leader = dict(
            x=0.0,
            y=0.0,
            )))
        self.current_time = rospy.Time.now()
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
            self.unpausePhysics()
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
        self.resetSimulation()
        
        orig_obs = self._get_observation()
        src_list,vel_list = self._map_obs(orig_obs)
        info = self._post_information()
        self.steps = 0
        # rospy.logwarn("\nEnd Environment Reset!!!\n")
        self.robot2_cmd_pub_num = 0
        self.robot2_cmd = Twist()
        self.cmd = Twist()
        

        return src_list,vel_list

    def step(self, action_1):
        """
        Manipulate logger_0 with action_0, logger_1 with action_1
        obs, rew, done, info = env.step(action_0, action_1)
        """
        
        rospy.logdebug("\nStart Environment Step")
        time_noise = abs(np.random.normal(0,self.time_noise))
        
        time.sleep(time_noise)
        self._take_action(action_1)

        orig_obs = self._get_observation()
        src_list,vel_list = self._map_obs(orig_obs)
        # reward, done = self._compute_reward()
        reward, done = 0, False
        info = self._post_information()
        self.steps += 1
        rospy.logdebug("End Environment Step\n")

        return (src_list,vel_list), reward, done, info

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
        # self.robot_position_data = rospy.wait_for_message('/gazebo/model_states',ModelStates)      
        
        self.point_cloud_list = []
        for point_clound_topic in self.point_clound_topics:
            try:
                msg = rospy.wait_for_message(point_clound_topic,PointCloud2, timeout=20.0)
                point_clound = self.point_cloud_callback(msg)
                self.point_cloud_list.append(point_clound)
            except rospy.exceptions.ROSException as e:
                self.resetSimulation()
                msg = rospy.wait_for_message(point_clound_topic,PointCloud2, timeout=20.0)
                rospy.logerr('wait for message timeout!')



        self.pausePhysics()
        
        self.client.wait_for_service()
        for get_relative_requset,get_robot1_requset,get_robot2_request,observation in zip(self.get_relative_requset,self.get_robot1_requset,self.get_robot2_request,self.observation):
            relative_response = self.client.call(get_relative_requset)
            # self._relative_pos_cmd = self.client.call(self.get_relative_requset2)
            robot1_response = self.client.call(get_robot1_requset)
            robot2_response = self.client.call(get_robot2_request)
        # position,ang,robot1_twist,robot2_twist = self._relative_pose(self.robot_position_data,'mycar1','mycar2')
            if relative_response.success and robot1_response.success and robot2_response.success:
                # self.observation['pose']['x'] = position[0]
                # self.observation['pose']['y'] = position[1]
                # self.observation['pose']['yaw'] = ang
                # self.observation['robot1']['v_x'] = robot1_twist.linear.x
                # self.observation['robot1']['v_y'] = robot1_twist.linear.y
                # self.observation['robot1']['v_ang'] = robot1_twist.angular.z
                # self.observation['robot2']['v_x'] = robot2_twist.linear.x
                # self.observation['robot2']['v_y'] = robot2_twist.linear.y
                # self.observation['robot2']['v_ang'] = robot2_twist.angular.z

                observation['pose']['x'] = relative_response.pose.position.x
                observation['pose']['y'] = relative_response.pose.position.y
                relative_r = Rotation.from_quat([relative_response.pose.orientation.x,relative_response.pose.orientation.y,relative_response.pose.orientation.z,relative_response.pose.orientation.w])
                # relative_r = Rotation.from_quat([robot1_response.pose.orientation.x,robot1_response.pose.orientation.y,robot1_response.pose.orientation.z,robot1_response.pose.orientation.w])
                relative_euler = relative_r.as_euler('zyx')

                # relative_euler = tf.transformations.euler_from_quaternion(relative_response.pose.orientation)
                observation['pose']['yaw'] = relative_euler[0]
                observation['follower']['x'] = robot1_response.pose.position.x
                observation['follower']['y'] = robot1_response.pose.position.y
                observation['leader']['x'] = robot2_response.pose.position.x
                observation['leader']['y'] = robot2_response.pose.position.y
                observation['robot1']['v_x'] = robot1_response.twist.linear.x
                observation['robot1']['v_y'] = robot1_response.twist.linear.y
                observation['robot1']['v_ang'] = robot1_response.twist.angular.z
                observation['relative']['v_x'] = robot2_response.twist.linear.x
                observation['relative']['v_y'] = robot2_response.twist.linear.y
                observation['relative']['v_ang'] = robot2_response.twist.angular.z
            else:
                rospy.logerr('get service error!!!!!!')
        
        self.unpausePhysics()

        return self.observation

    def _take_action(self, action_1_list):
        """
        Set linear and angular speed for logger_0 and logger_1 to execute.
        Args:
            action: 2x np.array([v_lin,v_ang]).
        """
        rospy.logdebug("\nStart Taking Actions")
        action_1_list = self._map_action(action_1_list)
        if self.leader_mode == 'straight':
            cmd_vel_2 = self.pub_robot2_str_cmd()
        elif self.leader_mode == 'circle':
            cmd_vel_2 = self.pub_robot2_cir_cmd()
        elif self.leader_mode == 'random':
            cmd_vel_2 = self.pub_robot2_cmd()
        else:
            rospy.logerr("\n leader mode not found!\n")


        # cmd_vel_2 = self.pub_robot2_cmd()
        cmd_vel_1_list = []        
        for action_1 in action_1_list:
        
            cmd_vel_1 = Twist()
            cmd_vel_1.linear.x = action_1[0]
            cmd_vel_1.linear.y = action_1[1]
            cmd_vel_1.angular.z = action_1[2]
            cmd_vel_1_list.append(cmd_vel_1)

        for _ in range(10):
            for robot1_cmd_vel_pub,cmd_vel_1 in zip(self.robot1_cmd_vel_pub,cmd_vel_1_list):
                robot1_cmd_vel_pub.publish(cmd_vel_1)
            self.robot2_cmd_vel_pub.publish(cmd_vel_2)
            self.rate.sleep()
        self.action_1 = action_1_list
        rospy.logdebug("\nrobot_1 take action ===> {}".format(cmd_vel_1))
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
       
        current_distance = math.sqrt(self.observation['pose']['x']**2 + self.observation['pose']['y']**2)
        current_angle_err = abs(self.observation['pose']['yaw'])
        self._episode_done = self._get_done(self.observation['pose']['x'],self.observation['pose']['y'],current_angle_err)
        # print('c_d:{},c_d_r:{},c_a_r:{}'.format(current_distance,current_distance_err,current_angle_err))

        self.reward = self.reward_distance(self.observation['pose']['x'],self.observation['pose']['y'],current_angle_err)

        if self.info['status'] in (0,1,2):
            self.reward = -30
        elif self.info['status'] == 3:
            self.reward == 30
        else:
            # self.reward = max(self.reward,0)
            self.reward = self.reward
        return self.reward, self._episode_done
    def _get_done(self,current_x_distance,current_y_distance,current_angle):
        if (current_x_distance > self.x_max_distance) or (abs(current_y_distance) > self.y_max_distance):
            done = True
            self.info['status'] = 0 # out of max distance
        elif current_x_distance < self.x_min_distance:
            done = True
            self.info['status'] = 1 # out of min distance 
        elif abs(current_angle) > self.max_angle:
            done = True
            self.info['status'] = 2 # out of max angle
        elif self.steps > self.max_step:
            done = True
            self.info['status'] = 3 # finished max steps
        else:
            done = False
            self.info['status'] = 4 # running
        return done
    def _post_information(self):
        """
        Return:
            info: {"system status"}
        """
        rospy.logdebug("\nStart Posting Information")
        self.info["status"] = 4
        rospy.logdebug("End Posting Information\n")

        return self.info
    def pub_robot2_cmd(self):
        global b        
        if self.robot2_cmd_pub_num % self.pub_leader_rate == 0:
            # x = np.random.uniform(self.low_bound[0],self.high_bound[0]-0.03)
            # y = np.random.uniform(self.low_bound[1],self.high_bound[1]-0.03)
            # z = np.random.uniform(self.low_bound[2],self.high_bound[2]-0.03)

            x = np.random.uniform(self.robot2_low_bound[0],self.robot2_high_bound[0])
            y = np.random.uniform(self.robot2_low_bound[1],self.robot2_high_bound[1])
            z = np.random.uniform(self.robot2_low_bound[2],self.robot2_high_bound[2])

            
            b = np.random.choice(self.behivor)
            if b == 0:
                self.cmd.linear.x = x
                self.cmd.linear.y = 0
                self.cmd.angular.z = 0
                # print(self.robot2_cmd_pub_num,'x直行')
            elif b == 1:
                self.cmd.linear.x = 0
                self.cmd.linear.y = y
                self.cmd.angular.z = 0
                # print(self.robot2_cmd_pub_num,'y直行')
            elif b == 2:
                self.cmd.linear.x = x
                self.cmd.linear.y = y
                self.cmd.angular.z = 0
                # print(self.robot2_cmd_pub_num,'斜行')
            elif b == 3:
                self.cmd.linear.x = 0
                self.cmd.linear.y = 0
                self.cmd.angular.z = z
                # print(self.robot2_cmd_pub_num,'旋转')
            else:
                self.cmd.linear.x = x
                self.cmd.linear.y = y
                self.cmd.angular.z = z
                # print(self.robot2_cmd_pub_num,'混合')
            self.robot2_cmd_pub_num = 0
        # else:
        #     if b == 0:
        #         print(self.robot2_cmd_pub_num,'x直行')
        #     elif b == 1:
        #         print(self.robot2_cmd_pub_num,'y直行')
        #     elif b == 2:
        #         print(self.robot2_cmd_pub_num,'斜行')
        #     elif b == 3:
                
        #         print(self.robot2_cmd_pub_num,'旋转')
        #     else:
        #         print(self.robot2_cmd_pub_num,'混合')
        self.robot2_cmd_pub_num += 1
        return self.cmd
    def pub_robot2_cir_cmd(self):
            x = np.random.uniform(self.robot2_low_bound[0],self.robot2_high_bound[0])
            y = np.random.uniform(self.robot2_low_bound[1],self.robot2_high_bound[1])
            z = np.random.uniform(self.robot2_low_bound[2],self.robot2_high_bound[2])
            z = 0.1
            r = 1.5
            self.cmd.linear.x = z*r
            self.cmd.linear.y = 0.0
            self.cmd.angular.z = z
            return self.cmd
    def pub_robot2_str_cmd(self):
            # x = np.random.uniform(self.robot2_low_bound[0],self.robot2_high_bound[0])
            # y = np.random.uniform(self.robot2_low_bound[1],self.robot2_high_bound[1])
            # z = np.random.uniform(self.robot2_low_bound[2],self.robot2_high_bound[2])
            # z = 0.1
            # r = 1.5
            self.cmd.linear.x = 0.15
            self.cmd.linear.y = 0.0
            self.cmd.angular.z = 0.0
            return self.cmd
    def _robot_matrix_transform(self,robot_position_data,name_id):
        id_robot = robot_position_data.name.index(name_id)
        robot_pose = robot_position_data.pose[id_robot]
        robot_twist = robot_position_data.twist[id_robot]
        robot_r = Rotation.from_quat([robot_pose.orientation.w,robot_pose.orientation.x,robot_pose.orientation.y,robot_pose.orientation.z])
        robot_r_m = robot_r.as_matrix()
        robot_r_e = robot_r.as_euler('xyz',True)             
        robot_matrix = np.zeros((4,4))
        robot_matrix[:3,:3] = robot_r_m
        robot_matrix[:,-1] = np.array([robot_pose.position.x,robot_pose.position.y,robot_pose.position.z,1])
        # print(robot_matrix,robot_r_e[0])
        return robot_matrix,robot_twist
    def _relative_pose(self,robot_position_data,name_id1,name_id2):
        robot1_matrix,robot1_twist = self._robot_matrix_transform(robot_position_data,name_id1)
        robot2_matrix,robot2_twist = self._robot_matrix_transform(robot_position_data,name_id2)
        robot_relative_pose = np.matmul(robot2_matrix,np.linalg.inv(robot1_matrix))
        robot_relative_r = robot_relative_pose[:3,:3]
        robot_relative_p = robot_relative_pose[:3,-1]
        robot_relative_r_r = Rotation.from_matrix(robot_relative_r)
        robot_relative_euler = robot_relative_r_r.as_euler('xyz',False)
        # print(robot_relative_euler)
        return robot_relative_p,robot_relative_euler[0],robot1_twist,robot2_twist
    
    def reward_distance(self, current_x_distance,current_y_distance, direction_error, dis_exp=None):
        #  reward = (100.0 / max(dis2target_now,100)) * np.cos(direction_error/360.0*np.pi)
        a = 1
        if dis_exp is None:
            dis_exp = self.exp_x_distance
        direction_error_normal = abs(direction_error)/self.max_angle_err
        current_x_distance_err = abs(dis_exp - current_x_distance)
        #  e_dis_relative = e_dis / self.dis_exp
        current_x_distance_err_normal = current_x_distance_err / self.x_max_distance_err
        current_y_distance_err = abs(current_y_distance)
        current_y_distance_err_normal = current_y_distance_err / self.y_max_distance_err
        # reward = 1 - min(e_dis_relative, 1) - min(direction_error, 1)
        reward = a - (direction_error_normal + current_x_distance_err_normal + current_y_distance_err_normal)/3.0
        reward = max(reward, -a)
        return reward
    def _map_action(self,model_output_act_list):
        mapped_action_list = []
        for model_output_act in model_output_act_list:
            model_output_act = np.array(model_output_act)
            mapped_action = self.low_bound + (model_output_act - (-1.0)) * (
                (self.high_bound - self.low_bound) / 2.0)
            mapped_action = np.clip(mapped_action, self.low_bound, self.high_bound)
            mapped_action_list.append(mapped_action)
        return mapped_action_list

    

    def _map_obs(self,orig_obs_list):

        if self.noise:
            position_noise = np.random.normal(0,self.pos_noise,[3])
            robot2_noise = np.random.normal(0,0.0,[3])
            robot1_noise = np.random.normal(0,0.0,[3])
            orig_obs = copy.deepcopy(orig_obs)
            orig_obs['pose']['x'] += position_noise[0]
            orig_obs['pose']['y'] += position_noise[1]
            orig_obs['pose']['yaw'] += position_noise[2]
            orig_obs['robot1']['v_x'] += robot1_noise[0]
            orig_obs['robot1']['v_y'] += robot1_noise[1]
            orig_obs['robot1']['v_ang'] += robot1_noise[2]
            orig_obs['relative']['v_x'] += robot2_noise[0]
            orig_obs['relative']['v_y'] += robot2_noise[1]
            orig_obs['relative']['v_ang'] += robot2_noise[2]
            



        # x_err = math.sqrt(orig_obs['pose']['x']**2 + orig_obs['pose']['y']**2)
        obs = []
        mapped_act_list = []
        for orig_obs,exp_x_distance,x_max_distance_err,exp_y_distance,y_max_distance_err in zip(orig_obs_list,self.exp_x_distance,self.x_max_distance_err,self.exp_y_distance,self.y_max_distance_err):
            x_err = orig_obs['pose']['x'] - exp_x_distance
            print('x_err',x_err)
            x_err_normal = x_err/x_max_distance_err
            y_err = orig_obs['pose']['y']-exp_y_distance
            print('y_err',y_err)
            y_err_normal = y_err/y_max_distance_err
            angle_err_normal = orig_obs['pose']['yaw']/self.max_angle_err
            act = np.array([orig_obs['robot1']['v_x'],orig_obs['robot1']['v_y'],orig_obs['robot1']['v_ang'],orig_obs['relative']['v_x'],orig_obs['relative']['v_y'],orig_obs['relative']['v_ang']])
            mapped_act = self._map_to_out_act(act)
            mapped_act_list.append(mapped_act[:3])


            pos = np.array([x_err_normal,y_err_normal,angle_err_normal])
            self.dt_time = rospy.Time.now() - self.current_time
            self.current_time = rospy.Time.now()
            self.dt_time = [self.dt_time.to_sec()]
            
            # print(self.dt_time)

            if self.obs_with_velocity:
                if self.obs_with_time:
                    obs.append(np.concatenate([pos,mapped_act,self.dt_time],-1))
                else:
                    if self.obs_with_relative_velocity:
                        obs.append(np.concatenate([pos,mapped_act],-1))
                    else:
                        obs.append(np.concatenate([pos,mapped_act[:3]],-1))
            else:
                if self.obs_with_time:
                    obs.append(np.concatenate([pos,self.dt_time],-1))
                else:
                    obs.append(pos)
        return np.array(self.point_cloud_list),np.array(mapped_act_list)
    def _map_to_out_act(self,obs_act):
        obs_act = np.array(obs_act)
        mapped_act = -1 + 2*(obs_act-np.concatenate([self.low_bound,self.robot2_low_bound],-1))/(np.concatenate([self.high_bound,self.robot2_high_bound],-1)-np.concatenate([self.low_bound,self.robot2_low_bound],-1))
        return mapped_act
    def point_cloud_callback(self,data):

        
        
        # self.random_point = np.zeros((self.point_size,3))
        xyzin = pc2.read_points(data,field_names=('x','y','z'),skip_nans=True)  
        xyzin = np.array(list(xyzin))
        
        indices = np.where(xyzin[:,-1]>-0.5)[0]
        new = xyzin[indices,:]

        indices_z2 = np.where(new[:,-1]<0.2)[0]
        new = new[indices_z2,:]

        indices_y = np.where(abs(new[:,-2])<0.6)[0]
        new = new[indices_y,:]
        indices_x = np.where(abs(new[:,-3])<3.0)[0]
        new = new[indices_x,:]
        indices_x = np.where(new[:,-3]>0.0)[0]
        new = new[indices_x,:]
        # print(new.shape)

        
        # new_in = np.random.randint(0,new.shape[0],size=(768))
        new_in = np.random.choice(range(new.shape[0]),size=(768),replace=True)
        # new_in = np.random.choice(self.list_sel,size=(768))
        new2 = new[new_in]
        # self.final_point = new2
        return new2
    


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
