#!/usr/bin/env python
"""
Task environment for two loggers escaping from the walled cell, cooperatively.
"""
from sklearn.cluster import DBSCAN
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
from gazebo_msgs.srv import GetModelState,GetModelStateRequest,GetWorldProperties
import tf
import yaml
from sensor_msgs.msg import PointCloud2,PointField
from sensor_msgs import point_cloud2 as pc2
from robot_leader_motion import LeaderMotion,Motion

import pandas as pd


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
        self.exp_x_distance = self.parms['exp_x_distance']
        self.x_max_distance = self.parms['x_max_distance']
        self.x_min_distance = self.parms['x_min_distance']
        self.pub_leader_rate = self.parms['pub_leader_rate']
        self.x_max_distance_err = self.x_max_distance - self.exp_x_distance
        self.y_max_distance = self.parms['y_max_distance']
        self.y_max_distance_err = self.y_max_distance
        self.max_angle = self.parms['max_angle']
        self.max_angle_err = self.max_angle
        self.eval = False

        self.exp_x_lidar_distance = self.exp_x_distance - 0.74
        self.x_lidar_max_distance_err = self.x_max_distance_err
        self.exp_y_lidar_distance = 0.0 
        self.y_lidar_max_distance_err = 1
        self.dbscan = DBSCAN(eps=0.2,min_samples=10)

        self.point_cloud_with_velocity = self.parms['point_cloud_with_velocity']
        self.point_cloud = self.parms['point_cloud']
        self.low_bound = np.array(self.parms['low_bound'])
        self.robot2_low_bound = np.array(self.parms['robot2_low_bound'])
        self.high_bound = np.array(self.parms['high_bound'])
        self.robot2_high_bound = np.array(self.parms['robot2_high_bound'])
        self.rate = rospy.Rate(100)
        self.noise = self.parms['noise']
        self.time_noise = self.parms['time_noise']
        self.pos_noise = self.parms['pos_noise']
        self.obs_with_relative_velocity = self.parms['obs_with_relative_velocity']
        self.point_size = 180
        # if self.parms['set_random_seed']:
        # np.random.seed(1)

        # init environment parameters
        self.observation = dict(
            pose = dict(x=0.0,
            y=0.0,
            yaw=0.0),
            robot1 = dict(
            v_x = 0.0,
            v_y = 0.0,
            v_ang = 0.0,
            x = 0.0,
            y = 0.0,
            yaw = 0.0),
            relative=dict(
            v_x = 0.0,
            v_y = 0.0,
            v_ang = 0.0),
            robot2 = dict(x=0.0,
            y=0.0,
            yaw=0.0),
            point_cloud = np.zeros((self.point_size,3)))
        self.action_1 = np.zeros(2)
        self.info = dict(status="")
        self.reward = 0
        self._episode_done = False
        self.success_count = 0
        self.max_step = self.parms['max_step']
        self.steps = 0
        self.status = ['trapped', 'trapped']
        self.behivor = [0,1,2,3,4]  # 0 x直行，1 y直行，2 斜行， 3 旋转，4 混合
        # self.final_point = np.random.rand(160,2)
        # self.random_point = np.zeros((self.point_size,3))
        # self.buffer = tf2_ros.Buffer()
        # self.listener = tf2_ros.TransformListener(self.buffer)
        self.robot_position_data = ModelStates()
        # services
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.unpause_physics_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state',SetModelState)
        # topic publisher
        self.robot1_cmd_vel_pub = rospy.Publisher("/robot1_cmd_vel", Twist, queue_size=1)
        self.robot2_cmd_vel_pub = rospy.Publisher("/robot2_cmd_vel", Twist, queue_size=1)
        self.get_relative_requset = GetModelStateRequest()
        self.get_relative_requset2 = GetModelStateRequest()
        self.get_relative_requset.model_name = 'mycar2'
        self.get_relative_requset2.model_name = 'mycar1'
        self.get_relative_requset.relative_entity_name = 'mycar1'
        self.get_relative_requset2.relative_entity_name = 'mycar2'

        self.get_robot1_requset = GetModelStateRequest()
        self.get_robot1_requset.model_name = 'mycar1'
        self.get_robot2_request = GetModelStateRequest()
        self.get_robot2_request.model_name = 'mycar2'
        self.client = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
        self.time_client = rospy.ServiceProxy('/gazebo/get_world_properties',GetWorldProperties)
        # self.cmds = pd.read_csv('test_many_finetune_room_chan/leader_follower_trajectory/follower_trajectory_rate_50_r_605_n_3.csv',header=None).to_numpy()
        # circle shaped
        # self.positions = pd.read_csv('test_many_finetune_room_chan/leader_follower_trajectory/follower_xyz_rate_50_r_-325_n_11.csv',header=None).to_numpy()
        # u shaped
        self.positions = pd.read_csv('test_u_shaped_finetune_room_chan/leader_follower_trajectory/follower2.csv',header=None).to_numpy()

        
        # self.velodyne = rospy.Subscriber('/robot1_velodyne_points',PointCloud2,self.point_cloud_callback) 
        # self.current_time = rospy.Time.now()
        
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

    def random_set_follower(self):
        model_state = ModelState()
        model_state.model_name = 'mycar1'
        model_state.reference_frame = 'mycar2'
        reqs = self.client.call(self.get_robot2_request)
        
        leader_x = reqs.pose.position.x 
        leader_y = reqs.pose.position.y
        leader_z = reqs.pose.position.z
        leader_r = Rotation.from_quat([reqs.pose.orientation.x,reqs.pose.orientation.y,reqs.pose.orientation.z,reqs.pose.orientation.w])
        leader_r_euler = leader_r.as_euler('xyz')
        model_state.pose.position.x = np.random.uniform(-1.5-0.2,-1.5+0.2)
        model_state.pose.position.y = np.random.uniform(-0.2,0.2)
        model_state.pose.position.z = leader_z
        leader_r_z1  = np.random.uniform(-0.2,-0.1)
        leader_r_z2  = np.random.uniform(0.1,0.2)
        leader_r_z = np.random.choice((leader_r_z1,leader_r_z2))
        leader_r_q = Rotation.from_euler('xyz',(0,0,leader_r_z))
        leader_r_q_q = leader_r_q.as_quat()

        model_state.pose.orientation.x = leader_r_q_q[0]
        model_state.pose.orientation.y = leader_r_q_q[1]
        model_state.pose.orientation.z = leader_r_q_q[2]
        model_state.pose.orientation.w = leader_r_q_q[3]

        # x = np.random.uniform(self.low_bound[0],self.high_bound[0])
        # y = np.random.uniform(self.low_bound[1],self.high_bound[1])
        # z = np.random.uniform(self.low_bound[2],self.high_bound[2])

        # model_state.twist.linear.x = x
        # model_state.twist.linear.y = y
        # model_state.twist.angular.z = z
        self.set_model_state_proxy(model_state)

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
        self.leader_cmd = np.empty((0,3))
        self.follower_cmd = np.empty((0,3))
        self.follower_pos = np.empty((0,6))
        self.leader_pos = np.empty((0,6))
        # self.resetWorld()
        self.leader_motion = LeaderMotion()
        self.motion = Motion()
        self.motion_done = True
        time.sleep(0.1)

        
        
        
        orig_obs = self._get_observation()
        
        obs,vel,high_feature = self._map_obs(orig_obs)
        info = self._post_information()
        self.steps = 0
        # rospy.logwarn("\nEnd Environment Reset!!!\n")
        self.robot2_cmd_pub_num = 0
        self.robot2_cmd = Twist()
        self.cmd = Twist()
        self.cmd_leader = Twist()
        self.pre_vel = vel
        # self.reset_model_state()
        # self.unpausePhysics()
        # time.sleep(0.1)
        # orig_obs2 = self._get_observation()
        
        # src,vel2,high_feature2 = self._map_obs(orig_obs)


        return obs,vel,high_feature

    def step(self, action_1):
        """
        Manipulate logger_0 with action_0, logger_1 with action_1
        obs, rew, done, info = env.step(action_0, action_1)
        """
        
        rospy.logdebug("\nStart Environment Step")
        # time_noise = abs(np.random.normal(0,self.time_noise))
        # if self.steps % 25 == 0:
        #     self.random_set_follower()
        

        # time.sleep(time_noise)
        self._take_action(action_1)

        orig_obs = self._get_observation()
        obs,vel,high_feature = self._map_obs(orig_obs)
        self.back_vel = vel
        reward, done = self._compute_reward()
        self.pre_vel = self.back_vel
        info = self._post_information()
        self.steps += 1
        rospy.logdebug("End Environment Step\n")        

        return (obs,vel,high_feature), reward, done, info

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
        try:
            msg = rospy.wait_for_message('/robot1_velodyne_points',PointCloud2, timeout=20.0)
        except rospy.exceptions.ROSException as e:
            self.resetSimulation()
            msg = rospy.wait_for_message('/robot1_velodyne_points',PointCloud2, timeout=20.0)
            rospy.logerr('wait for message timeout!')
        self.pausePhysics()
        
        self.point_cloud_callback(msg)
        
                
        
        # time.sleep(0.05)
        self.client.wait_for_service()
        self.time_client.wait_for_service()
        relative_response = self.client.call(self.get_relative_requset)
        self._relative_pos_cmd = self.client.call(self.get_relative_requset2)
        robot1_response = self.client.call(self.get_robot1_requset)
        robot2_response = self.client.call(self.get_robot2_request)
        time_response = self.time_client.call()
        # position,ang,robot1_twist,robot2_twist = self._relative_pose(self.robot_position_data,'mycar1','mycar2')
        if time_response.success and relative_response.success and robot1_response.success and robot2_response.success and self._relative_pos_cmd.success:
            # self.observation['pose']['x'] = position[0]
            # self.observation['pose']['y'] = position[1]
            # self.observation['pose']['yaw'] = ang
            # self.observation['robot1']['v_x'] = robot1_twist.linear.x
            # self.observation['robot1']['v_y'] = robot1_twist.linear.y
            # self.observation['robot1']['v_ang'] = robot1_twist.angular.z
            # self.observation['robot2']['v_x'] = robot2_twist.linear.x
            # self.observation['robot2']['v_y'] = robot2_twist.linear.y
            # self.observation['robot2']['v_ang'] = robot2_twist.angular.z

            self.observation['pose']['x'] = relative_response.pose.position.x
            self.observation['pose']['y'] = relative_response.pose.position.y
            self.sim_time = time_response.sim_time
            relative_r = Rotation.from_quat([relative_response.pose.orientation.x,relative_response.pose.orientation.y,relative_response.pose.orientation.z,relative_response.pose.orientation.w])
            robot1_r = Rotation.from_quat([robot1_response.pose.orientation.x,robot1_response.pose.orientation.y,robot1_response.pose.orientation.z,robot1_response.pose.orientation.w])
            robot2_r = Rotation.from_quat([robot2_response.pose.orientation.x,robot2_response.pose.orientation.y,robot2_response.pose.orientation.z,robot2_response.pose.orientation.w])
            # relative_r = Rotation.from_quat([robot1_response.pose.orientation.x,robot1_response.pose.orientation.y,robot1_response.pose.orientation.z,robot1_response.pose.orientation.w])
            relative_euler = relative_r.as_euler('zyx')
            robot1_euler = robot1_r.as_euler('zyx')
            robot2_euler = robot2_r.as_euler('zyx')

            # relative_euler = tf.transformations.euler_from_quaternion(relative_response.pose.orientation)
            self.observation['pose']['yaw'] = relative_euler[0]
            self.observation['robot1']['v_x'] = robot1_response.twist.linear.x
            self.observation['robot1']['v_y'] = robot1_response.twist.linear.y
            self.observation['robot1']['v_ang'] = robot1_response.twist.angular.z
            self.observation['robot1']['x'] = robot1_response.pose.position.x
            self.observation['robot1']['y'] = robot1_response.pose.position.y
            self.observation['robot1']['yaw'] = robot1_euler[0]
            self.observation['relative']['v_x'] = robot2_response.twist.linear.x
            self.observation['relative']['v_y'] = robot2_response.twist.linear.y
            self.observation['relative']['v_ang'] = robot2_response.twist.angular.z
            self.observation['robot2']['x'] = robot2_response.pose.position.x
            self.observation['robot2']['y'] = robot2_response.pose.position.y
            self.observation['robot2']['yaw'] = robot2_euler[0]
            self.observation['point_cloud']= self.random_point
            leader_pos = np.array([[robot2_response.pose.position.x,robot2_response.pose.position.y,robot2_response.pose.orientation.x,robot2_response.pose.orientation.y,robot2_response.pose.orientation.z,robot2_response.pose.orientation.w]])
            follower_pos = np.array([[robot1_response.pose.position.x,robot1_response.pose.position.y,robot1_response.pose.orientation.x,robot1_response.pose.orientation.y,robot1_response.pose.orientation.z,robot1_response.pose.orientation.w]])
            self.leader_pos = np.concatenate((self.leader_pos,leader_pos))
            self.follower_pos = np.concatenate((self.follower_pos,follower_pos))
        else:
            rospy.logerr('get service error!!!!!!')
        
        # self.unpausePhysics()

        return self.observation
    def u_shaped(self):
        cmd_vel = Twist()
        if self.steps < 20:
            x = np.random.uniform(0,0.03)
        elif self.steps < 40:
            x = np.random.uniform(0.03,0.06)
        elif self.steps < 60:
            x = np.random.uniform(0.06,0.09)
        elif self.steps < 80:
            x = np.random.uniform(0.09,0.12)
        elif self.steps < 100:
            x = np.random.uniform(0.12,0.15)
        else:
            x = np.random.uniform(0.15,0.2)
        print(self.observation['robot2']['x'] )
        if -0.00001 <= self.observation['robot2']['x'] < 3.5:
            cmd_vel.linear.x = x
            self.k = 0
        else:
            self.k+=1
            
            if 0<self.observation['robot2']['yaw'] <= 2.97:             
                cmd_vel.linear.x = x
                self.m = 0
                if self.k < 20:
                    z = np.random.uniform(0,0.05)
                elif self.k < 40:
                    z = np.random.uniform(0.05,0.1)
                else:
                    z = x/1.5
                cmd_vel.angular.z = z
            elif 0<self.observation['robot2']['yaw'] <= 3.14:
                    
                    self.m+=1
                    cmd_vel.linear.x = x
                    if self.m<20:
                        z = np.random.uniform(0.05,0.1)
                    elif self.m < 40:
                        z = np.random.uniform(0,0.05)
                    cmd_vel.angular.z = z
            else:
                if self.observation['robot2']['x'] > 0:
                    cmd_vel.linear.x = x
                    self.n = 0
                else:
                    self.n += 1
                    if self.observation['robot2']['yaw'] < 0.0:
                        cmd_vel.linear.x = x                        
                        if self.n < 20:
                            z = np.random.uniform(0,0.05)
                        elif self.n < 40:
                            z = np.random.uniform(0.05,0.1)
                        else:
                            z = x/1.5
                        cmd_vel.angular.z = z
                    else: 
                        cmd_vel.linear.x = x
        return cmd_vel
    
    def u_shaped2(self):
        
        if -0.00001 <= self.observation['robot2']['x'] < 3.5:
            self.cmd_leader.angular.z = 0.0
            self.cmd_leader.linear.x =  self.motion.acc_x(self.cmd_leader.linear.x,0.15)
            
            self.k = 0
        else:
            self.k+=1            
            if 0<self.observation['robot2']['yaw'] <= 2.97:             
                self.m = 0
                self.cmd_leader.angular.z =  self.motion.acc_yaw(self.cmd_leader.angular.z,0.1)
            elif 0<self.observation['robot2']['yaw'] <= 3.14:
                    self.m+=1
                    self.cmd_leader.angular.z =  self.motion.dec_yaw(self.cmd_leader.angular.z,0.0)
            else:
                
                if self.observation['robot2']['x'] > 0:
                    self.cmd_leader.angular.z = 0.0
                    self.n = 0
                else:
                    self.n += 1
                    if self.observation['robot2']['yaw'] < 0.0:
                        if self.observation['robot2']['yaw']<-0.27: 
                            self.cmd_leader.angular.z =  self.motion.acc_yaw(self.cmd_leader.angular.z,0.1)
                        else:
                            self.cmd_leader.angular.z =  self.motion.dec_yaw(self.cmd_leader.angular.z,0.0)
                    else: 
                        self.cmd_leader.angular.z = 0.0


        

            
            
    def _take_action(self, action_1):
        """
        Set linear and angular speed for logger_0 and logger_1 to execute.
        Args:
            action: 2x np.array([v_lin,v_ang]).
        """
        rospy.logdebug("\nStart Taking Actions")
        action_1 = self._map_action(action_1)
        # print('ang',action_1[2])
        cmd_vel_1 = Twist()
        cmd_vel_1.linear.x = action_1[0]
        cmd_vel_1.linear.y = action_1[1]
        cmd_vel_1.angular.z = action_1[2]
        cmd_vel_2 = self.pub_robot2_cmd()
        # index = min(self.steps,198)
        # if self.steps <= 150:
        #     cmd_vel_2.linear.x = 0.15
        #     cmd_vel_2.linear.y = 0.0
        #     cmd_vel_2.angular.z = 0.0
        #     self.k = 0
        # else:
        #     print('step:{},yaw:{}'.format(self.steps,self.observation['robot2']['yaw']))
        #     if (self.k ==0) and (self.observation['robot2']['yaw'] <= 3.14) and (self.observation['robot2']['yaw'] > 0.0): 
        #         cmd_vel_2.linear.x = 0.15
        #         cmd_vel_2.linear.y = 0.0
        #         cmd_vel_2.angular.z = 0.1
        #     elif self.k <= 500:
        #         self.k += 1
        #         cmd_vel_2.linear.x = 0.15
        #         cmd_vel_2.linear.y = 0.0
        #         cmd_vel_2.angular.z = 0.0
        #     else:
        #         cmd_vel_2.linear.x = 0.15
        #         cmd_vel_2.linear.y = 0.0
        #         cmd_vel_2.angular.z = 0.1

        # x,y,z = self.cmds[self.steps]
        # cmd_vel_2.linear.x = 0.0
        # cmd_vel_2.linear.y = 0.0
        # cmd_vel_2.angular.z = 0.0
        # cmd_vel_2 = self.u_shaped()
        # self.u_shaped2()
        # x = self.cmd
        x = np.random.uniform(-0.08,0.08)
        r = np.random.uniform(-8,8)
        radians = math.radians(r)
        self.cmd_leader.linear.x = x
        self.cmd_leader.angular.z = radians
        

        


        cmd_vel = np.array([[cmd_vel_2.linear.x,cmd_vel_2.linear.y,cmd_vel_2.angular.z]])
        follower_cmd_vel = np.array([[cmd_vel_1.linear.x,cmd_vel_1.linear.y,cmd_vel_1.angular.z]])
        self.leader_cmd = np.concatenate((self.leader_cmd,cmd_vel))
        self.follower_cmd = np.concatenate((self.follower_cmd,follower_cmd_vel))
        self.unpausePhysics()        
        for _ in range(10):
            self.robot1_cmd_vel_pub.publish(cmd_vel_1)
            self.robot2_cmd_vel_pub.publish(self.cmd_leader)
            self.rate.sleep()
        self.action_1 = action_1
        rospy.logdebug("\nrobot_1 take action ===> {}".format(cmd_vel_1))
        rospy.logdebug("End Taking Actions\n")
        # x,y,ox,oy,oz,ow = self.positions[self.steps]
        # self.reset_model_state(x,y,ox,oy,oz,ow)

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
            self.reward = -100
        elif self.info['status'] == 3:
            self.reward == 100
        else:
            # self.reward = max(self.reward,0)
            self.reward = self.reward
        
        return self.reward, self._episode_done
    def _get_done(self,current_x_distance,current_y_distance,current_angle):
        if (current_x_distance > self.x_max_distance) or (abs(current_y_distance) > self.y_max_distance):
            done = True
            self.info['status'] = 0 # out of max distance
            print('out of max distance')
        elif current_x_distance < self.x_min_distance:
            done = True
            self.info['status'] = 1 # out of min distance 
            print('out of min distance')

        elif abs(current_angle) > self.max_angle:
            done = True
            self.info['status'] = 2 # out of max angle
            print('out of max angle')

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
            x = np.random.uniform(self.robot2_low_bound[0],self.robot2_high_bound[0])
            y = np.random.uniform(self.robot2_low_bound[1],self.robot2_high_bound[1])
            z = np.random.uniform(self.robot2_low_bound[2],self.robot2_high_bound[2])            
            b = np.random.choice(self.behivor)
            # b = 0
            # print(b)
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
        self.robot2_cmd_pub_num += 1


        
        




        return self.cmd
    
    def _leader_motion_smooth(self):
        if self.motion_done:
            self.mode = np.random.choice(['vertical','forward lateral','reverse lateral','forward rotation','reverse rotation'])
            self.motion_done = False
        if self.mode == 'vertical':
            self.cmd,self.motion_done = self.leader_motion.x_motion(self.cmd.linear.x)
        elif self.mode == 'forward lateral':
            self.cmd,self.motion_done = self.leader_motion.y_motion(self.cmd.linear.y)
        elif self.mode == 'reverse lateral':
            self.cmd,self.motion_done = self.leader_motion.y_motion(self.cmd.linear.y,-0.2)
        elif self.mode == 'forward rotation':
            self.cmd,self.motion_done = self.leader_motion.ang_motion(self.cmd.angular.z)
        else:
            self.cmd,self.motion_done = self.leader_motion.ang_motion(self.cmd.angular.z,-0.1)
        self.robot2_cmd_pub_num += 1
        return self.cmd,self.motion_done
    def _leader_motion_random(self):
        global b        
        if self.robot2_cmd_pub_num % self.pub_leader_rate == 0:
            x = np.random.uniform(self.robot2_low_bound[0],self.robot2_high_bound[0])
            y = np.random.uniform(self.robot2_low_bound[1],self.robot2_high_bound[1])
            z = np.random.uniform(self.robot2_low_bound[2],self.robot2_high_bound[2])            
            b = np.random.choice(self.behivor)
            # b = 0
            print(b)
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
        self.robot2_cmd_pub_num += 1
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

       
        # angle_list = [0.05,0.1,0.15]
        # x_list = [0.01,0.025,0.05]
        # y_list = [0.01,0.025,0.05]

        angle_list = [0.05,0.1,0.15,0.2,0.25]
        x_list = [0.01,0.05,0.1,0.2,0.4]
        y_list = [0.01,0.05,0.1,0.2,0.4]


        def insert_sort(value_list,value):
            list_np = np.array(value_list)
            index = np.searchsorted(list_np, value)
            return index

        # angle_index = insert_sort(angle_list,abs(direction_error))
        # x_index = insert_sort(x_list,current_x_distance_err)
        # y_index = insert_sort(y_list,current_y_distance_err)

        # if angle_index == 0 and x_index == 0 and y_index ==0:
        #     reward = 1
        # elif angle_index <= 1 and y_index <= 1 and x_index <= 1:
        #     reward = 0
        # else:
        #     reward = -1

        angle_index = insert_sort(angle_list,abs(direction_error))
        x_index = insert_sort(x_list,current_x_distance_err)
        y_index = insert_sort(y_list,current_y_distance_err)

        if angle_index == 0 and x_index == 0 and y_index ==0:
            reward = 1
        elif angle_index <= 1 and y_index <= 1 and x_index <= 1:
            reward = 0.6
        elif angle_index <= 2 and y_index <= 2 and x_index <= 2:
            reward = 0.3
        elif angle_index <= 3 and y_index <= 3 and x_index <= 3:
            reward = 0.1
        elif angle_index <= 4 and y_index <= 4 and x_index <= 4:
            reward = 0.0
        else:
            reward = -1

        result = self.back_vel*self.pre_vel
        self.zhen_dang_reward = np.where(result >= 0, 0, -np.abs(self.back_vel - self.pre_vel)).sum()        
        # reward = a - (direction_error_normal + current_x_distance_err_normal + current_y_distance_err_normal)/3.0
        # reward = max(reward, -a)
        reward = reward + 0.1*self.zhen_dang_reward
        return reward
    def _map_action(self,model_output_act):
        model_output_act = np.array(model_output_act)
        mapped_action = self.low_bound + (model_output_act - (-1.0)) * (
            (self.high_bound - self.low_bound) / 2.0)
        mapped_action = np.clip(mapped_action, self.low_bound, self.high_bound)
        return mapped_action

    

    def _map_obs(self,orig_obs):

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
            



        #x_err = math.sqrt(orig_obs['pose']['x']**2 + orig_obs['pose']['y']**2)
        x_err = orig_obs['pose']['x'] - self.exp_x_distance
        x_err_normal = x_err/self.x_max_distance_err
        y_err = orig_obs['pose']['y']
        y_err_normal = y_err/self.y_max_distance_err
        angle_err_normal = orig_obs['pose']['yaw']/self.max_angle_err

       
        point_err = orig_obs['point_cloud'][:,:-1] - [self.exp_x_lidar_distance,self.exp_y_lidar_distance]
        # point_err_noraml = point_err/[self.x_lidar_max_distance_err,self.y_lidar_max_distance_err]
        # point_err_noraml = orig_obs['point_cloud'][:,:-1]

        point_err_noraml = orig_obs['point_cloud'][:,:-1]




        act = np.array([orig_obs['robot1']['v_x'],orig_obs['robot1']['v_y'],orig_obs['robot1']['v_ang'],orig_obs['relative']['v_x'],orig_obs['relative']['v_y'],orig_obs['relative']['v_ang']])
        mapped_act = self._map_to_out_act(act)


        pos = np.array([x_err_normal,y_err_normal,angle_err_normal])
        # self.dt_time = rospy.Time.now() - self.current_time
        # self.current_time = rospy.Time.now()
        # self.dt_time = [self.dt_time.to_sec()]
        point = np.array(point_err_noraml)
        point = np.reshape(point,(-1))
        combine_state = np.concatenate([pos,mapped_act,point],-1)

        # self.final_point_normal = self.final_point/[0.755,0.201]
        
        # print(self.dt_time)
        obs = self.final_point
        return obs,mapped_act[:3],np.concatenate([pos,mapped_act],axis=-1)
    def _map_to_out_act(self,obs_act):
        obs_act = np.array(obs_act)
        mapped_act = -1 + 2*(obs_act-np.concatenate([self.low_bound,self.robot2_low_bound],-1))/(np.concatenate([self.high_bound,self.robot2_high_bound],-1)-np.concatenate([self.low_bound,self.robot2_low_bound],-1))
        return mapped_act
    
    def point_cloud_callback(self,data):

        
        
        self.random_point = np.zeros((self.point_size,3))
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
        self.final_point = new2
        
        
    def get_angle(self,center):
        if len(center) == 2:
            x0 = center[0][0]
            y0 = center[0][1]
            z0 = center[0][2]
            x1 = center[1][0]
            y1 = center[1][1]
            z1 = center[1][2]
            if x0 == x1:
                angle = 0.0
            else:
                angle = np.arctan((x1-x0)/(y1-y0))
            if y0 > y1:            
                left = (x0,y0)
                right = (x1,y1)
            else:
                left = (x1,y1)
                right = (x0,y0)
        else:
            rospy.logerr('center is more than 2!!!!')
        return left,right


    def reset_model_state(self,x,y,o_x,o_y,o_z,o_w):
        model_state = ModelState()
        model_state.model_name = 'mycar2'
        # model_state.reference_frame = 'mycar2'
        # if self.eval:
        #     x = np.random.uniform(-2.0,-2.2)
        #     y1 = np.random.uniform(-0.2,-0.1)
        #     y2 = np.random.uniform(0.1,0.2)
        #     ang1 = np.random.uniform(-0.15,-0.1)
        #     ang2 = np.random.uniform(0.1,0.15)
        #     y = np.random.choice([y1,y2])
        #     ang = np.random.choice([ang1,ang2])
        # else:
        #     x = np.random.uniform(-1.5,-2.2)
        #     y = np.random.uniform(-0.2,0.2)
        #     ang = np.random.uniform(-0.15,0.15)

        model_state.pose.position.x = x
        model_state.pose.position.y = y
        # rotation = Rotation.from_euler('xyz',[0,0,ang])
        # quaterion = rotation.as_quat()
        
        # model_state.pose.orientation.x = quaterion[0]
        # model_state.pose.orientation.y = quaterion[1]
        # model_state.pose.orientation.z = quaterion[2]
        # model_state.pose.orientation.w = quaterion[3]

        model_state.pose.orientation.x = o_x
        model_state.pose.orientation.y = o_y
        model_state.pose.orientation.z = o_z
        model_state.pose.orientation.w = o_w
        self.set_model_state_proxy(model_state)


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