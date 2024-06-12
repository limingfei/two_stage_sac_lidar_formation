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
from gazebo_msgs.srv import GetModelState,GetModelStateRequest
import tf
import yaml
from sensor_msgs.msg import PointCloud2,PointField
from sensor_msgs import point_cloud2 as pc2
from robot_leader_motion import LeaderMotion

# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
# import utils
# from utils import double_utils

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
        self.leader_motion = LeaderMotion()
        self.motion_done = True

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
            v_ang = 0.0),
            relative=dict(
            v_x = 0.0,
            v_y = 0.0,
            v_ang = 0.0),
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
        self.leader_motion = LeaderMotion()
        self.motion_done = True
        
        # self.resetWorld()
        time.sleep(0.1)
        
        orig_obs = self._get_observation()
        obs = self._map_obs(orig_obs)
        info = self._post_information()
        self.steps = 0
        # rospy.logwarn("\nEnd Environment Reset!!!\n")
        self.robot2_cmd_pub_num = 0
        self.robot2_cmd = Twist()
        self.cmd = Twist()

        return obs

    def step(self, action_1):
        """
        Manipulate logger_0 with action_0, logger_1 with action_1
        obs, rew, done, info = env.step(action_0, action_1)
        """
        
        rospy.logdebug("\nStart Environment Step")
        # time_noise = abs(np.random.normal(0,self.time_noise))
        
        # time.sleep(time_noise)
        self._take_action(action_1)

        orig_obs = self._get_observation()
        obs = self._map_obs(orig_obs)
        reward, done = self._compute_reward()
        info = self._post_information()
        self.steps += 1
        rospy.logdebug("End Environment Step\n")

        return obs, reward, done, info

    
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
        relative_response = self.client.call(self.get_relative_requset)
        self._relative_pos_cmd = self.client.call(self.get_relative_requset2)
        robot1_response = self.client.call(self.get_robot1_requset)
        robot2_response = self.client.call(self.get_robot2_request)
        # position,ang,robot1_twist,robot2_twist = self._relative_pose(self.robot_position_data,'mycar1','mycar2')
        if relative_response.success and robot1_response.success and robot2_response.success and self._relative_pos_cmd.success:
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
            relative_r = Rotation.from_quat([relative_response.pose.orientation.x,relative_response.pose.orientation.y,relative_response.pose.orientation.z,relative_response.pose.orientation.w])
            # relative_r = Rotation.from_quat([robot1_response.pose.orientation.x,robot1_response.pose.orientation.y,robot1_response.pose.orientation.z,robot1_response.pose.orientation.w])
            relative_euler = relative_r.as_euler('zyx')

            # relative_euler = tf.transformations.euler_from_quaternion(relative_response.pose.orientation)
            self.observation['pose']['yaw'] = relative_euler[0]
            self.observation['robot1']['v_x'] = robot1_response.twist.linear.x
            self.observation['robot1']['v_y'] = robot1_response.twist.linear.y
            self.observation['robot1']['v_ang'] = robot1_response.twist.angular.z
            self.observation['relative']['v_x'] = robot2_response.twist.linear.x
            self.observation['relative']['v_y'] = robot2_response.twist.linear.y
            self.observation['relative']['v_ang'] = robot2_response.twist.angular.z
            self.observation['point_cloud']= self.random_point
        else:
            rospy.logerr('get service error!!!!!!')
        
        # self.unpausePhysics()

        return self.observation

    def _take_action(self, action_1):
        """
        Set linear and angular speed for logger_0 and logger_1 to execute.
        Args:
            action: 2x np.array([v_lin,v_ang]).
        """
        rospy.logdebug("\nStart Taking Actions")
        action_1 = self._map_action(action_1)
        # print('action1',action_1)
        cmd_vel_1 = Twist()
        cmd_vel_1.linear.x = action_1[0]
        cmd_vel_1.linear.y = action_1[1]
        cmd_vel_1.angular.z = action_1[2]
        cmd_vel_2,self.leader_done = self.get_robot2_cmd()
        self.unpausePhysics()        
        for _ in range(10):
            self.robot1_cmd_vel_pub.publish(cmd_vel_1)
            self.robot2_cmd_vel_pub.publish(cmd_vel_2)
            self.rate.sleep()
        self.action_1 = action_1
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
        # elif self.steps > self.max_step:
        elif self.leader_done:
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
    def get_robot2_cmd(self):
        # global b
        # if self.robot2_cmd_pub_num % self.pub_leader_rate == 0:
        #     # print(np.random.choice(self.behivor))
        #     # x = np.random.uniform(self.low_bound[0],self.high_bound[0]-0.03)
        #     # y = np.random.uniform(self.low_bound[1],self.high_bound[1]-0.03)
        #     # z = np.random.uniform(self.low_bound[2],self.high_bound[2]-0.03)

        #     x = np.random.uniform(self.robot2_low_bound[0],self.robot2_high_bound[0])
        #     y = np.random.uniform(self.robot2_low_bound[1],self.robot2_high_bound[1])
        #     z = np.random.uniform(self.robot2_low_bound[2],self.robot2_high_bound[2])

            
        #     # b = np.random.choice(self.behivor)
        #     b = 0
        #     if b == 0:
        #         self.cmd.linear.x = x
        #         self.cmd.linear.y = 0
        #         self.cmd.angular.z = 0
        #         # print(self.robot2_cmd_pub_num,'x直行')
        #     elif b == 1:
        #         self.cmd.linear.x = 0
        #         self.cmd.linear.y = y
        #         self.cmd.angular.z = 0
        #         # print(self.robot2_cmd_pub_num,'y直行')
        #     elif b == 2:
        #         self.cmd.linear.x = x
        #         self.cmd.linear.y = y
        #         self.cmd.angular.z = 0
        #         # print(self.robot2_cmd_pub_num,'斜行')
        #     elif b == 3:
        #         self.cmd.linear.x = 0
        #         self.cmd.linear.y = 0
        #         self.cmd.angular.z = z
        #         # print(self.robot2_cmd_pub_num,'旋转')
        #     else:
        #         self.cmd.linear.x = x
        #         self.cmd.linear.y = y
        #         self.cmd.angular.z = z
        #         # print(self.robot2_cmd_pub_num,'混合')
        #     self.robot2_cmd_pub_num = 0
        # # else:
        # #     if b == 0:
        # #         print(self.robot2_cmd_pub_num,'x直行')
        # #     elif b == 1:
        # #         print(self.robot2_cmd_pub_num,'y直行')
        # #     elif b == 2:
        # #         print(self.robot2_cmd_pub_num,'斜行')
        # #     elif b == 3:
                
        # #         print(self.robot2_cmd_pub_num,'旋转')
        # #     else:
        # #         print(self.robot2_cmd_pub_num,'混合')
        # print(self.motion_done)
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

        self.final_point_normal = self.final_point/[0.755,0.201]
        
        # print(self.dt_time)
        if self.point_cloud:
            # obs = combine_state
            obs = np.reshape(self.final_point_normal,(-1))
        elif self.point_cloud_with_velocity:
            obs = [np.array(point_err_noraml).transpose(1,0),np.concatenate([pos,mapped_act],-1)]
        else:
            if self.obs_with_velocity:
                if self.obs_with_time:
                    obs = np.concatenate([pos,mapped_act,self.dt_time],-1)
                else:
                    if self.obs_with_relative_velocity:
                        obs = np.concatenate([pos,mapped_act],-1)
                    else:
                        obs = np.concatenate([pos,mapped_act[:3]],-1)
            else:
                if self.obs_with_time:
                    obs = np.concatenate([pos,self.dt_time],-1)
                else:
                    obs = pos
        return obs
    def _map_to_out_act(self,obs_act):
        obs_act = np.array(obs_act)
        mapped_act = -1 + 2*(obs_act-np.concatenate([self.low_bound,self.robot2_low_bound],-1))/(np.concatenate([self.high_bound,self.robot2_high_bound],-1)-np.concatenate([self.low_bound,self.robot2_low_bound],-1))
        return mapped_act
    
    def point_cloud_callback(self,data):
        
        self.random_point = np.zeros((self.point_size,3))
        xyzin = pc2.read_points(data,field_names=('x','y','z','intensity'),skip_nans=True)  
        xyzin = np.array(list(xyzin))
        indices = np.where(xyzin[:,-1]>100)[0]
        new = xyzin[indices,:-1]

        self.dbscan.fit(new)
        center = []
        for label in set(self.dbscan.labels_):
            if label != -1:
                indice = self.dbscan.labels_ == label
                center.append(new[indice])

        center_mean_0 = np.array(center[0]).mean(0)[1]
        center_mean_1 = np.array(center[0]).mean(0)[1]
        if center_mean_0 > center_mean_1:
            center_left = np.array(center[0][:,:-1])
            center_right = np.array(center[1][:,:-1])
        else:
            center_left = np.array(center[1][:,:-1])
            center_right = np.array(center[0][:,:-1])
        np.random.shuffle(center_left)
        np.random.shuffle(center_right)
        random_left_point = np.zeros((80,2))
        random_right_point = np.zeros((80,2))
        random_left_point[:min(80,center_left.shape[0])] = center_left[:min(80,center_left.shape[0])]
        random_right_point[:min(80,center_right.shape[0])] = center_right[:min(80,center_right.shape[0])]
        sorted_left_indices = np.lexsort((random_left_point[:, -1],))
        left_point = random_left_point[sorted_left_indices]
        sorted_right_indices = np.lexsort((random_right_point[:, -1],))
        right_point = random_right_point[sorted_right_indices]
        self.final_point = np.concatenate([left_point,right_point],0)
        
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
