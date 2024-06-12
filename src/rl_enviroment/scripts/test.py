import torch
import copy
import tf2_ros
import tf
from nav_msgs.msg import Odometry
import torch.nn as nn
from model import GazeboModel
import yaml
import numpy as np
import rospy
import os
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import message_filters
from tensorboardX import SummaryWriter

class TestModel(object):
    def __init__(self,config_path):
        with open(config_path,'r') as file:
            self.parms = yaml.safe_load(file)
        self.model = torch.load(self.parms.model_params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.exp_x_distance = self.parms['exp_x_distance']
        self.x_max_distance = self.parms['x_max_distance']
        self.x_min_distance = self.parms['x_min_distance']
        self.x_max_distance_err = self.x_max_distance - self.exp_x_distance
        self.y_max_distance = self.parms['y_max_distance']
        self.y_max_distance_err = self.y_max_distance
        self.max_angle = self.parms['max_angle']
        self.max_angle_err = self.max_angle
        self.max_step = self.parms['max_step']
        
        self.low_bound = np.array(self.parms['low_bound'])
        self.robot2_low_bound = np.array(self.parms['robot2_low_bound'])
        self.high_bound = np.array(self.parms['high_bound'])
        self.robot2_high_bound = np.array(self.parms['robot2_high_bound'])

        self.leader_sub_topic = self.parms['leader_sub_topic']
        self.follower_sub_topic = self.parms['follower_sub_topic']
        self.leader_pub_topic = self.parms['leader_pub_topic']
        self.follower_pub_topic = self.parms['follower_pub_topic']
        self.force_topic = self.parms['force_topic']
        self.leader_link = self.parms['leader_link']
        self.follower_link = self.parms['follower_link']
        self.behivor = [0,1,2,3,4]  # 0 x直行，1 y直行，2 斜行， 3 旋转，4 混合
        self.robot2_cmd = Twist()
        self.robot2_cmd_pub_num = 0
        np.random.seed(1)

        log_path = './logs_realworld'
        root_log_path = os.path.join(log_path,'speed_{}'.format(self.parms['high_bound'][0]))
        self.writer = SummaryWriter(root_log_path)



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
            force=dict(
            x=0.0,
            y=0.0,
            z=0.0,
            x_w=0.0,
            y_w=0.0,
            z_w=0.0
            ))
        self.robot1_cmd_vel_pub = rospy.Publisher(self.follower_pub_topic,Twist,queue_size=10)
        self.robot2_cmd_vel_pub = rospy.Publisher(self.leader_pub_topic,Twist,queue_size=10)
        
        
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)
        

    def step(self):
        obs = self.get_observation() 
        obs = self.maped_obs(obs)
        obs = torch.tensor(obs.reshape(1,-1),dtype=torch.float32).to(self.device)
        
        act_mean, _ = self.model.policy(obs)
        action = torch.tanh(act_mean)
        action_numpy = action.cpu().detach().numpy().flatten()
        self._take_action(action_numpy)

    
    def maped_obs(self,orig_obs):
        x_err = orig_obs['pose']['x'] - self.exp_x_distance
        x_err_normal = x_err/self.x_max_distance_err
        y_err = orig_obs['pose']['y']
        y_err_normal = y_err/self.y_max_distance_err
        angle_err_normal = orig_obs['pose']['yaw']/self.max_angle_err
        act = np.array([orig_obs['robot1']['v_x'],orig_obs['robot1']['v_y'],orig_obs['robot1']['v_ang'],orig_obs['relative']['v_x'],orig_obs['relative']['v_y'],orig_obs['relative']['v_ang']])
        mapped_act = self._map_to_out_act(act)


        pos = np.array([x_err_normal,y_err_normal,angle_err_normal])       
        # print(self.dt_time)          
        obs = np.concatenate([pos,mapped_act],-1)        
        return obs
    def _map_to_out_act(self,obs_act):
        obs_act = np.array(obs_act)
        mapped_act = -1 + 2*(obs_act-np.concatenate([self.low_bound,self.robot2_low_bound],-1))/(np.concatenate([self.high_bound,self.robot2_high_bound],-1)-np.concatenate([self.low_bound,self.robot2_low_bound],-1))
        return mapped_act
    def _map_action(self,model_output_act):
        model_output_act = np.array(model_output_act)
        mapped_action = self.low_bound + (model_output_act - (-1.0)) * (
            (self.high_bound - self.low_bound) / 2.0)
        mapped_action = np.clip(mapped_action, self.low_bound, self.high_bound)
        return mapped_action
    def _take_action(self, action_1):
        """
        Set linear and angular speed for logger_0 and logger_1 to execute.
        Args:
            action: 2x np.array([v_lin,v_ang]).
        """
        action_1 = self._map_action(action_1)
        cmd_vel_1 = Twist()
        cmd_vel_1.linear.x = action_1[0]
        cmd_vel_1.linear.y = action_1[1]
        cmd_vel_1.angular.z = action_1[2]
        cmd_vel_2 = self.get_robot2_cmd()
         
        for _ in range(10):
            self.robot1_cmd_vel_pub.publish(cmd_vel_1)
            self.robot2_cmd_vel_pub.publish(cmd_vel_2)
    def get_observation(self):        
        self.get_relative_pose()
        rospy.Subscriber(self.follower_sub_topic,Odometry,self.callback1)
        rospy.Subscriber(self.leader_sub_topic,Odometry,self.callback2)
        rospy.Subscriber(self.force_topic,Float32MultiArray,self.callback3)
        obs = copy.deepcopy(self.observation)

        return obs
        
    def get_relative_pose(self):
        trans = self.buffer.lookup_transform('agv1/base_link', 'agv2/base_link', rospy.Time(0))
        self.observation['pose']['x'] = trans.transform.translation.x
        self.observation['pose']['y'] = trans.transform.translation.y
        rotation = np.array([trans.transform.rotation.x,trans.transform.rotation.y,trans.transform.rotation.z,trans.transform.rotation.w],dtype=float)
        quaternion = tf.transformations.euler_from_quaternion(rotation)
        self.observation['pose']['yaw'] = quaternion[2]
    def get_robot2_cmd(self):       
        if self.robot2_cmd_pub_num % 10 == 0:
            x = np.random.uniform(self.robot2_low_bound[0],self.robot2_high_bound[0])
            y = np.random.uniform(self.robot2_low_bound[1],self.robot2_high_bound[1])
            z = np.random.uniform(self.robot2_low_bound[2],self.robot2_high_bound[2])

            b = np.random.choice(self.behivor)
            if b == 0:
                self.robot2_cmd.linear.x = x
                self.robot2_cmd.linear.y = 0
                self.robot2_cmd.angular.z = 0
                # print(self.robot2_cmd_pub_num,'x直行')
            elif b == 1:
                self.robot2_cmd.linear.x = 0
                self.robot2_cmd.linear.y = y
                self.robot2_cmd.angular.z = 0
                # print(self.robot2_cmd_pub_num,'y直行')
            elif b == 2:
                self.robot2_cmd.linear.x = x
                self.robot2_cmd.linear.y = y
                self.robot2_cmd.angular.z = 0
                # print(self.robot2_cmd_pub_num,'斜行')
            elif b == 3:
                self.robot2_cmd.linear.x = 0
                self.robot2_cmd.linear.y = 0
                self.robot2_cmd.angular.z = z
                # print(self.robot2_cmd_pub_num,'旋转')
            else:
                self.robot2_cmd.linear.x = x
                self.robot2_cmd.linear.y = y
                self.robot2_cmd.angular.z = z
                # print(self.robot2_cmd_pub_num,'混合')
        self.robot2_cmd_pub_num += 1
        return self.robot2_cmd


    
    
    def callback1(self,data):
        self.observation['robot1']['v_x'] = data.twist.twist.linear.x
        self.observation['robot1']['v_y'] = data.twist.twist.linear.y
        self.observation['robot1']['v_ang'] = data.twist.twist.angular.z
    def callback2(self,data):
        v_x_2 = data.twist.twist.linear.x
        v_y_2 = data.twist.twist.linear.y
        v_ang_2 = data.twist.twist.angular.z
        yaw = self.observation['pose']['yaw']
        r_v_x = v_x_2*np.cos(yaw)-v_y_2*np.sin(yaw) - self.observation['robot1']['v_x']
        r_v_y = v_x_2*np.sin(yaw)+v_y_2*np.cos(yaw) - self.observation['robot1']['v_y']
        r_v_ang = v_ang_2 - self.observation['robot1']['v_ang']
        self.observation['relative']['v_x'] = r_v_x
        self.observation['relative']['v_y'] = r_v_y
        self.observation['relative']['v_ang'] = r_v_ang
    def callback3(self,data):
        force_x = data.data[0]
        force_y = data.data[1]
        force_z = data.data[2]
        force_x_w = data.data[3]
        force_y_w = data.data[4]
        force_z_w = data.data[5]
        self.observation['force']['x'] = force_x
        self.observation['force']['y'] = force_y
        self.observation['force']['z'] = force_z
        self.observation['force']['x_w'] = force_x_w
        self.observation['force']['y_w'] = force_y_w
        self.observation['force']['z_w'] = force_z_w
    def writer_data(self,step):
        self.writer.add_scalar('pose/x',self.observation['pose']['x'],step)
        self.writer.add_scalar('pose/y',self.observation['pose']['y'],step)
        self.writer.add_scalar('pose/yaw',self.observation['pose']['yaw'],step)
        self.writer.add_scalar('force/x',self.observation['force']['x'],step)
        self.writer.add_scalar('force/y',self.observation['force']['y'],step)
        self.writer.add_scalar('force/z',self.observation['force']['z'],step)
        self.writer.add_scalar('force/x_w',self.observation['force']['x_w'],step)
        self.writer.add_scalar('force/y_w',self.observation['force']['y_w'],step)
        self.writer.add_scalar('force/z_w',self.observation['force']['z_w'],step)


        




