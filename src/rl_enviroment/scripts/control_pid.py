#!/usr/bin/env python
import roslib
import rospy
import csv
import tf
import tf2_ros
import geometry_msgs.msg
# import tf2_geometry_msgs
from geometry_msgs.msg import Twist

import numpy as np
import math
from gazebo_msgs.srv import GetModelState,GetModelStateRequest

class PidControl(object):
    def __init__(self,params):
        self.msg1=Twist()
        self.msg2=Twist()
        self.cmd_x_val = 0  
        self.cmd_y_val = 0
        self.cmd_r_val = 0

        self.cmd_x_last_last_last_err = 0
        self.cmd_x_last_last_err = 0
        self.cmd_x_last_err =0
        self.cmd_x_now_err = 0
        self.cmd_y_last_last_last_err = 0
        self.cmd_y_last_last_err = 0
        self.cmd_y_last_err = 0
        self.cmd_y_now_err = 0
        self.cmd_r_last_last_last_err = 0
        self.cmd_r_last_last_err = 0
        self.cmd_r_last_err = 0
        self.cmd_r_now_err = 0
        self.time_difference_last_last = 0
        self.time_difference_last = 0
        self.time_difference = 0

        self.cmd_x_kp=0.7
        self.cmd_x_ki=0.07
        self.cmd_x_kd=0.00

        self.cmd_y_kp=0.7
        self.cmd_y_ki=0.07
        self.cmd_y_kd=0.0

        self.cmd_r_kp=0.6
        self.cmd_r_ki=0.04
        self.cmd_r_kd=0.0

        self.i = 0
        self.rate = rospy.Rate(10)
        # buffer = tf2_ros.Buffer()
        # listener = tf2_ros.TransformListener(buffer)

        self.client = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)

        self.get_relative_requset = GetModelStateRequest()
        self.get_relative_requset.model_name = 'mycar1'
        self.get_relative_requset.relative_entity_name = 'mycar2'
        
        self.follower_low_bound = np.array(params['low_bound'])
        self.leader_low_bound = np.array(params['robot2_low_bound'])
        self.follower_high_bound = np.array(params['high_bound'])
        self.leader_high_bound = np.array(params['robot2_high_bound'])

        self.target_translation =np.array([-1.5,0,0])
        self.target_quaternion =np.array([0,0,0])
    def pid_get_follower_cmd(self,relative_response):

       
        # relative_response = client.call(get_relative_requset)
        #trans = buffer.lookup_transform('robot1_base_footprint', 'robot2_base_footprint', rospy.Time(0))
        translation = np.array([relative_response.pose.position.x,relative_response.pose.position.y,relative_response.pose.position.z],dtype=float)
        rotation = np.array([relative_response.pose.orientation.x,relative_response.pose.orientation.y,relative_response.pose.orientation.z,relative_response.pose.orientation.w],dtype=float)
        quaternion = np.zeros(3)
        quaternion = tf.transformations.euler_from_quaternion(rotation)
        # print('ang',quaternion[2])
        time_now = relative_response.header.stamp
        rospy.Subscriber('/robot2_cmd_vel',Twist,self.front_callback)

        self.cmd_x_last_last_last_err = self.cmd_r_last_last_err
        self.cmd_x_last_last_err = self.cmd_x_last_err
        self.cmd_x_last_err = self.cmd_x_now_err
        self.cmd_x_now_err = self.target_translation[0] - translation[0]

        
        cmd_x_change_val = self.cmd_x_kp * (self.cmd_x_now_err - self.cmd_x_last_err) + self.cmd_x_ki * \
                self.cmd_x_now_err + self.cmd_x_kd * (self.cmd_x_now_err - 2 * self.cmd_x_last_err
                + self.cmd_x_last_last_err)
        self.cmd_x_val=self.cmd_x_val+cmd_x_change_val


        self.cmd_y_last_last_last_err = self.cmd_r_last_last_err
        self.cmd_y_last_last_err = self.cmd_y_last_err
        self.cmd_y_last_err = self.cmd_y_now_err
        self.cmd_y_now_err = self.target_translation[1] - translation[1]
        
        cmd_y_change_val = self.cmd_y_kp * (self.cmd_y_now_err - self.cmd_y_last_err) + self.cmd_y_ki * \
                self.cmd_y_now_err + self.cmd_y_kd * (self.cmd_y_now_err - 2 * self.cmd_y_last_err
                + self.cmd_y_last_last_err)
        self.cmd_y_val=self.cmd_y_val+cmd_y_change_val


        self.cmd_r_last_last_last_err = self.cmd_r_last_last_err
        self.cmd_r_last_last_err = self.cmd_r_last_err
        self.cmd_r_last_err = self.cmd_r_now_err
        self.cmd_r_now_err = self.target_quaternion[2] - quaternion[2]
        
        cmd_r_change_val = self.cmd_r_kp * (self.cmd_r_now_err - self.cmd_r_last_err) + self.cmd_r_ki * \
                self.cmd_r_now_err + self.cmd_r_kd * (self.cmd_r_now_err - 2 * self.cmd_r_last_err
                + self.cmd_r_last_last_err)
        self.cmd_r_val=self.cmd_r_val+cmd_r_change_val





        self.msg2.linear.x = self.msg2.linear.x + self.cmd_x_val
        self.msg2.linear.y = self.msg2.linear.y + self.cmd_y_val
        self.msg2.angular.z = self.msg2.angular.z + self.cmd_r_val
        # print('self.msg2.angular',self.msg2.angular.z)




        if np.linalg.norm(translation) < 1.2 :
            self.msg2.linear.x = 0
            self.msg2.linear.y = 0
            self.msg2.angular.z = 0

        if self.msg2.linear.x > self.follower_high_bound[0]:
            self.msg2.linear.x = self.follower_high_bound[0]

        if self.msg2.linear.y > self.follower_high_bound[1]:
            self.msg2.linear.y = self.follower_high_bound[1]

        if self.msg2.angular.z > self.follower_high_bound[2]:
            self.msg2.angular.z = self.follower_high_bound[2]

        if self.msg2.linear.x < self.follower_low_bound[0]:
            self.msg2.linear.x = self.follower_low_bound[0]

        if self.msg2.linear.y < self.follower_low_bound[1]:
            self.msg2.linear.y = self.follower_low_bound[1]

        if self.msg2.angular.z < self.follower_low_bound[2]:
            self.msg2.angular.z = self.follower_low_bound[2]

        if self.msg2.linear.x < 0.01 and self.msg2.linear.x > -0.01:
            self.msg2.linear.x = 0

        if self.msg2.linear.y < 0.01 and self.msg2.linear.y > -0.01:
            self.msg2.linear.y = 0

        if self.msg2.angular.z < 0.01 and self.msg2.angular.z > -0.01:
            self.msg2.angular.z = 0


        if self.msg2.linear.x > 1 or self.msg2.linear.x < -1:
            self.cmd_x_val = 0

        if self.msg2.linear.y > 1.5 or self.msg2.linear.y < -1.5:
            self.cmd_y_val = 0

        if self.msg2.angular.z > 0.5 or self.msg2.angular.z < -0.5:
            self.cmd_r_val = 0


        # talker2()
        msg_follower = Twist()
        msg_follower.linear.x = self.msg2.linear.x
        msg_follower.linear.y = self.msg2.linear.y
        msg_follower.angular.z = self.msg2.angular.z
        # print('follower***********',msg_follower.angular.z)
        

        self.msg1.linear.x = 0
        self.msg1.linear.y = 0
        self.msg1.angular.z = 0
        self.msg2.linear.x = 0
        self.msg2.linear.y = 0
        self.msg2.angular.z = 0
        ori_action = np.array([msg_follower.linear.x,msg_follower.linear.y,msg_follower.angular.z])
        # print('orig_act',ori_action)
        return self._map_to_out_act(ori_action)
    
    def front_callback(self,data):
   
        if data:
            self.msg1.linear.x = data.linear.x
            self.msg1.linear.y = data.linear.y
            self.msg1.angular.z = data.angular.z 
            self.msg2.linear.x = data.linear.x 
            self.msg2.linear.y = data.linear.y + data.angular.z * (self.target_translation[0])
            self.msg2.angular.z = data.angular.z 
        else :
            self.msg1.linear.x = 0
            self.msg1.linear.y = 0
            self.msg1.angular.z = 0
            self.msg2.linear.x = 0
            self.msg2.linear.y = 0
            self.msg2.angular.z = 0

    def talker1(self):
            
        pub = rospy.Publisher('/robot2_cmd_vel', Twist,queue_size=10)

        pub.publish(self.msg1)
    def talker2(self):
            
        pub = rospy.Publisher('/robot1_cmd_vel', Twist,queue_size=10)

        pub.publish(self.msg2)


    def _map_to_out_act(self,obs_act):
        obs_act = np.array(obs_act)
        mapped_act = -1 + 2*(obs_act-self.follower_low_bound)/(self.follower_high_bound-self.follower_low_bound)
        return mapped_act
