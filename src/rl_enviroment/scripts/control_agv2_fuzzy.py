#!/usr/bin/env python
import rospy
import tf
import tf2_ros
from geometry_msgs.msg import Twist
import numpy as np
from skfuzzy import control as f_ctrl


class FuzzyPidControl(object):
    def __init__(self,follower_high_bound,leader_high_bound,follower_low_bound,leader_low_bound):
        self.msg1=Twist()
        self.msg2=Twist()
        self.dt=0.1
        self.cmd_x_val = 0  
        self.cmd_y_val = 0
        self.cmd_r_val = 0

        self.cmd_x_last_last_last_err = 0
        self.cmd_x_last_last_err = 0
        self.cmd_x_last_err =0
        self.cmd_x_now_err = 0
        self.cmd_x_err_diff=0
        self.cmd_x_err_sum=0

        self.cmd_y_last_last_last_err = 0
        self.cmd_y_last_last_err = 0
        self.cmd_y_last_err = 0
        self.cmd_y_now_err = 0
        self.cmd_y_err_diff=0
        self.cmd_y_err_sum=0

        self.cmd_r_last_last_last_err = 0
        self.cmd_r_last_last_err = 0
        self.cmd_r_last_err = 0
        self.cmd_r_now_err = 0
        self.cmd_r_err_diff=0
        self.cmd_r_err_sum=0

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
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)
        
        self.fuzzy_PID_x= self._make_fuzzy_sim(max_Kp_add=0.1, max_Ki_add=0.001, max_Kd_add=0.0001, max_err=0.05, max_err_sum=0.01, max_err_diff=1.0, num=10)
        self.fuzzy_PID_y= self._make_fuzzy_sim(max_Kp_add=0.1, max_Ki_add=0.001, max_Kd_add=0.0001, max_err=0.05, max_err_sum=0.01, max_err_diff=3.0, num=10)
        self.fuzzy_PID_r= self._make_fuzzy_sim(max_Kp_add=0.05, max_Ki_add=0.001, max_Kd_add=0.0001, max_err=0.01, max_err_sum=0.005, max_err_diff=0.1, num=10)

        self.target_translation =np.array([-1.5,0,0])
        self.target_quaternion =np.array([0,0,0])

        self.follower_low_bound = follower_low_bound
        self.leader_low_bound = leader_low_bound
        self.follower_high_bound = follower_high_bound
        self.leader_high_bound = leader_high_bound

    def fuzzly_get_follower_cmd(self,relative_response,noise):

        
        # trans = self.buffer.lookup_transform('nexus1/base_link', 'nexus2/base_link', rospy.Time(0))
        # translation = np.array([trans.transform.translation.x,trans.transform.translation.y,trans.transform.translation.z],dtype=float)
        # rotation = np.array([trans.transform.rotation.x,trans.transform.rotation.y,trans.transform.rotation.z,trans.transform.rotation.w],dtype=float)
        # quaternion = np.zeros(3)
        # quaternion = tf.transformations.euler_from_quaternion(rotation)
        # time_now = trans.header.stamp
        translation = np.array([relative_response.pose.position.x,relative_response.pose.position.y,relative_response.pose.position.z],dtype=float)
        rotation = np.array([relative_response.pose.orientation.x,relative_response.pose.orientation.y,relative_response.pose.orientation.z,relative_response.pose.orientation.w],dtype=float)
        quaternion = np.zeros(3)
        quaternion = list(tf.transformations.euler_from_quaternion(rotation))
        time_now = relative_response.header.stamp
        # rospy.Subscriber('/robot2_cmd_vel',Twist,self.front_callback)

        self.cmd_x_last_last_last_err = self.cmd_r_last_last_err
        self.cmd_x_last_last_err = self.cmd_x_last_err
        self.cmd_x_last_err = self.cmd_x_now_err
        self.cmd_x_now_err = self.target_translation[0] - translation[0]
        self.cmd_x_err_diff = (self.cmd_x_now_err - self.cmd_x_last_err) / self.dt
        self.cmd_x_err_sum += self.cmd_x_now_err*self.dt

        self.fuzzy_PID_x.input['error'] = self.cmd_x_now_err
        self.fuzzy_PID_x.input['error_sum'] = self.cmd_x_err_sum
        self.fuzzy_PID_x.input['error_diff'] = self.cmd_x_err_diff
        

        self.fuzzy_PID_x.compute() 

        
        self.cmd_x_kp = np.clip(self.cmd_x_kp + self.fuzzy_PID_x.output['Kp_add'],0.5,0.9)
        self.cmd_x_ki = np.clip(self.cmd_x_ki + self.fuzzy_PID_x.output['Ki_add'],0.05,0.09)
        self.cmd_x_kd = np.clip(self.cmd_x_kd + self.fuzzy_PID_x.output['Kd_add'],-0.001,0.001)



        self.cmd_x_change_val = self.cmd_x_kp * (self.cmd_x_now_err - self.cmd_x_last_err) + self.cmd_x_ki * \
                self.cmd_x_now_err + self.cmd_x_kd * (self.cmd_x_now_err - 2 * self.cmd_x_last_err
                + self.cmd_x_last_last_err)
        self.cmd_x_val=self.cmd_x_val+self.cmd_x_change_val


        self.cmd_y_last_last_last_err = self.cmd_r_last_last_err
        self.cmd_y_last_last_err = self.cmd_y_last_err
        self.cmd_y_last_err = self.cmd_y_now_err
        self.cmd_y_now_err = self.target_translation[1] - translation[1]
        self.cmd_y_err_diff = (self.cmd_y_now_err - self.cmd_y_last_err) / self.dt
        self.cmd_y_err_sum += self.cmd_y_now_err*self.dt

        self.fuzzy_PID_y.input['error'] = self.cmd_y_now_err
        self.fuzzy_PID_y.input['error_sum'] = self.cmd_y_err_sum
        self.fuzzy_PID_y.input['error_diff'] = self.cmd_y_err_diff

        self.fuzzy_PID_y.compute() 

        self.cmd_y_kp = np.clip(self.cmd_y_kp + self.fuzzy_PID_y.output['Kp_add'],0.5,0.9)
        self.cmd_y_ki = np.clip(self.cmd_y_ki + self.fuzzy_PID_y.output['Ki_add'],0.05,0.09)
        self.cmd_y_kd = np.clip(self.cmd_y_kd + self.fuzzy_PID_y.output['Kd_add'],-0.001,0.001)
        
        self.cmd_y_change_val =self. cmd_y_kp * (self.cmd_y_now_err - self.cmd_y_last_err) + self.cmd_y_ki * \
                self.cmd_y_now_err + self.cmd_y_kd * (self.cmd_y_now_err - 2 * self.cmd_y_last_err
                + self.cmd_y_last_last_err)
        self.cmd_y_val=self.cmd_y_val+self.cmd_y_change_val


        self.cmd_r_last_last_last_err = self.cmd_r_last_last_err
        self.cmd_r_last_last_err = self.cmd_r_last_err
        self.cmd_r_last_err = self.cmd_r_now_err
        self.cmd_r_now_err = self.target_quaternion[2] - quaternion[2]
        self.cmd_r_err_diff = (self.cmd_r_now_err - self.cmd_r_last_err) / self.dt
        self.cmd_r_err_sum += self.cmd_r_now_err*self.dt

        self.fuzzy_PID_r.input['error'] = self.cmd_r_now_err
        self.fuzzy_PID_r.input['error_sum'] = self.cmd_r_err_sum
        self.fuzzy_PID_r.input['error_diff'] = self.cmd_r_err_diff

        self.fuzzy_PID_r.compute() 

        self.cmd_r_kp = np.clip(self.cmd_r_kp + self.fuzzy_PID_r.output['Kp_add'],0.4,0.8)
        self.cmd_r_ki = np.clip(self.cmd_r_ki + self.fuzzy_PID_r.output['Ki_add'],0.02,0.06)
        self.cmd_r_kd = np.clip(self.cmd_r_kd + self.fuzzy_PID_r.output['Kd_add'],-0.001,0.001)

        
        self.cmd_r_change_val = self.cmd_r_kp * (self.cmd_r_now_err - self.cmd_r_last_err) + self.cmd_r_ki * \
                self.cmd_r_now_err + self.cmd_r_kd * (self.cmd_r_now_err - 2 * self.cmd_r_last_err
                + self.cmd_r_last_last_err)
        self.cmd_r_val=self.cmd_r_val+self.cmd_r_change_val





        self.msg2.linear.x = self.msg2.linear.x + self.cmd_x_val
        self.msg2.linear.y = self.msg2.linear.y + self.cmd_y_val
        self.msg2.angular.z = self.msg2.angular.z + self.cmd_r_val




        
        # print("kp")
        # print(cmd_x_kp)
        # print(cmd_y_kp)
        # print(cmd_r_kp)

        # print("ki")
        # print(cmd_x_ki)
        # print(cmd_y_ki)
        # print(cmd_r_ki)
        
        # print("kd")
        # print(cmd_x_kd)
        # print(cmd_y_kd)
        # print(cmd_r_kd)            
        


        # self.talker2()

        msg_follower = Twist()
        msg_follower.linear.x = self.msg2.linear.x
        msg_follower.linear.y = self.msg2.linear.y
        msg_follower.angular.z = self.msg2.angular.z


        self.msg1.linear.x = 0
        self.msg1.linear.y = 0
        self.msg1.angular.z = 0
        self.msg2.linear.x = 0
        self.msg2.linear.y = 0
        self.msg2.angular.z = 0

        ori_action = np.array([msg_follower.linear.x,msg_follower.linear.y,msg_follower.angular.z])

        return self._map_to_out_act(ori_action)

       
    
    def _make_fuzzy_sim(self,max_Kp_add=0.01, max_Ki_add=0.001, max_Kd_add=0.0001, max_err=0.1, max_err_sum=0.01, max_err_diff=1.0, num=10):
        """ 生成模糊控制系统(1个dim) """
        # fuzzy input
        f_error = f_ctrl.Antecedent(np.linspace(-max_err, max_err, num), 'error')
        f_error_sum = f_ctrl.Antecedent(np.linspace(-max_err_sum, max_err_sum, num), 'error_sum')
        f_error_diff = f_ctrl.Antecedent(np.linspace(-max_err_diff, max_err_diff, num), 'error_diff')
        # fuzzy output
        f_Kp_add = f_ctrl.Consequent(np.linspace(-max_Kp_add, max_Kp_add, num), 'Kp_add')
        f_Ki_add = f_ctrl.Consequent(np.linspace(-max_Ki_add, max_Ki_add, num), 'Ki_add')
        f_Kd_add = f_ctrl.Consequent(np.linspace(-max_Kd_add, max_Kd_add, num), 'Kd_add')
        # 设置隶属度函数
        f_error.automf(3, names=['<0', '=0', '>0'])
        f_error_sum.automf(3, names=['<0', '=0', '>0'])
        f_error_diff.automf(3, names=['<0', '=0', '>0'])
        f_Kp_add.automf(3, names=['sub', 'keep', 'add'])
        f_Ki_add.automf(3, names=['sub', 'keep', 'add'])
        f_Kd_add.automf(3, names=['sub', 'keep', 'add'])
        # 设置模糊规则
        rules = [
            # Kp
            f_ctrl.Rule(f_error['<0'] | f_error['>0'], f_Kp_add['add']), # 误差过大, 增大快速性
            f_ctrl.Rule(f_error_diff['=0'], f_Kp_add['sub']),            # 阻尼较小, 减小快速性
            f_ctrl.Rule(f_error['=0'], f_Kp_add['keep']),
            # Ki
            f_ctrl.Rule(f_error['>0'] | f_error['<0'], f_Ki_add['add']),                         # 误差过大, 增大积分项
            f_ctrl.Rule(f_error['=0'] | f_error_sum['<0'] | f_error_sum['>0'], f_Ki_add['sub']), # 误差较小或积分较大, 减小积分项
            f_ctrl.Rule(f_error_sum['=0'], f_Ki_add['keep']),    
            # Kd
            f_ctrl.Rule((f_error['<0'] & f_error_diff['>0']) | (f_error['>0'] & f_error_diff['<0']), f_Kd_add['add']), # 靠近目标值, 增大阻尼
            f_ctrl.Rule((f_error['<0'] & f_error_diff['<0']) | (f_error['>0'] & f_error_diff['>0']), f_Kd_add['sub']), # 远离目标值, 减小阻尼
            f_ctrl.Rule(f_error['=0'] | f_error_diff['=0'], f_Kd_add['keep']),
        ]
        # 设置模糊推理系统
        fuzzy_sys = f_ctrl.ControlSystem(rules)
        fuzzy_sim = f_ctrl.ControlSystemSimulation(fuzzy_sys)
        return fuzzy_sim
    def talker1(self):
        
        pub = rospy.Publisher('nexus1/cmd_vel', Twist,queue_size=10)

        pub.publish(self.msg1)
    def talker2(self):
            
        pub = rospy.Publisher('nexus2/cmd_vel', Twist,queue_size=10)

        pub.publish(self.msg2)

    def _map_to_out_act(self,obs_act):
        obs_act = np.array(obs_act)
        mapped_act = -1 + 2*(obs_act-self.follower_low_bound)/(self.follower_high_bound-self.follower_low_bound)
        return mapped_act


    def front_callback(self,data):
        # global msg2
        # global msg1
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