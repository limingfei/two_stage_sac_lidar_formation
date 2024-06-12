#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import numpy as np
if __name__ == '__main__':
    rospy.init_node('test_node')
    pub = rospy.Publisher('/robot1_cmd_vel',Twist,queue_size=1)
    cmd = Twist()
    rate = rospy.Rate(10)
    behivor = [0,1,2,3,4] # 0 x直行，1 y直行，2 斜行， 3 旋转，4 混合
    pub_num = 0
    while True:
        if pub_num % 100 == 0:
            x = np.random.uniform(0,0.2)
            y = np.random.uniform(0,0.2)
            z = np.random.uniform(0,0.2)
            b = np.random.choice(behivor)
            if b == 0:
                cmd.linear.x = x
                cmd.linear.y = 0
                cmd.angular.z = 0
                print(pub_num,'x直行')
            elif b == 1:
                cmd.linear.x = 0
                cmd.linear.y = y
                cmd.angular.z = 0
                print(pub_num,'y直行')
            elif b == 2:
                cmd.linear.x = x
                cmd.linear.y = y
                cmd.angular.z = 0
                print(pub_num,'斜行')
            elif b == 3:
                cmd.linear.x = 0
                cmd.linear.y = 0
                cmd.angular.z = z
                print(pub_num,'旋转')
            else:
                cmd.linear.x = x
                cmd.linear.y = y
                cmd.angular.z = z
                print(pub_num,'混合')
            pub_num = 0
        else:
            if b == 0:
                print(pub_num,'x直行')
            elif b == 1:
                print(pub_num,'y直行')
            elif b == 2:
                print(pub_num,'斜行')
            elif b == 3:
                
                print(pub_num,'旋转')
            else:
                print(pub_num,'混合')
        pub_num += 1

        pub.publish(cmd)
        rate.sleep()

