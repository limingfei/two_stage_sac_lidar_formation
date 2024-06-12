#! /usr/bin/env python
import rospy
import message_filters
from std_srvs.srv import Empty,EmptyRequest
from geometry_msgs.msg import Twist,WrenchStamped
class AgvVelocity(object):
    def __init__(self):
        rospy.init_node('velocity_node',anonymous=True)
        self.robot1_velocity_pub = rospy.Publisher('/robot1_cmd_vel',Twist,queue_size=10)
        self.robot2_velocity_pub = rospy.Publisher('/robot2_cmd_vel',Twist,queue_size=10)
        self.reset_client = rospy.ServiceProxy('/gazebo/reset_simulation',Empty)
        self.robot1_velocity_msg = Twist()
        self.robot2_velocity_msg = Twist()
        self.a = 0
    def set_velocity(self,x1,y1,yaw1,x2,y2,yaw2):
        self.robot1_velocity_msg.linear.x = x1
        self.robot1_velocity_msg.linear.y = y1
        self.robot1_velocity_msg.angular.z = yaw1

        self.robot2_velocity_msg.linear.x = x2
        self.robot2_velocity_msg.linear.y = y2
        self.robot2_velocity_msg.angular.z = yaw2

        self.robot1_velocity_pub.publish(self.robot1_msg)
        self.robot2_velocity_pub.publish(self.robot2_msg)

    def reset_environments(self):
        self.reset_client.wait_for_service()
        req = EmptyRequest()
        rep = self.reset_client.call(req)
        rospy.loginfo('req:finish')

class AgvSub(object):
    def __init__(self):
        rospy.init_node('velocity_node',anonymous=True)
        self.robot1_force_sub = message_filters.Subscriber('/robot1_ft_sensor', WrenchStamped, queue_size=1)
        self.robot2_force_sub = message_filters.Subscriber('/robot2_ft_sensor', WrenchStamped, queue_size=1)
        self.a = 0
    def get_relative_position(self):
        sync_robot = message_filters.ApproximateTimeSynchronizer([self.robot1_force_sub, self.robot2_force_sub],10,0.1,allow_headerless=False)
        sync_robot.registerCallback(self.multi_callback)
        # rospy.spin()

    def multi_callback(self,robot1_force_data,robot2_force_data):
        print("finished!")
        self.a += 1


