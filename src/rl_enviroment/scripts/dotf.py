#! /usr/bin/env python
"""  
    动态的坐标系相对姿态发布(一个坐标系相对于另一个坐标系的相对姿态是不断变动的)

    需求: 启动 turtlesim_node,该节点中窗体有一个世界坐标系(左下角为坐标系原点)，乌龟是另一个坐标系，键盘
    控制乌龟运动，将两个坐标系的相对位置动态发布

    实现分析:
        1.乌龟本身不但可以看作坐标系，也是世界坐标系中的一个坐标点
        2.订阅 turtle1/pose,可以获取乌龟在世界坐标系的 x坐标、y坐标、偏移量以及线速度和角速度
        3.将 pose 信息转换成 坐标系相对信息并发布
    实现流程:
        1.导包
        2.初始化 ROS 节点
        3.订阅 /turtle1/pose 话题消息
        4.回调函数处理
            4-1.创建 TF 广播器
            4-2.创建 广播的数据(通过 pose 设置)
            4-3.广播器发布数据
        5.spin
"""
# 1.导包
import rospy
import tf2_ros
import tf
from geometry_msgs.msg import TransformStamped
from gazebo_msgs.msg import ModelState, LinkState, ModelStates, LinkStates
from rl_enviroment.msg import RelativePose
from tfwarning import suppress_TF_REPEATED_DATA
def dotf(robot_position_data,child_frame_id,index_name):
    broadcaster = tf2_ros.TransformBroadcaster()
    #         4-2.创建 广播的数据(通过 pose 设置)
    tfs = TransformStamped()
    tfs.header.frame_id = "world"
    tfs.header.stamp = rospy.Time.now()
    tfs.child_frame_id = child_frame_id
    id_robot1 = robot_position_data.name.index(index_name)
    pose = robot_position_data.pose[id_robot1]
    tfs.transform.translation.x = pose.position.x
    tfs.transform.translation.y = pose.position.y
    tfs.transform.translation.z = 0.0
    # qtn = tf.transformations.quaternion_from_euler(0,0,pose.theta)
    tfs.transform.rotation = pose.orientation
    #         4-3.广播器发布数据
    broadcaster.sendTransform(tfs)


#     4.回调函数处理
def doPose(robot_position_data):
    #         4-1.创建 TF 广播器
    dotf(robot_position_data,'robot2','mycar2')
    dotf(robot_position_data,'robot1','mycar1')

if __name__ == "__main__":
    # 2.初始化 ROS 节点
    sutf = suppress_TF_REPEATED_DATA()
    rospy.init_node("dynamic_tf_do_p")
    # 3.订阅 /turtle1/pose 话题消息
    sub = rospy.Subscriber('/gazebo/model_states',ModelStates,doPose)
    buffer = tf2_ros.Buffer()
    #     4.回调函数处理
    #         4-1.创建 TF 广播器
    #         4-2.创建 广播的数据(通过 pose 设置)
    #         4-3.广播器发布数据
    #     5.spin
    listener = tf2_ros.TransformListener(buffer)
    # pub = rospy.Publisher('robot_relative_pose',RelativePose,queue_size=1)
    rate = rospy.Rate(10)
    rospy.spin()
