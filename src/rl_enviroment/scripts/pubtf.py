import rospy
import tf2_ros
import tf
from geometry_msgs.msg import TransformStamped
from gazebo_msgs.msg import ModelState, LinkState, ModelStates, LinkStates
from rl_enviroment.msg import RelativePose

if __name__ == "__main__":

    # 2.初始化 ROS 节点
    rospy.init_node("testetst")
    # 3.创建 TF 订阅对象
    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():

        try:
        # 4.调用 API 求出 son1 相对于 son2 的坐标关系
            #lookup_transform(self, target_frame, source_frame, time, timeout=rospy.Duration(0.0)):
            tfs = buffer.lookup_transform("robot1","robot2",rospy.Time(0))
    
            rospy.loginfo("robot1 与 robot2 相对关系:")
            rospy.loginfo("父级坐标系:%s",tfs.header.frame_id)
            rospy.loginfo("子级坐标系:%s",tfs.child_frame_id)
            rospy.loginfo("相对坐标:x=%.2f, y=%.2f, z=%.2f",
                        tfs.transform.translation.x,
                        tfs.transform.translation.y,
                        tfs.transform.translation.z,
            )

        except Exception as e:
            rospy.logerr("错误提示:%s",e)


        rate.sleep()