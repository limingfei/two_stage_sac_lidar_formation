import rospy
from gazebo_msgs.srv import GetModelState,GetModelStateRequest
import tf2_ros
import tf
from geometry_msgs.msg import TransformStamped

def pub_tf(child,get_robot_requset):
    robot_response = client.call(get_robot_requset)
    if robot_response.success:
        x = robot_response.pose.position.x
        y = robot_response.pose.position.y
        z = robot_response.pose.position.z
        oren = robot_response.pose.orientation

        broadcaster = tf2_ros.TransformBroadcaster()
    #         4-2.创建 广播的数据(通过 pose 设置)
        tfs = TransformStamped()
        tfs.header.frame_id = "world"
        # tfs.header.stamp = rospy.Time.now()
        tfs.child_frame_id = child
        tfs.transform.translation.x = x
        tfs.transform.translation.y = y
        tfs.transform.translation.z = z
        tfs.transform.rotation = oren
        #         4-3.广播器发布数据
        broadcaster.sendTransform(tfs)
        print('pub tf sucessful !')




client = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
rospy.init_node('get')
get_robot1_request = GetModelStateRequest()
get_robot1_request.model_name = 'mycar1'
get_robot2_request = GetModelStateRequest()
get_robot2_request.model_name = 'mycar2'
while not rospy.is_shutdown():
    pub_tf('robot1_base_link',get_robot1_request)
    pub_tf('robot2_base_link',get_robot2_request)
        

