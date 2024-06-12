import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import open3d as o3d
from scipy.spatial.transform import Rotation as scipyR
import numpy as np
from geometry_msgs.msg import Vector3,PoseArray,Pose
from tf.transformations import quaternion_from_euler
from test5 import to_quat
def publish_normals(points, normals):
    # 创建一个 ROS 发布器
    pub = rospy.Publisher('visualized_normals', PoseArray, queue_size=10)
    # rospy.init_node('normal_publisher', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    # 创建 Marker 消息
    marker = PoseArray()
    marker.header.frame_id = "robot1_velodyne"
    marker.header.stamp = rospy.Time.now()
    pose_array = []
    quats = normals
    points = np.mean(points,0,keepdims=True)
    quats = np.mean(quats,0,keepdims=True)
    for point,quat in zip(points,quats):
        r = scipyR.from_quat(quat)
        # elu = r.as_euler('xyz')
        # print('x:{},y:{},z:{}'.format(elu[0],elu[1],elu[2]))
        pose = Pose()
        pose.position.x = point[0]
        pose.position.y = point[1]
        pose.position.z = point[2]

        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]

        pose_array.append(pose)
    # print(len(pose_array))
    
    marker.poses = pose_array
    
    
    

        


        # 发布 Marker 消息
    pub.publish(marker)
        # rate.sleep()
    # print('pub')





def callback(pointcloud_msg):
    # 将 PointCloud2 消息转换为 Numpy 数组
    points = []
    for point in point_cloud2.read_points(pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append(point)
    points = np.array(points)

    # 创建 Open3D 点云对象
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)

    # 估计法向量
    pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))

    # 获取法向量
    normals = np.array(pointcloud.normals,dtype='float64')

    toe = np.mean(normals,0,keepdims=True)
    x_t,y_t,z_t = to_quat(toe)
    print('x;{},y:{},z:{}'.format(x_t,y_t,z_t))

    

    quaternions = []
    for norm in normals:
        rotation_matrices = o3d.geometry.get_rotation_matrix_from_xyz(np.reshape(norm,(3,1)))
        rotation = scipyR.from_matrix(rotation_matrices)
        quaternion = rotation.as_quat()
        quaternions.append(quaternion)




    publish_normals(np.array(pointcloud.points),quaternions)

    # o3d.visualization.draw_geometries([pointcloud], window_name="法线估计",
    #                               point_show_normal=True,
    #                               width=800,  # 窗口宽度
    #                               height=600)  # 窗口高度


    # 打印前10个点的法向量
    # for i in range(10):
    #     print("Point ", np.array(pointcloud.points)[i], " normal: ", np.linalg.norm(normals[i]))

def listener():
    rospy.init_node('listener343', anonymous=True)
    rospy.Subscriber('/refine_point_cloud', PointCloud2, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
