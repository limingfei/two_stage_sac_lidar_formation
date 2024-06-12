#!/usr/bin/env python
import rospy
import tf
import tf2_ros
from gazebo_msgs.srv import GetModelState,GetModelStateRequest
from sensor_msgs.msg import PointCloud2,PointField
from sensor_msgs import point_cloud2 as pc2
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from geometry_msgs.msg import TransformStamped
import open3d
def get_angle(center):
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
            left = (x0,y0,z0)
        else:
            left = (x1,y1,z1)
    else:
        rospy.logerr('center is more than 2!!!!')
    return angle,left

def pub_tf(angle,left):
    # print(angle,left)
    tfs.header.frame_id = "robot1_velodyne"
    tfs.child_frame_id = 'robot2_left_reflector_panel'
    tfs.transform.translation.x = left[0]
    tfs.transform.translation.y = left[1]
    tfs.transform.translation.z = left[2]
    print('angle',angle)
    oren = tf.transformations.quaternion_from_euler(0,0,angle)
    tfs.transform.rotation.x = oren[0]
    tfs.transform.rotation.y = oren[1]
    tfs.transform.rotation.z = oren[2]
    tfs.transform.rotation.w = oren[3]
    # tfs.header.stamp = rospy.Time.now()
    broadcaster.sendTransform(tfs)

    tfs.header.frame_id = "robot2_left_reflector_panel"
    tfs.child_frame_id = 'robot2_base_link'
    tfs.transform.translation.x = 0.4050
    tfs.transform.translation.y = -0.20
    tfs.transform.translation.z = -0.3800
    tfs.transform.rotation.x = 0
    tfs.transform.rotation.y = 0
    tfs.transform.rotation.z = 0
    tfs.transform.rotation.w = 1
    # tfs.header.stamp = rospy.Time.now()
    broadcaster.sendTransform(tfs)

def pub_point(data,points):
    header = data.header
    header.stamp = rospy.Time.now()
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('intensity', 12, PointField.FLOAT32, 1)]
    
    pc2_msg = pc2.create_cloud(header,fields,points)
    pub.publish(pc2_msg)
def callback(data):
    # global num
    if data is None:
        rospy.logerr('none none none none')
    xyzin = pc2.read_points(data,field_names=('x','y','z','intensity'),skip_nans=True)  
    xyzin = np.array(list(xyzin))
    indices = np.where(xyzin[:,-1]>100)[0]
    new = xyzin[indices,:-1] 
    dbscan.fit(new)
    center = []
    for label in set(dbscan.labels_):
        if label != -1:
            indice = dbscan.labels_ == label
            center.append([new[indice].mean(0)[0],new[indice].mean(0)[1],new[indice].mean(0)[2],255])
    zero = np.zeros((new.shape[0],1))
    new_i = np.concatenate([new,zero],axis=-1)
    # print(np.array(center).shape)
    out = np.concatenate([new_i,center],axis=0)
    angle,left = get_angle(center)
    pub_tf(angle,left)
    trans1 = buffer.lookup_transform('robot1_base_link', 'robot2_base_link', rospy.Time(0))
    translation1 = np.array([trans1.transform.translation.x,trans1.transform.translation.y,trans1.transform.translation.z],dtype=float)
    # trans2 = buffer.lookup_transform('robot1_velodyne', 'robot2_right_reflector_panel', rospy.Time(0))
    # translation2 = np.array([trans2.transform.translation.x,trans2.transform.translation.y,trans2.transform.translation.z],dtype=float)
    rotation = np.array([trans1.transform.rotation.x,trans1.transform.rotation.y,trans1.transform.rotation.z,trans1.transform.rotation.w],dtype=float)
    quaternion = np.zeros(3)
    quaternion = tf.transformations.euler_from_quaternion(rotation)

    # print('x  :{:.2f},y  :{:.2f},z  :{:.2f}'.format(new[:,0].mean(),abs(new[:,1]).max(),new[:,2].min()))
    print('x_t1:{:.4f},y_t1:{:.4f},z_t1:{:.4f}'.format(quaternion [0],quaternion [1],quaternion [2]))
    # print('x_t2:{:.2f},y_t2:{:.2f},z_t2:{:.2f}'.format(translation2[0],translation2[1],translation2[2]))

    # print(new.shape)
    pub_point(data,out)

    
    # print(xyzin[:,-1].max())
if __name__ == '__main__':
    rospy.init_node('test')
    num = 1
    sub = rospy.Subscriber('/robot1_velodyne_points',PointCloud2,callback)
    pub = rospy.Publisher('/refine_point_cloud',PointCloud2,queue_size=10)
    dbscan = DBSCAN(eps=0.2,min_samples=10)
    broadcaster = tf2_ros.TransformBroadcaster()
    tfs = TransformStamped()
    
    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer)
   
    # print('wai',num)
    rospy.spin()