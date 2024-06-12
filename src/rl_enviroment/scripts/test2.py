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
import yaml

def square_distance(pcd1, pcd2):
    """
    Squared distance between any two points in the two point clouds.
    """
    return torch.sum((pcd1[:, None, :].contiguous() - pcd2[None, :, :].contiguous()) ** 2, dim=-1)


def pub_point(data,points):
    header = data.header
    header.stamp = rospy.Time.now()
    print(header.frame_id)
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)
                  
                  ]
    
    pc2_msg = pc2.create_cloud(header,fields,points)
    pub.publish(pc2_msg)
    # print('pub')
def callback(data):
    # global num
    if data is None:
        rospy.logerr('none none none none')



        
        
    xyzin = pc2.read_points(data,field_names=('x','y','z','intensity'),skip_nans=True)  
    xyzin = np.array(list(xyzin))[:,:-1]
    
    indices = np.where(xyzin[:,-1]>-0.5)[0]
    new = xyzin[indices,:]

    # indices_z2 = np.where(new[:,-1]<0.2)[0]
    # new = new[indices_z2,:]

    # indices_y = np.where(abs(new[:,-2])<0.6)[0]
    # new = new[indices_y,:]
    # indices_x = np.where(abs(new[:,-3])<3.0)[0]
    # new = new[indices_x,:]
    # indices_x = np.where(new[:,-3]>0.0)[0]
    # new = new[indices_x,:]



    print(new.shape)
    pub_point(data,new)


    # # new_in = np.random.randint(0,new.shape[0],size=(768))
    # new_in = np.random.choice(range(new.shape[0]),size=(768),replace=True)
    # # new_in = np.random.choice(self.list_sel,size=(768))
    # new2 = new[new_in]
    # print(new2.mean(0))
    # # self.final_point = new2


    #1.29659563
    #0.76414822
    
   
    
    # print('left_point:{},\nrandom:{}'.format(left_point,random_left_point))

    # print(count,count2)





    # print(random_point)
    # trans1 = buffer.lookup_transform('robot1_velodyne', 'robot2_left_reflector_panel', rospy.Time(0))
    # translation1 = np.array([trans1.transform.translation.x,trans1.transform.translation.y,trans1.transform.translation.z],dtype=float)
    # trans2 = buffer.lookup_transform('robot1_velodyne', 'robot2_right_reflector_panel', rospy.Time(0))
    # translation2 = np.array([trans2.transform.translation.x,trans2.transform.translation.y,trans2.transform.translation.z],dtype=float)


    # print('x  :{:.2f},y  :{:.2f},z  :{:.2f}'.format(new[:,0].mean(),abs(new[:,1]).max(),new[:,2].min()))
    # print('x_t1:{:.2f},y_t1:{:.2f},z_t1:{:.2f}'.format(translation1[0],translation1[1],translation1[2]))
    # print('x_t2:{:.2f},y_t2:{:.2f},z_t2:{:.2f}'.format(translation2[0],translation2[1],translation2[2]))

    # print(new.shape)



    
    # print(xyzin[:,-1].max())
if __name__ == '__main__':
    rospy.init_node('testwerew')
    num = 1
    sub = rospy.Subscriber('/robot1_velodyne_points',PointCloud2,callback)
    pub = rospy.Publisher('/refine_point_cloud',PointCloud2,queue_size=10)
    dbscan = DBSCAN(eps=0.2,min_samples=10)
   
    # buffer = tf2_ros.Buffer()
    # listener = tf2_ros.TransformListener(buffer)
   
    # print('wai',num)
    rospy.spin()