import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from sklearn.cluster import DBSCAN
import time
import random

class Test(object):
    def __init__(self):
        rospy.init_node('eioejeo')
        self.velodyne = rospy.Subscriber('/robot1_velodyne_points',PointCloud2,self.point_cloud_callback) 
        self.rate = rospy.Rate(10)
        self.point_size = 180
        self.dbscan = DBSCAN(eps=0.2,min_samples=10)
        np.random.seed(1)
        
    def step(self):
        a = np.random.choice([1,2,3,4,5])
        # self.rate.sleep()
        time.sleep(0.1)
        print(a)




    def point_cloud_callback(self,data):
        self.random_point = np.zeros((self.point_size,3))
        xyzin = pc2.read_points(data,field_names=('x','y','z','intensity'),skip_nans=True)  
        xyzin = np.array(list(xyzin))
        indices = np.where(xyzin[:,-1]>100)[0]
        new = xyzin[indices,:-1]

        self.dbscan.fit(new)
        center = []
        for label in set(self.dbscan.labels_):
            if label != -1:
                indice = self.dbscan.labels_ == label
                center.append(new[indice])

        center_mean_0 = np.array(center[0]).mean(0)[1]
        center_mean_1 = np.array(center[0]).mean(0)[1]
        if center_mean_0 > center_mean_1:
            center_left = np.array(center[0][:,:-1])
            center_right = np.array(center[1][:,:-1])
        else:
            center_left = np.array(center[1][:,:-1])
            center_right = np.array(center[0][:,:-1])
        random.shuffle(center_left)
        random.shuffle(center_right)
        random_left_point = np.zeros((80,2))
        random_right_point = np.zeros((80,2))
        random_left_point[:min(80,center_left.shape[0])] = center_left[:min(80,center_left.shape[0])]
        random_right_point[:min(80,center_right.shape[0])] = center_right[:min(80,center_right.shape[0])]
        sorted_left_indices = np.lexsort((random_left_point[:, -1],))
        left_point = random_left_point[sorted_left_indices]
        sorted_right_indices = np.lexsort((random_right_point[:, -1],))
        right_point = random_right_point[sorted_right_indices]
        self.final_point = np.concatenate([left_point,right_point],0)
        
        pass
if __name__ == '__main__':
    
 
    t = Test()
    for k in range(30):
        # print('***********')
        t.step()

