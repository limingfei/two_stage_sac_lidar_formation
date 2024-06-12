import message_filters
import rospy
from sensor_msgs.msg import Image,PointCloud2
from sensor_msgs import point_cloud2 as pc2
import os
import cv2
from cv_bridge import CvBridge
import numpy as np
import time

IMAGE_TOPIC = '/robot1_camera/color/image_raw'
PC_TOPIC = '/robot1_velodyne_points'
RATE = 1


class AgvSub(object):
    def __init__(self):
        rospy.init_node('velocity_node',anonymous=True)
        self.robot1_force_sub = message_filters.Subscriber(IMAGE_TOPIC, Image, queue_size=1)
        self.robot2_force_sub = message_filters.Subscriber(PC_TOPIC, PointCloud2, queue_size=1)
        self.bridge = CvBridge()
        self.img_folder = 'dataset_folder/img_bjut'
        self.pc_folder = 'dataset_folder/pc_bjut'
        if not os.path.exists(self.img_folder):
            os.makedirs(self.img_folder)
        if not os.path.exists(self.pc_folder):
            os.makedirs(self.pc_folder)
    def get_relative_position(self):
        sync_robot = message_filters.ApproximateTimeSynchronizer([self.robot1_force_sub, self.robot2_force_sub],10,0.1,allow_headerless=False)
        sync_robot.registerCallback(self.multi_callback)

    def multi_callback(self,img_data,point_cloud_data):
        img = self.bridge.imgmsg_to_cv2(img_data,'bgr8')
        cv2.imwrite(os.path.join(self.img_folder,"img_{}.png".format(len(os.listdir(self.img_folder)))),img)
        xyzin = pc2.read_points(point_cloud_data,field_names=('x','y','z','intensity'),skip_nans=True)  
        xyzin = np.array(list(xyzin))
        with open(os.path.join(self.pc_folder,'array_file_{}.bin'.format(len(os.listdir(self.pc_folder)))), 'wb') as file:
            xyzin.tofile(file)
        

if __name__ =='__main__':
    for k in range(RATE):
        sub = AgvSub()    
        sub.get_relative_position()
        time.sleep(0.1)

    