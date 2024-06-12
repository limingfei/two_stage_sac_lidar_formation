import numpy as np
import pandas as pd
import os
from gazebo_msgs.msg import ModelState, LinkState, ModelStates, LinkStates
import rospy
from gazebo_msgs.srv import SetModelState, SetLinkState
import time
from scipy.spatial.transform import Rotation
def reset_model_state(x,y,o_x,o_y,o_z,o_w,name):
        set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state',SetModelState)
        model_state = ModelState()
        model_state.model_name = name

        model_state.pose.position.x = x
        model_state.pose.position.y = y


        model_state.pose.orientation.x = o_x
        model_state.pose.orientation.y = o_y
        model_state.pose.orientation.z = o_z
        model_state.pose.orientation.w = o_w
        set_model_state_proxy(model_state)


if __name__ == '__main__':
    rospy.init_node('tttttt')
    lstm_0_4_data_replace = pd.read_csv('csv_files/test_many_1_1.csv')
    lstm_0_4_data_m = pd.read_csv('csv_files/test_many_1.csv')
    lstm_0_4_data2_replace = pd.read_csv('csv_files/test_many_2_1.csv')
    lstm_0_4_data2_m = pd.read_csv('csv_files/test_many_2.csv')
    # lstm_0_4_data3 = pd.read_csv('csv_file/test_many_3.csv')
    lstm_0_4_data3_replace = pd.read_csv('csv_files/test_many_3_1.csv')
    lstm_0_4_data3_m = pd.read_csv('csv_files/test_many_3.csv')
    lstm_0_4_data_m[:70] = lstm_0_4_data_replace[:70]
    lstm_0_4_data2_m[:70] = lstm_0_4_data2_replace[:70]
    lstm_0_4_data3_m[:70] = lstm_0_4_data3_replace[:70]

    num = 700
    lstm_0_4_l_x_m = lstm_0_4_data_m['eval/robot2_x'][:num]
    lstm_0_4_l_y_m = lstm_0_4_data_m['eval/robot2_y'][:num]
    lstm_0_4_l_yaw_m = lstm_0_4_data_m['eval/robot2_yaw'][:num]

    lstm_0_4_f_x_0_m = lstm_0_4_data_m['eval/robot1_x'][:num]
    lstm_0_4_f_y_0_m = lstm_0_4_data_m['eval/robot1_y'][:num]
    lstm_0_4_f_yaw_0_m = lstm_0_4_data_m['eval/robot1_yaw'][:num]

    lstm_0_4_f_x_1_m = lstm_0_4_data2_m['eval/robot1_x'][:num]
    lstm_0_4_f_y_1_m = lstm_0_4_data2_m['eval/robot1_y'][:num]
    lstm_0_4_f_yaw_1_m = lstm_0_4_data2_m['eval/robot1_yaw'][:num]

    lstm_0_4_f_x_2_m = lstm_0_4_data3_m['eval/robot1_x'][:num]
    lstm_0_4_f_y_2_m = lstm_0_4_data3_m['eval/robot1_y'][:num]
    lstm_0_4_f_yaw_2_m = lstm_0_4_data3_m['eval/robot1_yaw'][:num]

    poses = [[lstm_0_4_l_x_m,lstm_0_4_l_y_m,lstm_0_4_l_yaw_m],[lstm_0_4_f_x_0_m,lstm_0_4_f_y_0_m,lstm_0_4_f_yaw_0_m],
             [lstm_0_4_f_x_1_m,lstm_0_4_f_y_1_m,lstm_0_4_f_yaw_1_m],[lstm_0_4_f_x_2_m,lstm_0_4_f_y_2_m,lstm_0_4_f_yaw_2_m]]

    names = ['mycar2','mycar1','mycar3','mycar4']
    # rate = 280

    for rate in range(0,700,70):
         
        print(rate)
        for i,name in enumerate(names):
            x = poses[i][0][rate]
            y = poses[i][1][rate]
            r_q = Rotation.from_euler('xyz',(0,0,poses[i][2][rate]))
            ox,oy,oz,ow = r_q.as_quat()
            reset_model_state(x,y,ox,oy,oz,ow,name)
        time.sleep(10)
    rate = 699
    for i,name in enumerate(names):
            x = poses[i][0][rate]
            y = poses[i][1][rate]
            r_q = Rotation.from_euler('xyz',(0,0,poses[i][2][rate]))
            ox,oy,oz,ow = r_q.as_quat()
            reset_model_state(x,y,ox,oy,oz,ow,name)
            



    





