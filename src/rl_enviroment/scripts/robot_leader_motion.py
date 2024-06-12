from geometry_msgs.msg import Twist
import rospy


class LeaderMotion(object):
    def __init__(self):
        self.acc_phase = False
        self.uni_phase = False
        self.dec_phase = False
        self.velocity_phase = 0
        self.acc_v = 0.0025
        self.acc_ang = 0.00125
        self.x_max_steps = 160
        self.y_max_steps = 160
        self.z_max_steps = 160
        self.x_steps = 0
        self.tota_step = 0
        self.rate = rospy.Rate(10)
        current_v = 0
        self.motion_done = False
    def x_motion(self,current_v,expect_v=0.2):
        self.tota_step += 1
        cmd = Twist()
        if self.velocity_phase == 0:
            self.motion_done = False
            if current_v < expect_v:
                current_v += self.acc_v
                current_v = min(current_v,expect_v)                
            else:
                current_v = current_v
                self.velocity_phase = 1
        elif self.velocity_phase == 1:
            self.motion_done = False
            if self.x_steps < self.x_max_steps:
                self.x_steps += 1
                current_v = current_v
            else:
                self.x_steps = 0
                self.velocity_phase = 2
        elif self.velocity_phase == 2:
            if current_v > 0:
                current_v -= self.acc_v
                current_v = max(current_v,0)
                self.motion_done = False
            else:
                self.motion_done = True
                self.velocity_phase = 0
                self.tota_step = 0
        else:
            rospy.logerr('error in select motion mode!')
        cmd.linear.x = current_v
        return cmd,self.motion_done
    def y_motion(self,current_v,expect_v=0.2):
        self.tota_step += 1
        cmd = Twist()
        if expect_v < 0:
            self.ori = -1
        else:
            self.ori = 1
        if self.velocity_phase == 0:
            self.motion_done = False
            if abs(current_v) < abs(expect_v):
                current_v += self.ori*self.acc_v
                current_v = min(current_v,expect_v) if self.ori == 1 else max(current_v,expect_v)                
            else:
                current_v = current_v
                self.velocity_phase = 1
        elif self.velocity_phase == 1:
            self.motion_done = False
            if self.x_steps < self.x_max_steps:
                self.x_steps += 1
                current_v = current_v
            else:
                current_v = current_v
                self.x_steps = 0
                self.velocity_phase = 2
        elif self.velocity_phase == 2:
            if abs(current_v) > 0:
                current_v -= self.ori*self.acc_v
                current_v = max(current_v,0) if self.ori == 1 else min(current_v,0)
                self.motion_done = False
            else:
                current_v = current_v
                self.motion_done = True
                self.velocity_phase = 0
                self.tota_step = 0
        else:
            rospy.logerr('error in select motion mode!')
        cmd.linear.y = current_v
        return cmd,self.motion_done
    def ang_motion(self,current_v,expect_v=0.1):
        if expect_v < 0:
            self.ori = -1
        else:
            self.ori = 1
        self.tota_step += 1
        cmd = Twist()
        if self.velocity_phase == 0:
            self.motion_done = False
            if abs(current_v) < abs(expect_v):
                current_v += self.ori*self.acc_ang
                current_v = min(current_v,expect_v) if self.ori == 1 else max(current_v,expect_v)                
            else:
                current_v = current_v
                self.velocity_phase = 1
        elif self.velocity_phase == 1:
            self.motion_done = False
            if self.x_steps < self.x_max_steps:
                self.x_steps += 1
                current_v = current_v
            else:
                self.x_steps = 0
                self.velocity_phase = 2
                current_v = current_v
        elif self.velocity_phase == 2:
            if abs(current_v) > 0:
                current_v -= self.ori*self.acc_ang
                current_v = max(current_v,0) if self.ori == 1 else min(current_v,0)
                self.motion_done = False
            else:
                current_v = current_v
                self.motion_done = True
                self.velocity_phase = 0
                self.tota_step = 0
        else:
            rospy.logerr('error in select motion mode!')
        cmd.angular.z = current_v
        return cmd,self.motion_done
    def sleep(self):
        self.rate.sleep()
# rospy.init_node('leader_mo')   
# lm = LeaderMotion()
# for i in range(2):
#     print('--------------------------------')

#     c = Twist()
#     d = False
#     while not d:
#         c,d = lm.y_motion(c.linear.y,0.2)
#         print(c.linear.y,d,lm.tota_step)


class Motion(object):
    def __init__(self):
        self.acc_v = 0.0025
        self.acc_ang = 0.00125
        pass
    def acc_x(self,current_v,expect_v=0.2):
        if current_v < expect_v:
            current_v += self.acc_v
            current_v = min(current_v,expect_v)                
        else:
            current_v = current_v
        return current_v
    def dec_x(self,current_v,expect_v=0.2):
        
        if current_v > expect_v:
            current_v -= self.acc_v
            current_v = max(current_v,expect_v)                
        else:
            current_v = current_v
        return current_v
    def acc_yaw(self,current_v,expect_yaw=0.2):
       
        if current_v < expect_yaw:
            current_v += self.acc_ang
            current_v = min(current_v,expect_yaw)                
        else:
            current_v = current_v
        return current_v
    def dec_yaw(self,current_v,expect_yaw=0.2):
        if current_v > expect_yaw:
            current_v -= self.acc_ang
            current_v = max(current_v,expect_yaw)                
        else:
            current_v = current_v
        return current_v

