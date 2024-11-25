import os
import cv2
import numpy as np
import pybullet as p
from tqdm import tqdm
from env import ClutteredPushGrasp
from robot import Panda, UR5Robotiq85, UR5Robotiq140
from utilities import  Camera, CameraIntrinsic
import time
import math
from scipy.spatial.transform import Rotation as R
import itertools


def quaternion2euler_mat(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler('xyz', degrees=True)
    mat = r.as_matrix()
    return euler[0]*(math.pi/180.), euler[1]*(math.pi/180.),euler[2]*(math.pi/180.), mat





def user_control_demo():

    intrinsic = CameraIntrinsic(640, 480,567.53720406, 569.36175922, 312.66570357,257.1729701)

    # camera = None
    robot = Panda((0, 0, 0.), (0, 0, 0  ))
    env = ClutteredPushGrasp(robot, intrinsic, vis=True)

    env.reset()


    i = 0
    for ii in range(1000):


        ############## read robot info
        robot_info =  env.robot.get_joint_obs()
        ee_pos = robot_info['ee_pos']
        quaternion = [ee_pos[1][0], ee_pos[1][1], ee_pos[1][2], ee_pos[1][3] ]
        roll, pitch, yaw, rot_mat = quaternion2euler_mat(quaternion)
        x,y,z = ee_pos[0][0], ee_pos[0][1], ee_pos[0][2]

        T_link8_in_base = np.identity(4)
        T_link8_in_base[:3,:3] = rot_mat
        T_link8_in_base[:3,3] = np.array([ ee_pos[0][0], ee_pos[0][1], ee_pos[0][2]])
        T_hand_in_link8 = np.array([[ 0.70675836,  0.70745503 , 0. ,  0     ],
                                    [-0.70745503,  0.70675836 ,-0. ,  0     ],
                                    [-0.      ,    0.  ,        1.  , 0     ],
                                    [0.      ,    0.  ,        0.  , 1     ],])

        T_hand_in_base = np.dot(T_link8_in_base, T_hand_in_link8)
        print(T_hand_in_base)
        print(rot_mat)
        print(x,y,z )


        ## Do not move
        action = x, y, z, roll, pitch, yaw, 0.04
        obs = env.step(action, rot_mat, 'end')


        ## move
        # view = view_all[i]
        # rpy = R.from_matrix(view[:3, :3]).as_euler('xyz', degrees=False)
        # action = view[0, 3], view[1, 3], view[2, 3], rpy[0], rpy[1], rpy[2], 0.04
        # obs = env.step(action, view[:3, :3], 'end')


        rgb = obs['rgb']
        depth = obs['depth']
        seg = obs['seg']
        seg[seg==2] = 255
        seg[seg!=255] = 0

        cv2.imwrite('/home/sunh/6D_ws/ActivePose/FoundationPose-main/demo_data/franka_sim/rgb/{:06d}-color.png'.format(i),rgb)
        cv2.imwrite('/home/sunh/6D_ws/ActivePose/FoundationPose-main/demo_data/franka_sim/depth/{:06d}-depth.png'.format(i),np.uint16(depth*1000.0))
        # cv2.imwrite('/home/sunh/6D_ws/ActivePose/FoundationPose-main/demo_data/franka_sim/masks/{:06d}-color.png'.format(i),seg)
        cv2.imshow('rgb',rgb)
        cv2.waitKey()


if __name__ == '__main__':
    user_control_demo()


