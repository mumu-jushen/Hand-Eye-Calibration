import time
import math
import random

import numpy as np
import pybullet as p
import pybullet_data

from utilities import Camera
from collections import namedtuple
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


class FailToReachTargetError(RuntimeError):
    pass



class ClutteredPushGrasp:

    SIMULATION_STEP_DELAY = 1 / 240.

    def __init__(self, robot,  intrinsic=None, vis=False) -> None:
        self.robot = robot
        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        self.camera = Camera(p, intrinsic, 0.1,2.0)

        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeID = p.loadURDF("plane.urdf")

        self.robot.load()
        self.robot.step_simulation = self.step_simulation



        self.obj_ID = p.loadURDF("./object/Gear.urdf",
                                [0.4, 0.0, 0.15],
                                p.getQuaternionFromEuler([0, 1.57 , 0]), # 3.14
                                useFixedBase=True,
                                flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION)

        self.boxID = p.loadURDF("./tray/sunhan2.urdf",
                                [0.5, 0.0, 0.0],
                                p.getQuaternionFromEuler([0, 0 , 0]), # 3.14
                                useFixedBase=True,
                                flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION)

        for _ in range(120):
            self.step_simulation()

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)


    def step(self, action, rot_mat, control_method='joint'):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """
        assert control_method in ('joint', 'end')
        self.robot.move_ee(action[:-1], control_method)
        self.robot.move_gripper(action[-1])
        for _ in range(1):  # Wait for a few steps
            self.step_simulation()


        ################# for pbv camera
        T_ee = np.identity(4)
        T_ee[:3,:3] = rot_mat
        T_ee[:3,3] = np.array([action[0],action[1],action[2]])

        T_ee =  np.array([[ 0.04576107,  0.51490986 ,-0.85602206,  0.9      ],
                         [ 0.9985315  , 0.00129565  ,0.05415867,  -0.033        ],
                         [ 0.02899594 ,-0.85724335, -0.51409442,  0.8        ],
                         [ 0.       ,   0.       ,   0.      ,    1.        ]])

        
        return self.get_observation(T_ee)


    def get_observation(self,T_cam):
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.render(T_cam)
            obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None
        obs.update(self.robot.get_joint_obs())

        return obs

    def reset_box(self):
        # p.setJointMotorControl2(self.boxID, 0, p.POSITION_CONTROL, force=1)
        # p.setJointMotorControl2(self.boxID, 1, p.VELOCITY_CONTROL, force=0)
        print('**')

    def reset(self):
        self.robot.reset()
        # return self.get_observation(np.identity(4))

    def close(self):
        p.disconnect(self.physicsClient)








