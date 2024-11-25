#############################################################################################################################################################################################################################################
# Generate  a mask for rgb image, and save the mask
#############################################################################################################################################################################################################################################
# import cv2
# import numpy as np
#
# # 定义全局变量
# drawing = False  # 是否正在绘制
# start_x, start_y = -1, -1  # 开始点
# end_x, end_y = -1, -1  # 结束点
#
# # 鼠标回调函数
# def draw_rectangle(event, x, y, flags, param):
#     global drawing, start_x, start_y, end_x, end_y, img
#
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         start_x, start_y = x, y
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         end_x, end_y = x, y
#         # 画出矩形
#         cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
#         cv2.imshow('image', img)
#
# # 读取图像
# img = cv2.imread('/home/sunh/6D_ws/ActivePose/FoundationPose-main/demo_data/mp3/rgb/000001-color.png')
# if img is None:
#     print("Error: Could not load image.")
#     exit()
#
# # 创建窗口
# cv2.namedWindow('image')
# cv2.setMouseCallback('image', draw_rectangle)
#
# # 显示图像
# cv2.imshow('image', img)
#
# # 等待用户完成选择
# while True:
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q') or key == 27:  # 按 'q' 或 'Esc' 键退出
#         break
#
# # 生成 mask
# mask = np.zeros(img.shape[:2], dtype=np.uint8)
# cv2.rectangle(mask, (start_x, start_y), (end_x, end_y), 255, -1)
#
#
# cv2.imshow('mask', mask)
# cv2.waitKey(0)
# # 保存 mask
# cv2.imwrite('/home/sunh/6D_ws/ActivePose/FoundationPose-main/demo_data/mp3/masks/000001-color.png', mask)
#
# # 关闭所有窗口
# cv2.destroyAllWindows()



#############################################################################################################################################################################################################################################
# compute the
#############################################################################################################################################################################################################################################

import numpy as np
from scipy.spatial.transform import Rotation as R
import math


T_hand_in_cam = np.array([[-6.6938382e-01, -7.4287915e-01,  7.4758921e-03,  9.0268021e-04],
                         [ 1.0253182e-01, -8.2412146e-02 , 9.9130994e-01, -2.1194221e-01],
                         [-7.3580742e-01 , 6.6433346e-01 , 1.3133414e-01 , 6.1281210e-01],
                         [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00 , 1.0000000e+00]])


T_hand_in_base = np.array([[ 6.51533954e-01, -6.50923313e-01,  3.89618205e-01 , 2.64867659e-01],
                            [-7.06761111e-01, -7.07452280e-01, -4.70226965e-05, -4.56169896e-06],
                            [ 2.75666896e-01, -2.75336358e-01, -9.20976467e-01 , 6.69039437e-01],
                            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]])

T_cam_in_hand = np.linalg.inv(T_hand_in_cam)

qua = R.from_matrix(T_cam_in_hand[:3,:3]).as_quat()
print("calibration results: rosrun tf static_transform_publisher "+str(T_cam_in_hand[0,3])+' '+str(T_cam_in_hand[1,3])+' '+str(T_cam_in_hand[2,3])+' '+ str(qua[0])+' '+ str(qua[1])+' '+ str(qua[2])+' '+ str(qua[3])+" /hand /camera_link  50")

qua = R.from_matrix(T_hand_in_base[:3,:3]).as_quat()
print("calibration results: rosrun tf static_transform_publisher "+str(T_hand_in_base[0,3])+' '+str(T_hand_in_base[1,3])+' '+str(T_hand_in_base[2,3])+' '+ str(qua[0])+' '+ str(qua[1])+' '+ str(qua[2])+' '+ str(qua[3])+" /base /hand 50")

T_cam_in_base = np.dot( T_hand_in_base,  np.linalg.inv(T_hand_in_cam))



def re(R_est, R_gt): # rotation error
    assert R_est.shape == R_gt.shape == (3, 3)
    error_cos = float(0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0))

    # Avoid invalid values due to numerical errors.
    error_cos = min(1.0, max(-1.0, error_cos))

    error = math.acos(error_cos)
    error = 180.0 * error / np.pi  # Convert [rad] to [deg].
    return error


def te(t_est, t_gt): # translation error
    assert t_est.size == t_gt.size == 3
    error = np.linalg.norm(t_gt - t_est)
    return error

#### the groundtruth of the T_cam_in_base
T_cam_in_base_gt = np.array([[0.04576107, 0.51490986, -0.85602206, 0.9],
                 [0.9985315, 0.00129565, 0.05415867, -0.033],
                 [0.02899594, -0.85724335, -0.51409442, 0.8],
                 [0., 0., 0., 1.]])

R_ERROR = re(T_cam_in_base_gt[:3,:3], T_cam_in_base[:3,:3])
t_ERROR = te(T_cam_in_base_gt[:3,3], T_cam_in_base[:3,3])

print(R_ERROR, t_ERROR)


