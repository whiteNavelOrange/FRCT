#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from pymoveit2 import MoveIt2, GripperInterface
# from pymoveit2.robots import ur5
from sensor_msgs.msg import JointState
from math import pi
import Arm_Lib
import random
from rclpy.callback_groups import ReentrantCallbackGroup
from moveit.planning import MoveItPy
import os
from camera import Camera
from moveit_configs_utils import MoveItConfigsBuilder
from pathlib import Path
import numpy as np

right_joints_name = ['right_arm_joint1', 'right_arm_joint2', 'right_arm_joint3', 'right_arm_joint4', 'right_arm_joint5', 'right_grip_joint']
left_joints_name = ['left_arm_joint1', 'left_arm_joint2', 'left_arm_joint3', 'left_arm_joint4', 'left_arm_joint5', 'left_grip_joint']

# class DofbotMove(Node):
#     def __init__(self):
#         super().__init__("dofbotmove")
        
#         # 创建定时器用于随机运动
#         # self.timer = self.create_timer(5.0, self.random_move_callback)  # 每5秒执行一次
#         self.create_data()
#     def create_data(self):
#         poses = []
#         for pose in poses:
#             self.move(pose)
        


#     def move(self, pose):

#         pose_left = pose[:7]
#         pose_right = pose[8:15]
#         hand_left = pose[7]
#         hand_right = pose[15]

#         moveit2.move_to_pose(
#             position=pose_right,
#             quat_xyzw=[-0.0039459, -0.0036754, 0.68158, 0.73172],
#             cartesian=False,
#             cartesian_max_step=0.0025,
#             cartesian_fraction_threshold=0.0,
#         )

#         moveit2.wait_until_executed()
#         # 执行运动
#         # self.moveit2.execute()

from concurrent.futures import ThreadPoolExecutor
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped
import threading
class Save_img(Node):
    def __init__(self):
        super().__init__('Save_data')
        self.base = '/home/sgitai/FRCT/real_data/train/bimanual_grasp_banana/all_variations/episodes/episode1/'
        self.rgb_path = os.path.join(self.base, 'front_rgb')
        self.depth_path = os.path.join(self.base, 'front_depth')
        os.makedirs(self.rgb_path, exist_ok=True)
        os.makedirs(self.depth_path, exist_ok=True)
        
        self.index = 0
        # self.save_img()
        self.save_timer = self.create_timer(10, self.save_img)
        # self.create_data()
        # self.executor = ThreadPoolExecutor(max_workers=4)
        # future = self.executor.submit(self.create_data)

        # self.timer = self.create_timer(2.0, self.save)  # 每5秒执行一次
        


    def save_img(self):
        while camera_node.queue_color and camera_node.queue_depth:
            color_img = camera_node.queue_color[-1] 
            depth_frame = camera_node.queue_depth[-1].astype(np.uint16)
            # print(np.max(depth_frame))
            # transform = self.tf_buffer.lookup_transform(
            #     "base_link",  # 目标坐标系
            #     "right_arm_link5",  # 源坐标系
            #     rclpy.time.Time()  # 获取最新可用的变换
            # )
            # current_right_pose = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z, 
            # transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]
            # if self.right_grip > 0:
            #     current_right_pose.append(1.0)
            # else:
            #     current_right_pose.append(0.0)

            # transform = self.tf_buffer.lookup_transform(
            #     "base_link",  # 目标坐标系
            #     "left_arm_link5",  # 源坐标系
            #     rclpy.time.Time()  # 获取最新可用的变换
            # )
            # current_left_pose = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z, 
            # transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]
            # if self.left_grip > 0:
            #     current_left_pose.append(1.0)
            # else:
            #     current_left_pose.append(0.0)
            # # print(current_left_pose)
            # # print(current_right_pose)
            # self.poses['left'].append(current_left_pose)
            # self.poses['right'].append(current_right_pose)
            import cv2
            # depth_frame = depth_frame.astype('float') / 65535.0 * 255.0 # 将深度值映射到0-1之间
            # # depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR) * 255.0  # 将深度图转换为RGB图像，并将像素值缩放到0-255之间
            # depth_frame = depth_frame.astype('uint16')  # 将像素值重新转换为16位整型
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(self.rgb_path, 'rgb_{:04d}.png'.format(self.index)), color_img)
            cv2.imwrite(os.path.join(self.depth_path, 'depth_{:04d}.png'.format(self.index)), depth_frame)
            self.index += 1
            # input('pause')
            



if __name__ == '__main__':
    rclpy.init()
    # dofbot_node = DofbotMove()
    camera_node = Camera()
    
    # dofbot_node.camera = camera_node
    save_node = Save_img()
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(camera_node)
    executor.add_node(save_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        # 处理终止信号
        pass
    finally:
        # 销毁节点并关闭 ROS 2 系统
        
        camera_node.destroy_node()
        rclpy.shutdown()

# if __name__ == '__main__':
#     rclpy.init()
#     node = Node("init_node")
#     dofbot_node = DofbotRandomMove()

        
#     right_moveit2 = MoveIt2(
#         node=node,
#         joint_names=right_joints_name,
#         base_link_name='base_link',
#         end_effector_name='right_arm_link5',
#         group_name="dofbot_right",
#         callback_group=ReentrantCallbackGroup()
#     )
#     executor = rclpy.executors.MultiThreadedExecutor(2)
#     executor.add_node(node)
#     executor.add_node(dofbot_node)
#     try:
#         executor.spin()
#     except KeyboardInterrupt:
#         # 处理终止信号
#         pass
#     finally:
#         # 销毁节点并关闭 ROS 2 系统
#         dofbot_node.destroy_node()
#         node.destroy_node()
#         rclpy.shutdown()