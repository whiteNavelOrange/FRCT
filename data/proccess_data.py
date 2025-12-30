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
from concurrent.futures import ThreadPoolExecutor
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped
import threading
class Save_data(Node):
    def __init__(self):
        super().__init__('Save_data')
        self.base = '/home/sgitai/FRCT/real_data/train/bimanual_grasp_banana/all_variations/episodes/episode1/'
        self.rgb_path = os.path.join(self.base, 'front_rgb')
        self.depth_path = os.path.join(self.base, 'front_depth')
        os.makedirs(self.rgb_path, exist_ok=True)
        os.makedirs(self.depth_path, exist_ok=True)
        self.right_grip = 0
        self.left_grip = 0
        self.poses = {'left': [], 'right': []}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.grip_topic,
            10
        )
        self.index = 0
        # 创建定时器定期获取位姿
        self.timer = self.create_timer(5, self.create_data)
        # self.save_timer = self.create_timer(0.1, self.save)
        # self.create_data()
        # self.executor = ThreadPoolExecutor(max_workers=4)
        # future = self.executor.submit(self.create_data)

        # self.timer = self.create_timer(2.0, self.save)  # 每5秒执行一次
    def create_data(self):
        self.poses =   [[-0.12392, 0.18041, 0.092393, -0.60642, 0.72777, 0.21045, 0.24149, 0,
        0.15567, 0.18713, 0.073922, -0.66819, 0.69698, 0.17877, 0.18916, 0],
        [-0.20151, 0.17873, 0.014439, 0.78778, -0.59903, -0.11228, -0.089233, 1, 
        0.26051, 0.11156, 0.035369, -0.62696, 0.76769, 0.010168, 0.13222, 1],
        [-0.20151, 0.17873, 0.014439, 0.78778, -0.59903, -0.11228, -0.089233, 1, 
        0.03507, 0.10625, 0.032926, -0.69505, 0.70803, 0.12483, -0.0029322, 1],
        [0.015097, 0.10229, 0.15273, -0.23609, 0.85153, 0.12809, 0.45029, 1,
        0.03507, 0.10625, 0.032926, -0.69505, 0.70803, 0.12483, -0.0029322, 1],
        [0.0066319, 0.1318, 0.044089, -0.32945, 0.92032, 0.070314, 0.19883, 1,
        0.03507, 0.10625, 0.032926, -0.69505, 0.70803, 0.12483, -0.0029322, 1],
        [0.0066319, 0.1318, 0.044089, -0.32945, 0.92032, 0.070314, 0.19883, 0,
        0.03507, 0.10625, 0.032926, -0.69505, 0.70803, 0.12483, -0.0029322, 0],
        [-0.12392, 0.18041, 0.092393, -0.60642, 0.72777, 0.21045, 0.24149, 0,
        0.15567, 0.18713, 0.073922, -0.66819, 0.69698, 0.17877, 0.18916, 0],
        ]

        for pose in self.poses:
            self.pose = pose
            # self.move_data(pose)
            thread1 = threading.Thread(target=self.move_left)
            thread2 = threading.Thread(target=self.move_right)
            thread1.start()
            thread2.start()
            thread1.join()
            thread2.join()
            left_moveit2.wait_until_executed()
            right_moveit2.wait_until_executed()
            # self.save()
        import pickle
        with open(os.path.join(self.base, 'low_dim_obs.pkl'), 'wb') as f:
	        pickle.dump(self.poses, f)
        a=input("pasue")
        

    def move_left(self):
        pose_left = self.pose[:7]
        hand_left = self.pose[7]

        succ = False
        while not succ:
            succ = left_moveit2.move_to_pose(
                position=pose_left[:3],
                quat_xyzw=pose_left[3:],
                cartesian=False,
                cartesian_max_step=0.0025,
                cartesian_fraction_threshold=0.8,
            )
        # left_moveit2.wait_until_executed()

        if hand_left == 1.0:  # 抓取
            print("Closing gripper...")
            left_gripper_interface.close()
            # gripper_interface.wait_until_executed()
            # time.sleep(1)
        elif hand_left == 0.0:  # 放置
            print("Opening gripper...")
            # import ipdb; ipdb.set_trace()
            left_gripper_interface.open()
            # gripper_interface.wait_until_executed()
            # time.sleep(1)
        
    def move_right(self):
        pose_right = self.pose[8:15]
        hand_right = self.pose[15]
        succ = False
        while not succ:
            succ = right_moveit2.move_to_pose(
                position=pose_right[:3],
                quat_xyzw=pose_right[3:],
                cartesian=False,
                cartesian_max_step=0.0025,
                cartesian_fraction_threshold=0.8,
            )
        # right_moveit2.wait_until_executed()
        if hand_right == 1.0:  # 抓取
            print("Closing gripper...")
            right_gripper_interface.close()
            # gripper_interface.wait_until_executed()
            # time.sleep(1)
        elif hand_right == 0.0:  # 放置
            print("Opening gripper...")
            # import ipdb; ipdb.set_trace()
            right_gripper_interface.open()
            # gripper_interface.wait_until_executed()
            # time.sleep(1)

    def move_data(self, pose):

        pose_left = pose[:7]
        pose_right = pose[8:15]
        hand_left = pose[7]
        hand_right = pose[15]

        right_moveit2.move_to_pose(
            position=pose_right[:3],
            quat_xyzw=pose_right[3:],
            cartesian=False,
            cartesian_max_step=0.0025,
            cartesian_fraction_threshold=0.8,
        )
        right_moveit2.wait_until_executed()
        left_moveit2.move_to_pose(
            position=pose_left[:3],
            quat_xyzw=pose_left[3:],
            cartesian=False,
            cartesian_max_step=0.0025,
            cartesian_fraction_threshold=0.8,
        )
        left_moveit2.wait_until_executed()

        if hand_left == 1.0:  # 抓取
            print("Closing gripper...")
            left_gripper_interface.close()
            # gripper_interface.wait_until_executed()
            # time.sleep(1)
        elif hand_left == 0.0:  # 放置
            print("Opening gripper...")
            # import ipdb; ipdb.set_trace()
            left_gripper_interface.open()
            # gripper_interface.wait_until_executed()
            # time.sleep(1)
        if hand_right == 1.0:  # 抓取
            print("Closing gripper...")
            right_gripper_interface.close()
            # gripper_interface.wait_until_executed()
            # time.sleep(1)
        elif hand_right == 0.0:  # 放置
            print("Opening gripper...")
            # import ipdb; ipdb.set_trace()
            right_gripper_interface.open()
            # gripper_interface.wait_until_executed()
            # time.sleep(1)
        
        # self.save()


    def save(self):
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
            cv2.imwrite(os.path.join(self.rgb_path, 'rgb_{:04d}.png'.format(self.index)), color_img)
            cv2.imwrite(os.path.join(self.depth_path, 'depth_{:04d}.png'.format(self.index)), depth_frame)
            self.index += 1
            
    
    def destroy_node(self):
        import pickle
        with open(os.path.join(self.base, 'low_dim_obs.pkl'), 'wb') as f:
	        pickle.dump(self.poses, f)
        super().destroy_node()


    def grip_topic(self, msg):
        if not isinstance(msg, JointState): 
            return
        # print(msg)
        # 定义关节角度容器
        # joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.left_grip = -(msg.position[3] * 116) - 90
        self.right_grip = -(msg.position[-1] * 116) - 90



if __name__ == '__main__':
    rclpy.init()
    right_node = Node("right_node")
    left_node = Node("left_node")
    # dofbot_node = DofbotMove()
    camera_node = Camera()
    
    # dofbot_node.camera = camera_node
    save_node = Save_data()
    right_moveit2 = MoveIt2(
        node=right_node,
        joint_names=right_joints_name,
        base_link_name='base_link',
        end_effector_name='right_arm_link5',
        group_name="dofbot_right",
        callback_group=ReentrantCallbackGroup()
    )
    right_moveit2.planner_id = 'RRTConnectkConfigDefault'
    left_moveit2 = MoveIt2(
        node=left_node,
        joint_names=left_joints_name,
        base_link_name='base_link',
        end_effector_name='left_arm_link5',
        group_name="dofbot_left",
        callback_group=ReentrantCallbackGroup()
    )
    left_moveit2.planner_id = 'RRTConnectkConfigDefault'
    left_gripper_interface = GripperInterface(
        node=left_node,
        gripper_joint_names=['left_grip_joint'],
        open_gripper_joint_positions=[-1.5],
        closed_gripper_joint_positions=[1.3],
        gripper_group_name='hand_left',
        callback_group=ReentrantCallbackGroup(),
        gripper_command_action_name="gripper_action_controller/gripper_cmd",
        max_effort=1.0
    )    
    right_gripper_interface = GripperInterface(
        node=right_node,
        gripper_joint_names=['right_grip_joint'],
        open_gripper_joint_positions=[-1.5],
        closed_gripper_joint_positions=[1.3],
        gripper_group_name='hand_right',
        callback_group=ReentrantCallbackGroup(),
        gripper_command_action_name="gripper_action_controller/gripper_cmd",
        max_effort=1.0
    )  
    
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(right_node)
    executor.add_node(left_node)
    executor.add_node(save_node)
    # executor.add_node(dofbot_node)
    executor.add_node(camera_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        # 处理终止信号
        pass
    finally:
        # 销毁节点并关闭 ROS 2 系统
        
        # node.destroy_node()
        camera_node.destroy_node()
        save_node.destroy_node()
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