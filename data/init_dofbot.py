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
class DofbotMove(Node):
    def __init__(self):
        super().__init__('Save_data')

        # 创建定时器定期获取位姿
        self.timer = self.create_timer(5, self.init)

        # self.timer = self.create_timer(2.0, self.save)  # 每5秒执行一次
    def init(self):
        poses = [[-0.12392, 0.18041, 0.092393, -0.60642, 0.72777, 0.21045, 0.24149, 0,
        0.15567, 0.18713, 0.073922, -0.66819, 0.69698, 0.17877, 0.18916, 0]]

        for pose in poses:
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
        

    def move_left(self):
        pose_left = self.pose[:7]
        hand_left = self.pose[7]
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

            
    

    # def grip_topic(self, msg):
    #     if not isinstance(msg, JointState): 
    #         return
    #     # print(msg)
    #     # 定义关节角度容器
    #     # joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     self.left_grip = -(msg.position[3] * 116) - 90
    #     self.right_grip = -(msg.position[-1] * 116) - 90



if __name__ == '__main__':
    rclpy.init()
    right_node = Node("right_node")
    left_node = Node("left_node")
    dofbot_node = DofbotMove()
    
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
    executor.add_node(dofbot_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        # 处理终止信号
        pass
    finally:
        # 销毁节点并关闭 ROS 2 系统
        
        left_node.destroy_node()
        right_node.destroy_node()
        dofbot_node.destroy_node()
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