#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from pymoveit2 import MoveIt2
# from pymoveit2.robots import ur5
from sensor_msgs.msg import JointState
from math import pi
import Arm_Lib
import random
from rclpy.callback_groups import ReentrantCallbackGroup
from moveit.planning import MoveItPy
import os
joints_name = ['right_arm_joint1', 'right_arm_joint2', 'right_arm_joint3', 'right_arm_joint4', 'right_arm_joint5', 'right_grip_joint']
from moveit_configs_utils import MoveItConfigsBuilder
from pathlib import Path


class DofbotRandomMove(Node):
    def __init__(self):
        super().__init__("dofbot_random_move")


        # pkg_path = "/home/sgitai/dofbot2_moveit/"
        # urdf_path = os.path.join(pkg_path, 'config', 'dofbot2.urdf')
        # srdf_path = os.path.join(pkg_path, 'config', 'yahboom_dofbot_dual_arm.srdf')
        
        # # 2. 读取文件内容
        # with open(urdf_path, 'r') as file:
        #     robot_description = file.read()
        # with open(srdf_path, 'r') as file:
        #     robot_description_semantic = file.read()
        
        # # 3. 创建参数字典，用于初始化MoveItPy
        # moveit_config = {
        #     "robot_description": robot_description,
        #     "robot_description_semantic": robot_description_semantic,
        #     "robot_description_kinematics": {"planning_group": "dofbot_right"} # 替换为你的规划组名
        # }
        # moveit_config = (
        #     MoveItConfigsBuilder(robot_name="ur", package_name="ur_moveit_config")
        #     .robot_description_semantic(Path("srdf") / "ur.srdf.xacro", {"name": "ur10e"})
        #     .moveit_cpp(Path("config") / "moveit_cpp.yaml")
        #     .to_moveit_configs()
        # ).to_dict()
        
        # moveit_config = (
        #     MoveItConfigsBuilder("yahboom_dofbot_dual_arm", package_name="dofbot2_moveit")
        #     .robot_description(file_path="config/yahboom_dofbot_dual_arm.urdf.xacro")
        #     .robot_description_semantic(file_path="config/yahboom_dofbot_dual_arm.srdf")
        #     .trajectory_execution(file_path="config/moveit_controllers.yaml")
        #     .moveit_cpp(file_path="config/moveit_cpp.yaml")
        #     .to_moveit_configs()
        # ).to_dict()

        # # rclpy.init()
        # # 4. 使用配置字典初始化MoveItPy
        # self.moveit_interface = MoveItPy(
        #     node_name="yahboom_dofbot_dual_arm",
        #     config_dict=moveit_config
        # )
        # # self.moveit_interface = MoveItPy()
        # self.moveit = moveit_interface.get_planning_component("dofbot_right")
        # self.moveit.set_start_state_to_current_state()

        # moveit.shutdown()
        # 初始化 MoveIt2
        # node = Node("init_node")
        # self.moveit2 = MoveIt2(
        #     node=node,
        #     joint_names=joints_name,
        #     base_link_name='base_link',
        #     end_effector_name='right_grip_joint',
        #     group_name="dofbot_right",
        #     callback_group=ReentrantCallbackGroup()
        # )

        # executor = rclpy.executors.MultiThreadedExecutor(2)
        # self.moveit2.planner_id = "RRTConnectkConfigDefault"
        
        # self.moveit2.num_planning_attempts = 10
        # self.moveit2.cartesian_avoid_collisions = False
        # self.moveit2.cartesian_jump_threshold = 0.0
        
        
        # 创建定时器用于随机运动
        self.timer = self.create_timer(5.0, self.random_move_callback)  # 每5秒执行一次
        
        # 定义关节限制（根据您的机械臂实际情况调整）
        self.pos_limits = [
            (-1.0, 1.0),  # joint1
            (-1.0, 1.0),  # joint2
            (-1.0, 1.0),  # joint3
        ]


    def random_move_callback(self):
        # 生成随机关节角度
        pos = []
        for min_val, max_val in self.pos_limits:
            pos.append(random.uniform(min_val, max_val))
        pos = [0.083003, -0.0074994, 0.42076]


        # self.moveit.set_goal_state(pose=pose)
        # plan_result = self.moveit.plan()
        # if plan_result.success:
        #     self.moveit_interface.execute(plan_result.trajectory)
        # 设置目标关节角度
        # moveit2.move_to_configuration([0.1,0.2,0.2,0.2,0.2,0.2])

        moveit2.move_to_pose(
            position=pos,
            quat_xyzw=[-0.0039459, -0.0036754, 0.68158, 0.73172],
            cartesian=False,
            cartesian_max_step=0.0025,
            cartesian_fraction_threshold=0.0,
        )

        moveit2.wait_until_executed()
        # 执行运动
        # self.moveit2.execute()

# def main(args=None):
    

if __name__ == '__main__':
    rclpy.init()
    node = Node("init_node")
    dofbot_node = DofbotRandomMove()

        
    moveit2 = MoveIt2(
        node=node,
        joint_names=joints_name,
        base_link_name='base_link',
        end_effector_name='right_arm_link5',
        group_name="dofbot_right",
        callback_group=ReentrantCallbackGroup()
    )
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor.add_node(dofbot_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        # 处理终止信号
        pass
    finally:
        # 销毁节点并关闭 ROS 2 系统
        dofbot_node.destroy_node()
        node.destroy_node()
        rclpy.shutdown()