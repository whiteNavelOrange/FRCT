#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
# from pymoveit2.robots import ur5
from sensor_msgs.msg import JointState
from math import pi
import Arm_Lib
import random

RA2DE = 180 / pi

joints_name = ['right_arm_joint1', 'right_arm_joint2', 'right_arm_joint3', 'right_arm_joint4', 'right_arm_joint5']

class DofbotRandomMove(Node):
    def __init__(self):
        super().__init__("dofbot_random_move")
        
        # 初始化机械臂设备
        self.sbus_left = Arm_Lib.Arm_Device(com="/dev/ttyUSB0")
        self.sbus_right = Arm_Lib.Arm_Device(com="/dev/ttyUSB1")
        
        # 创建订阅者
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.topic_callback,
            10
        )
        
    def topic_callback(self, msg):
        # 如果不是该话题的数据直接返回
        if not isinstance(msg, JointState): 
            return
        print(msg)
        # 定义关节角度容器
        # joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        left_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        left_joints[0] = (msg.position[0] * RA2DE) + 90
        left_joints[1] = (msg.position[1] * RA2DE) + 90
        left_joints[2] = (msg.position[5] * RA2DE) + 90
        left_joints[3] = (msg.position[-2] * RA2DE) + 90
        left_joints[4] = (msg.position[2] * RA2DE) + 90
        left_joints[5] = (msg.position[3] * 116) + 180
        if left_joints[5] > 90:
            left_joints[5] = 130
        right_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        right_joints[0] = (msg.position[4] * RA2DE) + 90
        right_joints[1] = (msg.position[8] * RA2DE) + 90
        right_joints[2] = (msg.position[6] * RA2DE) + 90
        right_joints[3] = (msg.position[7] * RA2DE) + 90
        right_joints[4] = (msg.position[9] * RA2DE) + 90
        right_joints[5] = (msg.position[-1] * 116) + 180
        if right_joints[5] > 90:
            right_joints[5] = 165
        # 将接收到的弧度[-1.57,1.57]转换成角度[0,180]
        # for i in range(6): 
        #     joints[i] = (msg.position[i] * RA2DE) + 90
        #     if i == 5:
        #         joints[i] = (msg.position[i] * 116) + 180
        
        # 发送到驱动
        self.sbus_left.Arm_serial_servo_write6_array(left_joints, 100)
        self.sbus_right.Arm_serial_servo_write6_array(right_joints, 100)



def main(args=None):
    rclpy.init(args=args)
    
    dofbot_node = DofbotRandomMove()
    
    try:
        rclpy.spin(dofbot_node)
    except KeyboardInterrupt:
        pass
    finally:
        # 显式销毁节点
        dofbot_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()