#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import ros2_numpy
import numpy as np
import cv2

class Camera(Node):
    def __init__(self):
        super().__init__('kinectdk_node')

        self.queue_hbody = []
        self.queue_color = []
        self.queue_depth = []
        self.queue_left_hand = []
        self.rgb_info = None
        self.depth_info = None
        self.get_logger().info('*' * 20)

        # 创建CvBridge实例
        self.cvbridge = CvBridge()

        # 订阅器（ROS2）
        qos_profile = QoSProfile(depth=10)
        self.color_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.color_callback,
            qos_profile
        )
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            qos_profile
        )
        self.rgbinfo_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.rgbinfo_callback,
            qos_profile
        )
        self.depthinfo_sub = self.create_subscription(
            CameraInfo,
            '/camera/depth/camera_info',
            self.depthinfo_callback,
            qos_profile
        )

    def rgbinfo_callback(self, camera_info):
        self.rgb_info = camera_info
        # 如果只需要一次信息，可以在这里取消订阅
        # self.rgbinfo_sub.destroy()

    def depthinfo_callback(self, camera_info):
        self.depth_info = camera_info
        # 如果只需要一次信息，可以在这里取消订阅
        # self.depthinfo_sub.destroy()

    def color_callback(self, image_msg):
        try:
            cv_image = ros2_numpy.numpify(image_msg)
            if len(self.queue_color) >= 3:
                self.queue_color.pop(0)
            self.queue_color.append(np.asarray(cv_image))
            # print(self.queue_color[0])
            self.get_logger().debug(f"Color Image Received: {cv_image.shape}")
        except Exception as e:
            self.get_logger().error(f"Error in color_callback: {e}")

    def depth_callback(self, image_msg):
        try:
            cv_image = ros2_numpy.numpify(image_msg)
            if len(self.queue_depth) >= 3:
                self.queue_depth.pop(0)
            self.queue_depth.append(np.asarray(cv_image))
            # print(self.queue_depth[0])
            self.get_logger().debug(f"Depth Image Received: {cv_image.shape}")
        except Exception as e:
            self.get_logger().error(f"Error in depth_callback: {e}")

    def release(self):
        self.get_logger().info("Kinect release")
        self.color_sub.destroy()
        self.depth_sub.destroy()
