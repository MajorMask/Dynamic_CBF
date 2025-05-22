import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np

class DataLoader(Node):
    def __init__(self, slam_method="cuvslam"):
        super().__init__('data_loader')
        self.bridge = CvBridge()
        self.slam_method = slam_method

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10
        )
        self.pose_sub = self.create_subscription(
            Odometry, '/visual_slam/tracking/odometry', self.pose_callback, 10
        )

        # Data storage
        self.rgb_image = None
        self.depth_image = None
        self.pose = None

    def image_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def pose_callback(self, msg):
        # Convert pose based on SLAM method
        from nerfbridge.pose_utils import (
    cuvslam_to_nerfstudio,
    orbslam3_to_nerfstudio,
    mocap_to_nerfstudio,
)

class DataLoader(Node):
    def pose_callback(self, msg):
        """
        Handle pose data based on the selected SLAM method.
        """
        if self.slam_method == "cuvslam":
            hom_pose = pose_utils.ros_pose_to_homogenous(msg.pose.pose)
            self.pose = cuvslam_to_nerfstudio(hom_pose)
        elif self.slam_method == "orbslam3":
            hom_pose = pose_utils.ros_pose_to_homogenous(msg.pose)
            self.pose = orbslam3_to_nerfstudio(hom_pose)
        elif self.slam_method == "mocap":
            hom_pose = pose_utils.ros_pose_to_homogenous(msg.pose)
            self.pose = mocap_to_nerfstudio(hom_pose)
        elif self.slam_method == "zedsdk":
            hom_pose = pose_utils.ros_pose_to_homogenous(msg.pose)
            self.pose = cuvslam_to_nerfstudio(hom_pose)
        else:
            raise NameError("Unsupported SLAM algorithm. Must be one of {cuvslam, orbslam3, mocap, zedsdk}")
    def convert_pose_cuvslam(self, pose):
        # Convert pose using cuvslam_to_nerfstudio
        from nerfbridge.pose_utils import cuvslam_to_nerfstudio
        return cuvslam_to_nerfstudio(pose)

    def convert_pose_orbslam3(self, pose):
        # Convert pose using orbslam3_to_nerfstudio
        from nerfbridge.pose_utils import orbslam3_to_nerfstudio
        return orbslam3_to_nerfstudio(pose)