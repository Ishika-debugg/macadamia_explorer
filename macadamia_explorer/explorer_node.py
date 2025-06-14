#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from sensor_msgs.msg import Image, LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped, Twist
import tf2_ros
import tf2_geometry_msgs
import cv2
from cv_bridge import CvBridge
import numpy as np

class MacadamiaExplorer(Node):
    def __init__(self):
        super().__init__('macadamia_explorer')

        self.bridge = CvBridge()
        self.scan = None

        self.sub_image = self.create_subscription(Image, '/oak/rgb/image_raw', self.image_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.marker_pub = self.create_publisher(Marker, '/tennis_ball_marker', 10)
        self.home_trigger_pub = self.create_publisher(Empty, '/trigger_home', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.detected_points = []
        self.detection_count = 0
        self.frame_count = 0

        self.green_lower = np.array([25, 100, 100])
        self.green_upper = np.array([40, 255, 255])

        self.other_color_ranges = [
            ([0, 100, 100], [10, 255, 255], "red"),
            ([170, 100, 100], [180, 255, 255], "red"),
            ([100, 100, 100], [130, 255, 255], "blue"),
            ([15, 100, 100], [25, 255, 255], "yellow"),
            ([10, 100, 100], [15, 255, 255], "orange"),
        ]

        self.state = 'ROW_NAVIGATION'
        self.last_avoid_time = self.get_clock().now()
        self.avoid_duration = 2.0

        self.get_logger().info('Macadamia Explorer node initialized!')

    def scan_callback(self, msg):
        self.scan = msg

    def image_callback(self, msg):
        if self.scan is None:
            self.get_logger().warn('Waiting for LiDAR scan data...')
            return

        try:
            self.frame_count += 1
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            green_ball = self.detect_green_tennis_ball(cv_image)

            if green_ball:
                self.process_macadamia_nut(cv_image, green_ball)
                if self.state == 'ROW_NAVIGATION':
                    self.get_logger().info('Nut detected! Initiating avoidance.')
                    self.state = 'AVOIDING_NUT'
                    self.last_avoid_time = self.get_clock().now()
                    self.avoid_nut()
            elif self.state == 'AVOIDING_NUT':
                now = self.get_clock().now()
                if (now - self.last_avoid_time).nanoseconds > self.avoid_duration * 1e9:
                    self.get_logger().info('Avoidance complete. Resuming row navigation.')
                    self.state = 'ROW_NAVIGATION'
                    self.resume_navigation()
            else:
                self.row_navigation()

        except Exception as e:
            self.get_logger().error(f'Error in macadamia nut detection: {str(e)}')

    def detect_green_tennis_ball(self, cv_image):
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area < 200:
            return None
        x, y, w, h = cv2.boundingRect(largest_contour)
        return {
            'contour': largest_contour,
            'center_x': x + w / 2,
            'center_y': y + h / 2,
            'area': area,
            'bbox': (x, y, w, h)
        }

    def process_macadamia_nut(self, cv_image, ball_data):
        center_x = ball_data['center_x']
        center_y = ball_data['center_y']
        area = ball_data['area']
        world_coords = self.get_world_coordinates(center_x, cv_image.shape[1])
        if world_coords:
            lidar_x, lidar_y, map_x, map_y = world_coords
            self.detection_count += 1
            self.get_logger().info('MACADAMIA NUT DETECTED!')
            point_lidar = PointStamped()
            point_lidar.header.frame_id = self.scan.header.frame_id
            point_lidar.header.stamp = self.scan.header.stamp
            point_lidar.point.x = lidar_x
            point_lidar.point.y = lidar_y
            point_lidar.point.z = 0.0
            self.transform_and_publish(point_lidar)

    def get_world_coordinates(self, center_x, image_width):
        try:
            normalized_x = center_x / image_width
            angle = self.scan.angle_min + normalized_x * (self.scan.angle_max - self.scan.angle_min)
            index = int((angle - self.scan.angle_min) / self.scan.angle_increment)
            if index < 0 or index >= len(self.scan.ranges):
                return None
            distance = self.scan.ranges[index]
            if not np.isfinite(distance) or not (self.scan.range_min <= distance <= self.scan.range_max):
                return None
            lidar_x = distance * np.cos(angle)
            lidar_y = distance * np.sin(angle)
            point_lidar = PointStamped()
            point_lidar.header.frame_id = self.scan.header.frame_id
            point_lidar.header.stamp = self.scan.header.stamp
            point_lidar.point.x = lidar_x
            point_lidar.point.y = lidar_y
            point_lidar.point.z = 0.0
            map_coords = self.transform_to_map(point_lidar)
            if map_coords:
                map_x, map_y = map_coords
                return (lidar_x, lidar_y, map_x, map_y)
            else:
                return (lidar_x, lidar_y, None, None)
        except Exception as e:
            self.get_logger().error(f'Error getting world coordinates: {str(e)}')
            return None

    def transform_to_map(self, point_lidar):
        try:
            timeout = rclpy.duration.Duration(seconds=0.1)
            if self.tf_buffer.can_transform('map', point_lidar.header.frame_id, rclpy.time.Time(), timeout=timeout):
                point_map = tf2_geometry_msgs.do_transform_point(
                    point_lidar,
                    self.tf_buffer.lookup_transform('map', point_lidar.header.frame_id, rclpy.time.Time(), timeout=timeout)
                )
                return (point_map.point.x, point_map.point.y)
            else:
                return None
        except Exception as e:
            self.get_logger().error(f'TF exception: {e}')
            return None

    def is_new_ball(self, new_point, threshold=0.3):
        for point in self.detected_points:
            dx = new_point.point.x - point.point.x
            dy = new_point.point.y - point.point.y
            if np.hypot(dx, dy) < threshold:
                return False
        return True

    def transform_and_publish(self, point_lidar):
        try:
            timeout = rclpy.duration.Duration(seconds=0.1)
            if self.tf_buffer.can_transform('map', point_lidar.header.frame_id, rclpy.time.Time(), timeout=timeout):
                point_map = tf2_geometry_msgs.do_transform_point(
                    point_lidar,
                    self.tf_buffer.lookup_transform('map', point_lidar.header.frame_id, rclpy.time.Time(), timeout=timeout)
                )
                if self.is_new_ball(point_map):
                    self.publish_marker(point_map)
                    self.detected_points.append(point_map)
                    self.home_trigger_pub.publish(Empty())
        except Exception as e:
            self.get_logger().error(f'TF exception: {e}')

    def publish_marker(self, point_map):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'macadamia_nuts'
        marker.id = len(self.detected_points)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = point_map.point
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.15
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.9
        self.marker_pub.publish(marker)

    def avoid_nut(self):
        twist = Twist()
        twist.linear.y = -0.2
        self.cmd_vel_pub.publish(twist)

    def resume_navigation(self):
        twist = Twist()
        twist.linear.x = 0.2
        self.cmd_vel_pub.publish(twist)

    def row_navigation(self):
        twist = Twist()
        twist.linear.x = 0.2
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = MacadamiaExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
