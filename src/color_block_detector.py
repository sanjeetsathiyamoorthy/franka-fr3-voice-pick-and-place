#!/usr/bin/env python3
"""
color_block_detector.py
=======================
Detects colored blocks using HSV filtering + ArUco homography.
Publishes per-color poses to:
  /red_pose    (geometry_msgs/Point)
  /green_pose  (geometry_msgs/Point)
  /blue_pose   (geometry_msgs/Point)

Works ALONGSIDE aruco_vision_detector.py — doesn't replace it.
color_selector_node will use these color-specific topics
instead of the /detected_block/pixel fallback.

Requires: the same 4 ArUco corner markers (id 0-3) that
aruco_vision_detector uses for homography calibration.
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import String


# ── Camera ─────────────────────────────────────────────────────────────────
CAMERA_INDEX = 2        # change if wrong — check ls /dev/video*

# ── ArUco ──────────────────────────────────────────────────────────────────
ARUCO_DICT   = aruco.DICT_4X4_50
MARKER_IDS   = [0, 1, 2, 3]   # corner markers — same as aruco_vision_detector

# Physical workspace corners in robot frame (metres)
# Order matches MARKER_IDS: [0=top-right, 1=bottom-right, 2=top-left, 3=bottom-left]
# These should match what aruco_vision_detector uses
WORKSPACE_ROBOT = np.float32([
    [0.50,  0.17],   # id=0 top-right
    [0.50, -0.17],   # id=1 bottom-right
    [0.25,  0.17],   # id=2 top-left
    [0.25, -0.17],   # id=3 bottom-left
])

# ── HSV color ranges ────────────────────────────────────────────────────────
# Tune these for your lighting conditions
# Hue: 0-180, Saturation: 0-255, Value: 0-255

COLOR_RANGES = {
    'red': [
        # Red wraps around 0/180 in HSV — need two ranges
        (np.array([0,   100, 80]),  np.array([10,  255, 255])),
        (np.array([165, 100, 80]),  np.array([180, 255, 255])),
    ],
    'green': [
        (np.array([35, 60, 60]),    np.array([85,  255, 255])),
    ],
    'blue': [
        (np.array([90, 60, 60]),    np.array([135, 255, 255])),
    ],
}

MIN_CONTOUR_AREA = 500   # pixels² — ignore tiny detections


class ColorBlockDetector(Node):

    def __init__(self):
        super().__init__('color_block_detector')

        # Publishers — one per color
        self._pubs = {
            'red':   self.create_publisher(Point, '/red_pose',   10),
            'green': self.create_publisher(Point, '/green_pose', 10),
            'blue':  self.create_publisher(Point, '/blue_pose',  10),
        }
        self._status_pub = self.create_publisher(String, '/color_detector_status', 10)

        # Homography matrix — computed from ArUco markers
        self._H = None

        # Camera
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            self.get_logger().error(
                f'Camera {CAMERA_INDEX} not available. '
                f'Run: ls /dev/video* and update CAMERA_INDEX'
            )
            return

        self._aruco_dict   = aruco.getPredefinedDictionary(ARUCO_DICT)
        self._aruco_params = aruco.DetectorParameters()

        # Timer — 10 Hz detection loop
        self.create_timer(0.1, self._detect)
        self.get_logger().info(
            f'ColorBlockDetector ready — camera={CAMERA_INDEX}, 10Hz'
        )

    # ── Detection loop ─────────────────────────────────────────────────────

    def _detect(self):
        ret, frame = self._cap.read()
        if not ret:
            return

        # Update homography from ArUco markers
        self._update_homography(frame)

        if self._H is None:
            self.get_logger().warn_once(
                'Homography not ready — waiting for all 4 ArUco markers'
            )
            return

        # Detect each color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for color, ranges in COLOR_RANGES.items():
            cx, cy = self._find_color_centroid(hsv, ranges)
            if cx is None:
                continue

            # Convert pixel → robot coordinates via homography
            pt_px  = np.float32([[[cx, cy]]])
            pt_rob = cv2.perspectiveTransform(pt_px, self._H)[0][0]

            msg   = Point()
            msg.x = float(pt_rob[0])
            msg.y = float(pt_rob[1])
            msg.z = 0.0
            self._pubs[color].publish(msg)
            self.get_logger().debug(
                f'{color}: pixel=({cx},{cy}) → robot=({msg.x:.3f},{msg.y:.3f})'
            )

    # ── ArUco homography ───────────────────────────────────────────────────

    def _update_homography(self, frame: np.ndarray):
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(
            gray, self._aruco_dict, parameters=self._aruco_params
        )

        if ids is None:
            return

        ids_flat = ids.flatten()
        src_pts  = []   # pixel corners
        dst_pts  = []   # robot frame corners

        for i, marker_id in enumerate(MARKER_IDS):
            if marker_id not in ids_flat:
                continue
            idx = np.where(ids_flat == marker_id)[0][0]
            # Use centre of ArUco marker
            c   = corners[idx][0]
            cx  = float(np.mean(c[:, 0]))
            cy  = float(np.mean(c[:, 1]))
            src_pts.append([cx, cy])
            dst_pts.append(WORKSPACE_ROBOT[i])

        if len(src_pts) >= 4:
            src = np.float32(src_pts)
            dst = np.float32(dst_pts)
            self._H, _ = cv2.findHomography(src, dst)

    # ── Color centroid ─────────────────────────────────────────────────────

    def _find_color_centroid(self, hsv, ranges):
        """
        Returns (cx, cy) pixel centroid of largest detected blob, or (None, None).
        """
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lo, hi) in ranges:
            mask |= cv2.inRange(hsv, lo, hi)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
            return None, None

        M  = cv2.moments(largest)
        if M['m00'] == 0:
            return None, None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return cx, cy

    def destroy_node(self):
        if self._cap.isOpened():
            self._cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ColorBlockDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
