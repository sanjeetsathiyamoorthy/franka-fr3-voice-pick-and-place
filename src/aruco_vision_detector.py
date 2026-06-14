#!/usr/bin/env python3
"""
aruco_vision_detector.py  —  Franka FR3 pick-and-place vision node
===================================================================
Refactored for dynamic ArUco bounds.
Publishes workspace limits and all detected block coordinates unconditionally.
"""

import math
from collections import deque

import cv2
import cv2.aruco as aruco
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

MARKER_INNER_CORNER_ROBOT = {
    0: np.array([0.3063,  0.1689]),
    1: np.array([0.4923,  0.1675]),
    2: np.array([0.3013, -0.3184]),
    3: np.array([0.4870, -0.3208]),
}

CUBE_Z_HEIGHT = 0.030

# ─────────────────────────────────────────────────────────────────────────────
# FIXED CONFIG
# ─────────────────────────────────────────────────────────────────────────────

ARUCO_DICT = aruco.DICT_4X4_50

MARKER_INNER_CORNER_IDX = {
    0: 3,
    1: 0,
    2: 2,
    3: 1,
}

MIN_CONTOUR_AREA = 400
H_MAX_AGE        = 30
CAMERA_INDEX     = 1

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT HSV VALUES  — tune with trackbars at runtime
# ─────────────────────────────────────────────────────────────────────────────

# Red (wraps around 0/180 in HSV — uses two ranges)
DEFAULT_RED_H_MIN, DEFAULT_RED_H_MAX = 0,   10
DEFAULT_RED_S_MIN, DEFAULT_RED_S_MAX = 120, 255
DEFAULT_RED_V_MIN, DEFAULT_RED_V_MAX = 80,  255

# Green
DEFAULT_GREEN_H_MIN, DEFAULT_GREEN_H_MAX = 40,  85
DEFAULT_GREEN_S_MIN, DEFAULT_GREEN_S_MAX = 60,  255
DEFAULT_GREEN_V_MIN, DEFAULT_GREEN_V_MAX = 60,  255

# Blue
DEFAULT_BLUE_H_MIN, DEFAULT_BLUE_H_MAX = 100, 130
DEFAULT_BLUE_S_MIN, DEFAULT_BLUE_S_MAX = 60,  255
DEFAULT_BLUE_V_MIN, DEFAULT_BLUE_V_MAX = 60,  255


# ─────────────────────────────────────────────────────────────────────────────
# NODE
# ─────────────────────────────────────────────────────────────────────────────

class ArucoVisionDetector(Node):

    def __init__(self):
        super().__init__('aruco_vision_detector')

        # /detected_block/pixel — red block, backward-compatible with real_pick_and_place
        self.pub = self.create_publisher(Point, '/detected_block/pixel', 10)

        # Per-color pose publishers
        self.pub_red   = self.create_publisher(Point,   '/red_pose',   10)
        self.pub_green = self.create_publisher(Point,   '/green_pose', 10)
        self.pub_blue  = self.create_publisher(Point,   '/blue_pose',  10)

        # Per-color yaw publishers — gripper rotation to match block tilt (radians)
        self.pub_red_yaw   = self.create_publisher(Float64, '/red_yaw',   10)
        self.pub_green_yaw = self.create_publisher(Float64, '/green_yaw', 10)
        self.pub_blue_yaw  = self.create_publisher(Float64, '/blue_yaw',  10)

        # Dynamic workspace bounds publisher
        self.pub_workspace = self.create_publisher(Float64MultiArray, '/workspace_bounds', 10)

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            self.get_logger().fatal(f'Cannot open camera index {CAMERA_INDEX}')
            raise RuntimeError('Camera not available')

        self.aruco_dict     = aruco.getPredefinedDictionary(ARUCO_DICT)
        self.aruco_params   = aruco.DetectorParameters()
        self.aruco_detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.H     = None
        self.H_age = 0

        # Per-colour circular buffers for angle smoothing (10 frames each).
        # Using a deque so old readings drop off automatically.
        self._angle_history = {
            'RED':   deque(maxlen=10),
            'GREEN': deque(maxlen=10),
            'BLUE':  deque(maxlen=10),
        }

        self._build_trackbars()
        self.timer = self.create_timer(1.0 / 30.0, self._process_frame)

        self.get_logger().info('aruco_vision_detector started — Dynamic Bounds Active')

    def _build_trackbars(self):
        # Red
        cv2.namedWindow('HSV Tuning', cv2.WINDOW_NORMAL)
        for name, val, mx in [
            ('H min', DEFAULT_RED_H_MIN, 179),
            ('H max', DEFAULT_RED_H_MAX, 179),
            ('S min', DEFAULT_RED_S_MIN, 255),
            ('S max', DEFAULT_RED_S_MAX, 255),
            ('V min', DEFAULT_RED_V_MIN, 255),
            ('V max', DEFAULT_RED_V_MAX, 255),
        ]:
            cv2.createTrackbar(name, 'HSV Tuning', val, mx, lambda _: None)

        # Green
        cv2.namedWindow('Green HSV', cv2.WINDOW_NORMAL)
        for name, val, mx in [
            ('H min', DEFAULT_GREEN_H_MIN, 179),
            ('H max', DEFAULT_GREEN_H_MAX, 179),
            ('S min', DEFAULT_GREEN_S_MIN, 255),
            ('S max', DEFAULT_GREEN_S_MAX, 255),
            ('V min', DEFAULT_GREEN_V_MIN, 255),
            ('V max', DEFAULT_GREEN_V_MAX, 255),
        ]:
            cv2.createTrackbar(name, 'Green HSV', val, mx, lambda _: None)

        # Blue
        cv2.namedWindow('Blue HSV', cv2.WINDOW_NORMAL)
        for name, val, mx in [
            ('H min', DEFAULT_BLUE_H_MIN,  179),
            ('H max', DEFAULT_BLUE_H_MAX,  179),
            ('S min', DEFAULT_BLUE_S_MIN,  255),
            ('S max', DEFAULT_BLUE_S_MAX,  255),
            ('V min', DEFAULT_BLUE_V_MIN,  255),
            ('V max', DEFAULT_BLUE_V_MAX,  255),
        ]:
            cv2.createTrackbar(name, 'Blue HSV', val, mx, lambda _: None)

    def _read_trackbars(self, window: str):
        g = lambda n: cv2.getTrackbarPos(n, window)
        return g('H min'), g('H max'), g('S min'), g('S max'), g('V min'), g('V max')

    def _update_homography(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is None:
            cv2.putText(frame, 'No ArUco markers detected',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 60, 255), 2)
            return False

        aruco.drawDetectedMarkers(frame, corners, ids)
        ids_flat = ids.flatten().tolist()
        required = list(MARKER_INNER_CORNER_ROBOT.keys())

        missing = [m for m in required if m not in ids_flat]
        if missing:
            cv2.putText(frame, f'Missing markers: {missing}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 60, 255), 2)
            return False

        pts_pixel = []
        pts_robot = []
        for mid in required:
            idx_in_result  = ids_flat.index(mid)
            marker_corners = corners[idx_in_result][0]
            inner_px       = marker_corners[MARKER_INNER_CORNER_IDX[mid]]
            pts_pixel.append(inner_px)
            pts_robot.append(MARKER_INNER_CORNER_ROBOT[mid])
            cv2.circle(frame, (int(inner_px[0]), int(inner_px[1])), 6, (0, 255, 0), -1)

        H, _ = cv2.findHomography(
            np.float32(pts_pixel), np.float32(pts_robot))
        if H is None:
            self.get_logger().warn('findHomography failed — skipping frame.')
            return False

        self.H     = H
        self.H_age = 0
        return True

    def _detect_color(self, frame, hsv, h_min, h_max, s_min, s_max, v_min, v_max,
                      is_red: bool = False, color_key: str = ''):
        # ── Mask ──────────────────────────────────────────────────────────────
        if is_red:
            # Red wraps around 0/180 in HSV — combine both halves
            mask = cv2.bitwise_or(
                cv2.inRange(hsv,
                            np.array([0,     s_min, v_min]),
                            np.array([h_max, s_max, v_max])),
                cv2.inRange(hsv,
                            np.array([170,   s_min, v_min]),
                            np.array([179,   s_max, v_max])),
            )
        else:
            mask = cv2.inRange(hsv,
                               np.array([h_min, s_min, v_min]),
                               np.array([h_max, s_max, v_max]))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

        # ── Contour ───────────────────────────────────────────────────────────
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, 0.0, mask

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
            return None, None, 0.0, mask

        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None, None, 0.0, mask

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # ── Angle — computed in robot frame ───────────────────────────────────
        # FIX 1: take the convex hull before fitting the rectangle.
        # Raw contours have jagged edges from HSV noise that skew the box fit.
        hull = cv2.convexHull(largest)
        rect = cv2.minAreaRect(hull)
        (cx_r, cy_r), (w, h), _ = rect
        box  = cv2.boxPoints(rect).astype(np.float32)

        # FIX 2: identify the LONG axis endpoints explicitly.
        # The old code relied on rect[2] which is ambiguous when width ≈ height
        # and whose sign convention changes depending on OpenCV version.
        e0_len = float(np.linalg.norm(box[1] - box[0]))
        e1_len = float(np.linalg.norm(box[2] - box[1]))
        if e0_len >= e1_len:
            p1_px, p2_px = box[0], box[1]
        else:
            p1_px, p2_px = box[1], box[2]

        # FIX 3: compute the angle IN ROBOT FRAME, not pixel frame.
        # Camera may be rotated/skewed relative to the robot X-axis — computing
        # atan2 in pixel space gives the wrong yaw to send to the gripper.
        # Transforming both long-axis endpoints through the homography and then
        # computing atan2 in robot coordinates corrects for any camera tilt.
        if self.H is not None and self.H_age <= H_MAX_AGE:
            wx1, wy1 = self._pixel_to_robot(float(p1_px[0]), float(p1_px[1]))
            wx2, wy2 = self._pixel_to_robot(float(p2_px[0]), float(p2_px[1]))
            raw_angle = math.atan2(wy2 - wy1, wx2 - wx1)
        else:
            # Homography not ready — fall back to pixel-space angle
            raw_angle = math.atan2(
                float(p2_px[1] - p1_px[1]),
                float(p2_px[0] - p1_px[0])
            )

        # FIX 4: normalise to [-π/4, +π/4]  (90° periodicity).
        # The blocks are cubes — square cross-section — so the gripper can
        # pick them equivalently at 0°, 90°, 180°, or 270°.  Folding into a
        # quarter-circle makes the result invariant to which edge of the
        # bounding box happens to be identified as "long".  Without this,
        # near-square hulls (where both edges are almost the same length) can
        # flip the axis selection frame-to-frame, producing a 90° jump in the
        # published yaw even though the block hasn't moved — the exact symptom
        # seen with the blue block whose arrow flipped 90° vs red and green.
        raw_angle = (raw_angle + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0

        # FIX 5: temporal smoothing with outlier rejection.
        # Single-frame angle estimates are noisy (lighting flicker, partial
        # occlusion, HSV boundary pixels).  We keep a 10-frame circular buffer
        # per colour and compute the circular mean.  Any frame whose angle
        # differs from the running mean by more than 25° is treated as a
        # spurious spike and discarded so it cannot corrupt the smooth value.
        angle_rad = raw_angle
        if color_key and color_key in self._angle_history:
            hist = self._angle_history[color_key]
            if hist:
                sin_m  = sum(math.sin(a) for a in hist) / len(hist)
                cos_m  = sum(math.cos(a) for a in hist) / len(hist)
                smooth = math.atan2(sin_m, cos_m)
                diff   = abs(math.atan2(
                    math.sin(raw_angle - smooth),
                    math.cos(raw_angle - smooth)
                ))
                if diff > math.radians(25):
                    # Outlier — keep the current smooth value, don't add to history
                    angle_rad = smooth
                else:
                    hist.append(raw_angle)
                    sin_m = sum(math.sin(a) for a in hist) / len(hist)
                    cos_m = sum(math.cos(a) for a in hist) / len(hist)
                    angle_rad = math.atan2(sin_m, cos_m)
            else:
                # First detection for this colour — seed the buffer
                hist.append(raw_angle)
                angle_rad = raw_angle

        # ── Visualisation ─────────────────────────────────────────────────────
        box_int = box.astype(np.int32)
        cv2.drawContours(frame, [box_int], -1, (0, 200, 255), 2)

        # Draw the long-axis arrow using pixel-space direction (cosmetic only)
        dx = float(p2_px[0] - p1_px[0])
        dy = float(p2_px[1] - p1_px[1])
        norm = math.hypot(dx, dy) or 1.0
        length = 35
        ex = int(cx + length * dx / norm)
        ey = int(cy + length * dy / norm)
        cv2.arrowedLine(frame, (cx, cy), (ex, ey), (255, 255, 0), 2, tipLength=0.3)

        # Show the robot-frame angle so you can verify it live
        cv2.putText(frame, f'{math.degrees(angle_rad):.1f}',
                    (cx + 12, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        return cx, cy, angle_rad, mask

    def _pixel_to_robot(self, cx, cy):
        world = cv2.perspectiveTransform(
            np.float32([[[cx, cy]]]), self.H)[0][0]
        return float(world[0]), float(world[1])

    def _try_publish(self, frame, cx, cy, angle_rad: float,
                     color_name: str, pub, yaw_pub, also_legacy=False):
        """Convert pixel to robot coords and publish unconditionally. Filtering moved downstream."""
        if self.H is None or self.H_age > H_MAX_AGE:
            return

        wx, wy = self._pixel_to_robot(cx, cy)
        
        # Log detected block coordinates
        self.get_logger().info(
            f'{color_name} block: ({wx:.3f},{wy:.3f})  yaw={math.degrees(angle_rad):.1f}°', 
            throttle_duration_sec=1.0)

        label = f'{color_name} X:{wx:.3f} Y:{wy:.3f}'
        cv2.putText(frame, label,
                    (cx + 10, cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        pt   = Point()
        pt.x = wx
        pt.y = wy
        pt.z = CUBE_Z_HEIGHT
        pub.publish(pt)

        yaw_msg = Float64()
        yaw_msg.data = float(angle_rad)
        yaw_pub.publish(yaw_msg)

        if also_legacy:
            self.pub.publish(pt)   # /detected_block/pixel backward compat

    def _process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Camera read failed.')
            return

        updated = self._update_homography(frame)
        if not updated:
            self.H_age += 1

        # DYNAMIC WORKSPACE BOUNDARIES
        msg_bounds = Float64MultiArray()
        if self.H is not None and self.H_age <= H_MAX_AGE:
            xs = [pt[0] for pt in MARKER_INNER_CORNER_ROBOT.values()]
            ys = [pt[1] for pt in MARKER_INNER_CORNER_ROBOT.values()]
            msg_bounds.data = [float(min(xs)), float(max(xs)), float(min(ys)), float(max(ys))]
        else:
            msg_bounds.data = [] # Empty array implies workspace not detected
        self.pub_workspace.publish(msg_bounds)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ── Red ───────────────────────────────────────────────────────────────
        rh_min, rh_max, rs_min, rs_max, rv_min, rv_max = self._read_trackbars('HSV Tuning')
        rcx, rcy, r_angle, red_mask = self._detect_color(
            frame, hsv, rh_min, rh_max, rs_min, rs_max, rv_min, rv_max,
            is_red=True, color_key='RED')
        if rcx is not None:
            cv2.circle(frame, (rcx, rcy), 8, (0, 0, 255), -1)
            self._try_publish(frame, rcx, rcy, r_angle, 'RED',
                              self.pub_red, self.pub_red_yaw, also_legacy=True)

        # ── Green ─────────────────────────────────────────────────────────────
        gh_min, gh_max, gs_min, gs_max, gv_min, gv_max = self._read_trackbars('Green HSV')
        gcx, gcy, g_angle, green_mask = self._detect_color(
            frame, hsv, gh_min, gh_max, gs_min, gs_max, gv_min, gv_max,
            color_key='GREEN')
        if gcx is not None:
            cv2.circle(frame, (gcx, gcy), 8, (0, 255, 0), -1)
            self._try_publish(frame, gcx, gcy, g_angle, 'GREEN',
                              self.pub_green, self.pub_green_yaw)

        # ── Blue ──────────────────────────────────────────────────────────────
        bh_min, bh_max, bs_min, bs_max, bv_min, bv_max = self._read_trackbars('Blue HSV')
        bcx, bcy, b_angle, blue_mask = self._detect_color(
            frame, hsv, bh_min, bh_max, bs_min, bs_max, bv_min, bv_max,
            color_key='BLUE')
        if bcx is not None:
            cv2.circle(frame, (bcx, bcy), 8, (255, 0, 0), -1)
            self._try_publish(frame, bcx, bcy, b_angle, 'BLUE',
                              self.pub_blue, self.pub_blue_yaw)

        # ── Status bar ────────────────────────────────────────────────────────
        if self.H is not None and self.H_age == 0:
            cv2.putText(frame, 'ArUco: OK — all 4 markers locked',
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        elif self.H is not None:
            cv2.putText(frame, f'ArUco: cached ({self.H_age} frames old)',
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)

        cv2.imshow('Camera Feed', frame)
        cv2.imshow('Red Mask',   red_mask   if rcx is not None else np.zeros(frame.shape[:2], np.uint8))
        cv2.imshow('Green Mask', green_mask if gcx is not None else np.zeros(frame.shape[:2], np.uint8))
        cv2.imshow('Blue Mask',  blue_mask  if bcx is not None else np.zeros(frame.shape[:2], np.uint8))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoVisionDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()