#!/usr/bin/env python3
"""
color_selector_node.py
======================
Refactored for Dynamic ArUco Bounds, Left/Right Placement, and Collision Avoidance.
Fixed: Race condition timing and occupancy logging clarity.
"""

import math
import subprocess
import threading
import time
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float64, Float64MultiArray
from geometry_msgs.msg import Point

SUPPORTED_COLORS    = {'RED', 'GREEN', 'BLUE', 'YELLOW'}
PLACE_Z_HEIGHT      = 0.027
CLEARANCE_RADIUS    = 0.08  # 8cm clearance between blocks
MARGIN              = 0.04  # 4cm margin from workspace edges
QUEUE_DELAY_SEC     = 30.0  # Time allowed for real robot motion

# espeak words per second
TTS_WPS = 3.0
TTS_BUFFER = 4.0

class ColorSelectorNode(Node):

    def __init__(self):
        super().__init__('color_selector_node')

        self._poses         = {c: None for c in SUPPORTED_COLORS}
        self._yaws          = {c: 0.0  for c in SUPPORTED_COLORS}
        self._workspace     = None  
        
        self._queue         = deque()
        self._busy          = False

        # Subscribers
        self.create_subscription(String, '/pick_command',  self._command_cb,  10)
        self.create_subscription(Float64MultiArray, '/workspace_bounds', self._workspace_cb, 10)
        
        self.create_subscription(Point,  '/red_pose',    lambda m: self._pose_cb('RED',    m), 10)
        self.create_subscription(Point,  '/green_pose',  lambda m: self._pose_cb('GREEN',  m), 10)
        self.create_subscription(Point,  '/blue_pose',   lambda m: self._pose_cb('BLUE',   m), 10)
        
        self.create_subscription(Float64, '/red_yaw',   lambda m: self._yaw_cb('RED',   m), 10)
        self.create_subscription(Float64, '/green_yaw', lambda m: self._yaw_cb('GREEN', m), 10)
        self.create_subscription(Float64, '/blue_yaw',  lambda m: self._yaw_cb('BLUE',  m), 10)

        # Publishers
        self._pick_pub       = self.create_publisher(Point,   '/voice_pick_target',  10)
        self._place_pub      = self.create_publisher(Point,   '/place_target',       10)
        self._home_pub       = self.create_publisher(Bool,    '/go_home',            10)
        self._tts_pub        = self.create_publisher(Float64, '/tts_cooldown',       10)
        self._status_pub     = self.create_publisher(String,  '/selector_status',    10)
        self._grasp_yaw_pub  = self.create_publisher(Float64, '/grasp_yaw',          10)

        self.get_logger().info('ColorSelectorNode ready — Dynamic Workspace & Shelf-Filling Active')
        self._speak('Robot ready')
        self._pub_status('IDLE')

    def _speak(self, text: str) -> None:
        words    = len(text.split())
        cooldown = max(words / TTS_WPS + TTS_BUFFER, 4.0)
        msg = Float64()
        msg.data = cooldown
        self._tts_pub.publish(msg)
        try:
            subprocess.Popen(['espeak', '-s', '145', '-v', 'en', text],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            pass
        self.get_logger().info(f'TTS: "{text}" (cooldown={cooldown:.1f}s)')

    def _pub_status(self, status: str):
        msg = String()
        msg.data = status
        self._status_pub.publish(msg)

    def _publish_grasp_yaw(self, color: str) -> None:
        yaw = self._yaws.get(color, 0.0)
        msg = Float64()
        msg.data = float(yaw)
        self._grasp_yaw_pub.publish(msg)

    def _workspace_cb(self, msg: Float64MultiArray):
        if len(msg.data) == 4:
            self._workspace = msg.data
        else:
            self._workspace = None

    def _pose_cb(self, color: str, msg: Point):
        self._poses[color] = msg

    def _yaw_cb(self, color: str, msg: Float64):
        self._yaws[color] = msg.data

    def _command_cb(self, msg: String):
        cmd = msg.data.strip().upper()
        self.get_logger().info(f'Received: {cmd}')

        if cmd == 'STOP':
            self._queue.clear()
            self._busy = False
            self._speak('Stopping')
            self._pub_status('STOPPED')
            return
        if cmd == 'HOME':
            self._queue.clear()
            self._busy = False
            self._speak('Returning to home position')
            self._home_pub.publish(Bool(data=True))
            self._pub_status('HOME')
            return
        if cmd == 'STATUS':
            queue_info = f'{len(self._queue)} queued' if self._queue else 'queue empty'
            busy       = 'busy' if self._busy else 'idle'
            self.get_logger().info(f'Status: {busy} | {queue_info}')
            self._speak(f'System {busy}. {queue_info}.')
            return

        self._queue.append(cmd)
        self._process_next()

    def _process_next(self):
        if self._busy or not self._queue:
            return
        cmd = self._queue.popleft()
        self._busy = True
        self._execute(cmd)

    def _release_and_next(self):
        self._busy = False
        self._process_next()

    def _execute(self, cmd: str):
        parts = cmd.split('_')
        if cmd.startswith('PICK_') and len(parts) >= 3:
            color, side = parts[1], parts[2]
            self._handle_pick_and_place(color, side)
        elif cmd.startswith('PICK_'):
            color = parts[1]
            self._speak(f'Picking {color.lower()}, but no side specified.')
            self._release_and_next()
        else:
            self.get_logger().warn(f'Unhandled command format: {cmd}')
            self._release_and_next()

    def _frange(self, start, stop, step):
        r = start
        while r <= stop + 1e-5:
            yield r
            r += step

    def _frange_reverse(self, start, stop, step):
        r = start
        while r >= stop - 1e-5:
            yield r
            r -= step

    def _handle_pick_and_place(self, color: str, side: str):
        if not self._workspace:
            self.get_logger().error('workspace not detected')
            self._speak('Workspace not detected. Cannot proceed.')
            self._release_and_next()
            return

        xmin, xmax, ymin, ymax = self._workspace
        self.get_logger().info(f'Workspace: xmin={xmin:.3f} xmax={xmax:.3f} ymin={ymin:.3f} ymax={ymax:.3f}')

        pose = self._poses.get(color)
        if pose is None:
            self.get_logger().error(f'Cannot see the {color.lower()} block')
            self._speak(f'Cannot see the {color.lower()} block')
            self._release_and_next()
            return

        if not (xmin <= pose.x <= xmax and ymin <= pose.y <= ymax):
            self.get_logger().error('block out of boundaries')
            self._speak(f'The {color.lower()} block is out of boundaries.')
            self._release_and_next()
            return
        
        self.get_logger().info('Block inside workspace')

        mid_y = (ymin + ymax) / 2.0
        if side == 'LEFT':
            zone_ymin, zone_ymax = mid_y, ymax
            self.get_logger().info(f'Left zone limits: Y from {zone_ymin:.3f} to {zone_ymax:.3f}')
        else:
            zone_ymin, zone_ymax = ymin, mid_y
            self.get_logger().info(f'Right zone limits: Y from {zone_ymin:.3f} to {zone_ymax:.3f}')

        obstacles = []
        for c, p in self._poses.items():
            if c != color and p is not None:
                if xmin <= p.x <= xmax and ymin <= p.y <= ymax:
                    obstacles.append((p.x, p.y, c))
                    
        # Filter visually for the terminal log only
        zone_obs = [f"{c} at ({px:.3f},{py:.3f})" for px, py, c in obstacles if zone_ymin <= py <= zone_ymax]
        if zone_obs:
            self.get_logger().info(f'{side.capitalize()} zone occupied by: {", ".join(zone_obs)}')
        else:
            self.get_logger().info(f'{side.capitalize()} zone is empty')

        best_spot = None
        search_xmin = xmin + MARGIN
        search_xmax = xmax - MARGIN
        search_ymin = zone_ymin + MARGIN
        search_ymax = zone_ymax - MARGIN

        if side == 'LEFT':
            y_vals = list(self._frange_reverse(search_ymax, search_ymin, 0.02))
        else:
            y_vals = list(self._frange(search_ymin, search_ymax, 0.02))

        x_vals = list(self._frange(search_xmin, search_xmax, 0.02))

        for cy in y_vals:
            for cx in x_vals:
                conflict = False
                for ox, oy, _ in obstacles:
                    if math.dist((cx, cy), (ox, oy)) < CLEARANCE_RADIUS:
                        conflict = True
                        break
                if not conflict:
                    best_spot = (cx, cy)
                    break
            if best_spot:
                break

        if not best_spot:
            self.get_logger().error('target zone full')
            self._speak(f'The {side.lower()} target zone is full.')
            self._release_and_next()
            return

        self.get_logger().info(f'Selected placement coordinate:\n({best_spot[0]:.3f}, {best_spot[1]:.3f})')
        self._speak(f'Picking {color.lower()}, placing on the {side.lower()}')

        drop_pt = Point(x=float(best_spot[0]), y=float(best_spot[1]), z=PLACE_Z_HEIGHT)
        
        self._place_pub.publish(drop_pt)
        self._publish_grasp_yaw(color)
        
        # FIX: Sleep guarantees Place and Yaw targets arrive and process BEFORE Pick triggers execution
        time.sleep(0.15)
        
        self._pick_pub.publish(pose)
        
        t = threading.Timer(QUEUE_DELAY_SEC, self._release_and_next)
        t.daemon = True
        t.start()

def main(args=None):
    rclpy.init(args=args)
    node = ColorSelectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()