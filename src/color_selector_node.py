#!/usr/bin/env python3
"""
color_selector_node.py
======================
Refactored for Dynamic ArUco Bounds, Left/Right Placement, and Collision Avoidance.
Fixed: Race condition timing and occupancy logging clarity.

CHANGE — placement ordering fix:
  Previously the free-spot search scanned a fixed grid row-by-row (lowest Y
  first) and returned the very first conflict-free cell.  This caused the 3rd
  placed block to land beside the 1st block rather than the 2nd, because the
  scan found a cell that cleared both earlier blocks but was geometrically
  close to block 1.

  Fix: self._placed_positions tracks placement coordinates in order.  When at
  least one block has already been placed the candidate grid is sorted by
  distance to the LAST placed block before the conflict check runs.  The
  search therefore naturally prefers a spot near the most recently placed
  block, so each new block clusters beside the previous one.

  self._placed_positions is cleared on STOP and HOME commands.
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

        # FIX: tracks (x, y) placement coordinates in the order blocks were
        # placed.  Used to bias the free-spot search toward the last placed
        # block so successive blocks cluster together rather than scattering.
        self._placed_positions = []

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
            # FIX: clear placement history so a fresh session starts from
            # scratch rather than trying to cluster near a now-stale position.
            self._placed_positions.clear()
            self._speak('Stopping')
            self._pub_status('STOPPED')
            return
        if cmd == 'HOME':
            self._queue.clear()
            self._busy = False
            # FIX: same as STOP — reset placement history on home.
            self._placed_positions.clear()
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

        # ── FIX: Check if the block is already in the requested target zone ──
        if zone_ymin <= pose.y <= zone_ymax:
            self.get_logger().info(f'[{color}] block is already on the {side} side. Aborting redundant move.')
            self._speak(f'The {color.lower()} block is already on the {side.lower()} side.')
            self._release_and_next()
            return
        # ─────────────────────────────────────────────────────────────────────

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

        search_xmin = xmin + MARGIN
        search_xmax = xmax - MARGIN
        search_ymin = zone_ymin + MARGIN
        search_ymax = zone_ymax - MARGIN

        if side == 'LEFT':
            y_vals = list(self._frange_reverse(search_ymax, search_ymin, 0.02))
        else:
            y_vals = list(self._frange(search_ymin, search_ymax, 0.02))

        # Reversed so the search starts at large X (near markers id=1 and id=3,
        # far from the robot base) rather than small X (near id=0 and id=2,
        # close to the base and kinematically awkward to reach).
        x_vals = list(self._frange_reverse(search_xmax, search_xmin, 0.02))

        # ── FIX: build the full candidate list, then sort by proximity to the
        # last placed block.  This makes each successive block land as close as
        # possible to the previous one (e.g. 3rd block beside 2nd, not 1st)
        # rather than at whatever cell happens to be first in the raw grid.
        # When no block has been placed yet (_placed_positions is empty) the
        # candidates stay in the original row-major order, preserving the
        # existing behaviour for the very first placement.
        candidates = [(cx, cy) for cy in y_vals for cx in x_vals]

        # FIX: only anchor to blocks that were placed in the SAME zone.
        # Using a cross-zone block as anchor (e.g. right-zone block when
        # placing on the left) drags the search to the zone boundary instead
        # of starting near id=1/id=3 as intended.  Filter _placed_positions
        # to same-zone entries first; if none exist yet (first placement in
        # this zone), skip the sort entirely so the reversed-X default order
        # takes over and the block lands near id=1 (left) or id=3 (right).
        same_zone_placed = [
            p for p in self._placed_positions
            if search_ymin <= p[1] <= search_ymax
        ]
        if same_zone_placed:
            anchor = same_zone_placed[-1]
            candidates.sort(key=lambda pt: math.dist(pt, anchor))
            self.get_logger().info(
                f'[placement] Anchoring search near last same-zone block at '
                f'({anchor[0]:.3f}, {anchor[1]:.3f})'
            )
        else:
            self.get_logger().info(
                '[placement] No prior blocks in this zone — using default order (near id=1/id=3)'
            )
        # ─────────────────────────────────────────────────────────────────────

        best_spot = None
        for cx, cy in candidates:
            conflict = False
            for ox, oy, _ in obstacles:
                if math.dist((cx, cy), (ox, oy)) < CLEARANCE_RADIUS:
                    conflict = True
                    break
            if not conflict:
                best_spot = (cx, cy)
                break

        if not best_spot:
            self.get_logger().error('target zone full')
            self._speak(f'The {side.lower()} target zone is full.')
            self._release_and_next()
            return

        self.get_logger().info(f'Selected placement coordinate:\n({best_spot[0]:.3f}, {best_spot[1]:.3f})')

        # FIX: record the chosen placement so the next call can anchor to it.
        self._placed_positions.append(best_spot)
        self.get_logger().info(
            f'[placement] History ({len(self._placed_positions)} block(s)): '
            + ' → '.join(f'({x:.3f},{y:.3f})' for x, y in self._placed_positions)
        )

        self._speak(f'Picking {color.lower()}, placing on the {side.lower()}')

        drop_pt = Point(x=float(best_spot[0]), y=float(best_spot[1]), z=PLACE_Z_HEIGHT)

        self._place_pub.publish(drop_pt)
        self._publish_grasp_yaw(color)

        # Sleep guarantees Place and Yaw targets arrive and process BEFORE Pick triggers execution
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