#!/usr/bin/env python3
"""
real_pick_and_place.py  —  fr3_delivery_sim  (REAL Franka Research 3 hardware)
=============================================================================
Updated for voice-controlled pick-and-place pipeline.

Three execution modes (all triggered by color_selector_node):
  1. Full pick+place : /voice_pick_target + /place_target  → run()
  2. Pick only       : /pick_only_target                   → run_pick_only()
  3. Place only      : /place_only_target                  → run_place_only()
  4. Zone retrieval  : /zone_pick_target + /place_target   → run()  (no bounds check)
  5. Go home         : /go_home                            → go_home()
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from moveit_msgs.action import MoveGroup as MoveGroupAction
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.msg import (
    MotionPlanRequest, RobotState, Constraints, JointConstraint,
    PositionConstraint, OrientationConstraint, BoundingVolume,
    CollisionObject, PlanningScene,
)
from shape_msgs.msg import SolidPrimitive, Plane
from control_msgs.action import FollowJointTrajectory

from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, String
from std_msgs.msg import Header
from builtin_interfaces.msg import Duration

try:
    from franka_msgs.action import Grasp, Move, Homing
    _FRANKA_GRIPPER_AVAILABLE = True
except ImportError:
    try:
        from franka_gripper.action import Grasp, Move, Homing
        _FRANKA_GRIPPER_AVAILABLE = True
    except ImportError:
        _FRANKA_GRIPPER_AVAILABLE = False

# ── Constants ──────────────────────────────────────────────────────────────────
ARM_JOINTS = [
    'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
    'fr3_joint5', 'fr3_joint6', 'fr3_joint7',
]
HOME = [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]

MOVEIT_SUCCESS = 1

# Camera workspace bounds — blocks picked from vision must be within these
BLOCK_X_MIN, BLOCK_X_MAX =  0.25,  0.55
BLOCK_Y_MIN, BLOCK_Y_MAX = -0.35,  0.20

# ── Behind-robot zone handling ─────────────────────────────────────────────────
BEHIND_ROBOT_X  = 0.0    
BEHIND_EXTRA_Z  = 0.25   
BEHIND_OVERHEAD = (0.07, 0.0, 0.52)  


def quaternion_from_euler(roll, pitch, yaw):
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (qx, qy, qz, qw)


class RealPickAndPlace(Node):

    def __init__(self):
        super().__init__('real_pick_and_place')

        self.declare_parameter('group_name',     'fr3_arm')
        self.declare_parameter('planning_frame', 'fr3_link0')
        self.declare_parameter('ee_link',        'fr3_hand_tcp')
        self.declare_parameter('block_topic',    '/voice_pick_target')
        self.declare_parameter('arm_action',     '/fr3_arm_controller/follow_joint_trajectory')
        self.declare_parameter('gripper_ns',     '/franka_gripper')
        self.declare_parameter('table_z',         0.0)
        self.declare_parameter('cube_half',       0.030)
        self.declare_parameter('approach_height', 0.15)
        self.declare_parameter('lift_height',     0.20)
        self.declare_parameter('grasp_z',         0.025)
        self.declare_parameter('grasp_yaw',       0.0)
        self.declare_parameter('grasp_x_offset',  0.016)
        self.declare_parameter('grasp_y_offset',  0.003)
        self.declare_parameter('drop_x',          0.3916)
        self.declare_parameter('drop_y',          0.1742)
        self.declare_parameter('drop_z',          0.025)
        self.declare_parameter('return_home',     True)
        self.declare_parameter('gripper_open_pos',        0.08)
        self.declare_parameter('gripper_close_pos',       0.045)
        self.declare_parameter('gripper_speed',           0.05)
        self.declare_parameter('gripper_grasp_force',     40.0)
        self.declare_parameter('gripper_epsilon_inner',   0.03)
        self.declare_parameter('gripper_epsilon_outer',   0.03)
        self.declare_parameter('gripper_homing_on_start', True)
        self.declare_parameter('gripper_timeout',         10.0)
        self.declare_parameter('vel_scale',               0.15)
        self.declare_parameter('acc_scale',               0.15)
        self.declare_parameter('planning_time',           10.0)
        self.declare_parameter('detection_timeout',       300.0)
        self.declare_parameter('cart_speed',              0.03)
        self.declare_parameter('cartesian_step',          0.005)
        self.declare_parameter('min_cartesian_fraction',  0.5)

        gp = self.get_parameter
        self.group_name       = gp('group_name').value
        self.planning_frame   = gp('planning_frame').value
        self.ee_link          = gp('ee_link').value
        self.arm_action       = gp('arm_action').value
        self.gripper_ns       = gp('gripper_ns').value.rstrip('/')
        self.table_z          = gp('table_z').value
        self.cube_half        = gp('cube_half').value
        self.approach_height  = gp('approach_height').value
        self.lift_height      = gp('lift_height').value
        self.grasp_z          = gp('grasp_z').value
        self.grasp_yaw        = gp('grasp_yaw').value
        self.grasp_x_offset   = gp('grasp_x_offset').value
        self.grasp_y_offset   = gp('grasp_y_offset').value
        self.drop             = (gp('drop_x').value, gp('drop_y').value, gp('drop_z').value)
        self.return_home      = gp('return_home').value
        self.grip_open        = gp('gripper_open_pos').value
        self.grip_close       = gp('gripper_close_pos').value
        self.grip_speed       = gp('gripper_speed').value
        self.grip_force       = gp('gripper_grasp_force').value
        self.grip_eps_in      = gp('gripper_epsilon_inner').value
        self.grip_eps_out     = gp('gripper_epsilon_outer').value
        self.grip_home_start  = gp('gripper_homing_on_start').value
        self.grip_timeout     = gp('gripper_timeout').value
        self.vel_scale        = gp('vel_scale').value
        self.acc_scale        = gp('acc_scale').value
        self.planning_time    = gp('planning_time').value
        self.detection_timeout= gp('detection_timeout').value
        self.cart_speed       = gp('cart_speed').value
        self.cartesian_step   = gp('cartesian_step').value
        self.min_fraction     = gp('min_cartesian_fraction').value

        self._latest_joints         = None
        self._grasp_yaw_override    = None
        self._latest_block_pose     = None
        self._block_pose_stamp      = 0.0
        self._skip_bounds_check     = False
        self._custom_grasp_z        = None
        self._pick_only_pose        = None
        self._pick_only_stamp       = 0.0
        self._pick_only_skip_bounds = False
        self._pick_only_grasp_z     = None
        self._place_only_pose       = None
        self._place_only_stamp      = 0.0
        self._is_holding            = False
        self._busy                  = False

        # State Publisher
        self._state_pub = self.create_publisher(String, "/robot_state", 10)

        self.create_subscription(Point, gp('block_topic').value, self._block_cb, 10)
        self.create_subscription(Point, '/zone_pick_target', self._zone_pick_cb, 10)
        self.create_subscription(Point, '/pick_only_target', self._pick_only_cb, 10)
        self.create_subscription(Point, '/place_only_target', self._place_only_cb, 10)
        self.create_subscription(JointState, '/joint_states', self._joints_cb, 10)
        self.create_subscription(Point, '/place_target', self._place_target_cb, 10)
        self.create_subscription(Bool, '/go_home', self._go_home_cb, 10)
        self.create_subscription(Float64, '/grasp_yaw', self._grasp_yaw_cb, 10)

        self._move_client = ActionClient(self, MoveGroupAction, '/move_action')
        self._jtc_client  = ActionClient(self, FollowJointTrajectory, self.arm_action)
        self._cart_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')

        self._grasp_client  = None
        self._move_g_client = None
        self._homing_client = None
        if _FRANKA_GRIPPER_AVAILABLE:
            self._grasp_client  = ActionClient(self, Grasp,  f'{self.gripper_ns}/grasp')
            self._move_g_client = ActionClient(self, Move,   f'{self.gripper_ns}/move')
            self._homing_client = ActionClient(self, Homing, f'{self.gripper_ns}/homing')
        else:
            self.get_logger().error('franka_msgs.action not found — Gripper steps SKIPPED.')

        self.get_logger().info('Waiting for MoveIt (/move_action) + arm controller...')
        self._move_client.wait_for_server()
        self._jtc_client.wait_for_server()
        self._cart_client.wait_for_service()

        if self._grasp_client is not None:
            self.get_logger().info(f'Waiting for gripper under {self.gripper_ns} ...')
            self._grasp_client.wait_for_server()
            self._move_g_client.wait_for_server()
            self._homing_client.wait_for_server()

        self._scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 1)
        self._setup_collision_scene()
        
        # Initialize state to HOME
        self.publish_state("HOME")

    def publish_state(self, state: str):
        """Helper to publish robot status to Whisper nodes."""
        msg = String()
        msg.data = state
        self._state_pub.publish(msg)

    def _setup_collision_scene(self):
        scene = PlanningScene()
        scene.is_diff = True

        table       = CollisionObject()
        table.id    = 'table'
        table.header.frame_id = self.planning_frame
        table_box         = SolidPrimitive()
        table_box.type    = SolidPrimitive.BOX
        table_box.dimensions = [2.0, 2.0, 0.05]
        table_pose = Pose()
        table_pose.position.x    = 0.0
        table_pose.position.y    = 0.0
        table_pose.position.z    = self.table_z - 0.025
        table_pose.orientation.w = 1.0
        table.primitives.append(table_box)
        table.primitive_poses.append(table_pose)
        table.operation = CollisionObject.ADD

        floor       = CollisionObject()
        floor.id    = 'floor'
        floor.header.frame_id = self.planning_frame
        floor_box         = SolidPrimitive()
        floor_box.type    = SolidPrimitive.BOX
        floor_box.dimensions = [4.0, 4.0, 0.02]
        floor_pose = Pose()
        floor_pose.position.x    = 0.0
        floor_pose.position.y    = 0.0
        floor_pose.position.z    = self.table_z - 0.12
        floor_pose.orientation.w = 1.0
        floor.primitives.append(floor_box)
        floor.primitive_poses.append(floor_pose)
        floor.operation = CollisionObject.ADD

        scene.world.collision_objects.append(table)
        scene.world.collision_objects.append(floor)

        time.sleep(0.5)
        self._scene_pub.publish(scene)
        time.sleep(0.3)
        self._scene_pub.publish(scene)
        self.get_logger().info('Collision scene published.')

    def _block_cb(self, msg: Point):
        if self._busy: return
        self._latest_block_pose = (msg.x, msg.y)
        self._block_pose_stamp  = time.time()
        self._skip_bounds_check = False
        self._custom_grasp_z    = None
        self.get_logger().info(f'[voice] Pick+place → ({msg.x:.3f}, {msg.y:.3f})')

    def _zone_pick_cb(self, msg: Point):
        if self._busy: return
        self._latest_block_pose = (msg.x, msg.y)
        self._block_pose_stamp  = time.time()
        self._skip_bounds_check = True
        self._custom_grasp_z    = msg.z if msg.z > 0.001 else None

    def _pick_only_cb(self, msg: Point):
        if self._busy: return
        self._pick_only_pose        = (msg.x, msg.y)
        self._pick_only_stamp       = time.time()
        self._pick_only_skip_bounds = msg.z < -0.5
        self._pick_only_grasp_z     = abs(msg.z) if msg.z < -0.001 else None

    def _place_only_cb(self, msg: Point):
        if self._busy: return
        self._place_only_pose  = (msg.x, msg.y, msg.z)
        self._place_only_stamp = time.time()

    def _joints_cb(self, msg: JointState):
        self._latest_joints = msg

    def _place_target_cb(self, msg: Point):
        z = msg.z if msg.z > 0.001 else self.drop[2]
        self.drop = (msg.x, msg.y, z)
        self.get_logger().info(f'[zone] Drop target → x={msg.x:.3f} y={msg.y:.3f} z={z:.3f}')

    def _go_home_cb(self, msg: Bool):
        if msg.data:
            self.go_home()

    def _grasp_yaw_cb(self, msg: Float64):
        self._grasp_yaw_override = msg.data
        self.get_logger().info(f'[grasp_yaw] Block tilt received: {math.degrees(msg.data):.1f}°')

    def _wait_for(self, predicate, timeout, what):
        deadline = time.time() + timeout
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if predicate(): return True
        return False

    def _settle(self, seconds=1.0):
        deadline = time.time() + seconds
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        return True

    def _send_action(self, client, goal, label):
        if client is None or not client.server_is_ready(): return False
        send_fut = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut, timeout_sec=self.grip_timeout)
        gh = send_fut.result()
        if gh is None or not gh.accepted: return False
        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=self.grip_timeout)
        result = res_fut.result()
        if result is None: return False
        success = getattr(result.result, 'success', True)
        if success:
            self.get_logger().info(f'[gripper] {label}: done.')
            return True
        return False

    def home_gripper(self):
        return self._send_action(self._homing_client, Homing.Goal(), 'homing')

    def open_gripper(self):
        goal = Move.Goal()
        goal.width = float(self.grip_open)
        goal.speed = float(self.grip_speed)
        return self._send_action(self._move_g_client, goal, 'open')

    def close_gripper(self):
        goal = Grasp.Goal()
        goal.width = float(self.grip_close)
        goal.speed = float(self.grip_speed)
        goal.force = float(self.grip_force)
        goal.epsilon.inner = float(self.grip_eps_in)
        goal.epsilon.outer = float(self.grip_eps_out)
        return self._send_action(self._grasp_client, goal, 'grasp')

    def localize_block(self, pose_xy=None, skip_bounds=False, custom_gz=None):
        xy = pose_xy if pose_xy is not None else self._latest_block_pose
        if xy is None: return None
        x, y = xy
        x += self.grasp_x_offset
        y += self.grasp_y_offset
        gz = custom_gz if custom_gz else (self.table_z + self.cube_half)
        return (x, y, gz)

    def _down_quat(self, yaw: float = None) -> Quaternion:
        effective_yaw = yaw if yaw is not None else self.grasp_yaw
        x, y, z, w = quaternion_from_euler(math.pi, 0.0, effective_yaw)
        q = Quaternion()
        q.x, q.y, q.z, q.w = x, y, z, w
        return q

    def _pose_constraints(self, xyz, quat: Quaternion):
        c = Constraints()
        pc = PositionConstraint()
        pc.header.frame_id = self.planning_frame
        pc.link_name = self.ee_link
        pc.target_point_offset = Vector3()
        bv = BoundingVolume()
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.01]
        bv.primitives.append(sphere)
        region = Pose()
        region.position.x = float(xyz[0])
        region.position.y = float(xyz[1])
        region.position.z = float(xyz[2])
        region.orientation.w = 1.0
        bv.primitive_poses.append(region)
        pc.constraint_region = bv
        pc.weight = 1.0
        c.position_constraints.append(pc)
        oc = OrientationConstraint()
        oc.header.frame_id = self.planning_frame
        oc.link_name = self.ee_link
        oc.orientation = quat
        oc.absolute_x_axis_tolerance = 0.15
        oc.absolute_y_axis_tolerance = 0.15
        oc.absolute_z_axis_tolerance = 0.15
        oc.weight = 1.0
        c.orientation_constraints.append(oc)
        return c

    def _joint_constraints(self, joints):
        c = Constraints()
        for name, pos in zip(ARM_JOINTS, joints):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        return c

    def _plan_goal(self, constraints, extra_planning_time: float = 0.0):
        req = MotionPlanRequest()
        req.group_name = self.group_name
        req.allowed_planning_time = self.planning_time + extra_planning_time
        req.max_velocity_scaling_factor = self.vel_scale
        req.max_acceleration_scaling_factor = self.acc_scale
        req.goal_constraints.append(constraints)
        req.num_planning_attempts = 5
        req.workspace_parameters.header.frame_id = self.planning_frame
        req.workspace_parameters.min_corner.x = -1.2
        req.workspace_parameters.min_corner.y = -1.2
        req.workspace_parameters.min_corner.z = self.table_z
        req.workspace_parameters.max_corner.x =  1.2
        req.workspace_parameters.max_corner.y =  1.2
        req.workspace_parameters.max_corner.z =  1.5
        goal = MoveGroupAction.Goal()
        goal.request = req
        goal.planning_options.plan_only = True
        fut = self._move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        gh = fut.result()
        if gh is None or not gh.accepted: return None
        rfut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rfut, timeout_sec=self.planning_time + 5.0)
        if not rfut.done(): return None
        res = rfut.result().result
        if res.error_code.val != MOVEIT_SUCCESS: return None
        return res.planned_trajectory.joint_trajectory

    def move_to_pose(self, xyz, label='pose', tries=3, yaw: float = None, extra_time: float = 0.0):
        goal_c = self._pose_constraints(xyz, self._down_quat(yaw))
        for attempt in range(1, tries + 1):
            self.get_logger().info(f'--> {label} {tuple(round(c, 3) for c in xyz)}')
            traj = self._plan_goal(goal_c, extra_planning_time=extra_time)
            if traj is not None: return self._execute(traj)
            self._settle(0.5)
        return False

    def move_to_joints(self, joints, label='home'):
        self.get_logger().info(f'--> {label} (joint space)')
        traj = self._plan_goal(self._joint_constraints(joints))
        return self._execute(traj)

    def go_home(self):
        self.publish_state("BUSY")
        self.get_logger().info(f'Planning home position: {HOME}')
        traj = self._plan_goal(self._joint_constraints(HOME))
        if traj is not None: self._execute(traj)
        self.publish_state("HOME")

    def cartesian_move(self, xyz, travel_dist, label, yaw: float = None):
        if self._cart_client.service_is_ready():
            traj = self._plan_cartesian(xyz, travel_dist, yaw)
            if traj is not None:
                self.get_logger().info(f'--> {label} (cartesian)')
                return self._execute(traj)
        return self.move_to_pose(xyz, label, yaw=yaw)

    def _plan_cartesian(self, xyz, travel_dist, yaw: float = None):
        target = Pose()
        target.position.x = float(xyz[0])
        target.position.y = float(xyz[1])
        target.position.z = float(xyz[2])
        target.orientation = self._down_quat(yaw)
        req = GetCartesianPath.Request()
        req.header.frame_id = self.planning_frame
        req.group_name = self.group_name
        req.link_name = self.ee_link
        req.waypoints = [target]
        req.max_step = self.cartesian_step
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        if self._latest_joints is not None:
            rs = RobotState()
            rs.joint_state = self._latest_joints
            rs.is_diff = False
            req.start_state = rs
        fut = self._cart_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        resp = fut.result()
        if resp is None or resp.fraction < self.min_fraction: return None
        traj = resp.solution.joint_trajectory
        if not traj.points: return None
        return self._retime(traj, abs(travel_dist))

    def _retime(self, traj, distance_m):
        n = len(traj.points)
        total = max(distance_m / max(self.cart_speed, 1e-3), 1.0)
        for i, pt in enumerate(traj.points):
            t = total * (i / (n - 1)) if n > 1 else total
            pt.time_from_start = Duration(sec=int(t), nanosec=int((t - int(t)) * 1e9))
            pt.velocities = []
            pt.accelerations = []
            pt.effort = []
        return traj

    def _execute(self, traj):
        if traj is None or not traj.points: return False
        traj.header.stamp.sec = 0
        traj.header.stamp.nanosec = 0
        jtc_goal = FollowJointTrajectory.Goal()
        jtc_goal.trajectory = traj
        fut = self._jtc_client.send_goal_async(jtc_goal)
        rclpy.spin_until_future_complete(self, fut)
        gh = fut.result()
        if gh is None or not gh.accepted: return False
        rfut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rfut)
        code = rfut.result().result.error_code
        if code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info('Execution complete.')
            return True
        return False

    def _run_steps(self, steps):
        for name, action in steps:
            self.get_logger().info(f'=== STEP: {name} ===')
            if not action(): return False
        return True

    def run(self):
        if self._busy: return
        self._busy = True
        self.publish_state("BUSY")
        
        try:
            self.get_logger().info('=== real_pick_and_place starting ===')
            self._wait_for(lambda: self._latest_joints is not None, 5.0, 'joint_states')

            if self.grip_home_start and self._homing_client is not None:
                self.home_gripper()

            block = self.localize_block(pose_xy=self._latest_block_pose, skip_bounds=self._skip_bounds_check, custom_gz=self._custom_grasp_z)
            if block is None: return

            bx, by, _ = block
            drop = self.drop
            behind_robot = drop[0] < BEHIND_ROBOT_X
            transit_z = (self.table_z + self.cube_half + self.lift_height + (BEHIND_EXTRA_Z if behind_robot else 0.0))

            hover = (bx, by, self.table_z + self.cube_half + self.approach_height)
            grasp = (bx, by, self.grasp_z)
            lift = (bx, by, transit_z)
            drop_above = (drop[0], drop[1], transit_z)

            pick_yaw = self._grasp_yaw_override
            self._grasp_yaw_override = None

            pick_steps = [
                ('1. Hover above block',  lambda: self.move_to_pose(hover, 'hover', yaw=pick_yaw)),
                ('   settle',             lambda: self._settle(0.8)),
                ('2. Open gripper',       self.open_gripper),
                ('   settle',             lambda: self._settle(0.6)),
                ('3. Descend to grasp',   lambda: self.cartesian_move(grasp, hover[2] - grasp[2], 'descend', yaw=pick_yaw)),
                ('   settle',             lambda: self._settle(0.5)),
                ('4. Grasp block',        self.close_gripper),
                ('   settle',             lambda: self._settle(0.6)),
                ('5. Lift block',         lambda: self.cartesian_move(lift, transit_z - grasp[2], 'lift', yaw=pick_yaw)),
                ('   settle',             lambda: self._settle(0.5)),
            ]

            if behind_robot:
                place_steps = [
                    ('6. Via-point (overhead)',   lambda: self.move_to_pose(BEHIND_OVERHEAD, 'overhead-via', extra_time=5.0)),
                    ('   settle',                 lambda: self._settle(0.6)),
                    ('7. Transfer to drop',       lambda: self.move_to_pose(drop_above, 'above-drop', extra_time=5.0)),
                    ('   settle',                 lambda: self._settle(0.6)),
                    ('8. Place down (jnt-space)', lambda: self.move_to_pose(drop, 'place', extra_time=5.0)),
                    ('   settle',                 lambda: self._settle(0.5)),
                    ('9. Open gripper',           self.open_gripper),
                    ('   settle',                 lambda: self._settle(0.6)),
                    ('10. Retreat up',            lambda: self.move_to_pose(drop_above, 'retreat-up', extra_time=5.0)),
                    ('    settle',                lambda: self._settle(0.4)),
                    ('11. Via-point (return)',    lambda: self.move_to_pose(BEHIND_OVERHEAD, 'overhead-return', extra_time=5.0)),
                ]
            else:
                place_steps = [
                    ('6. Transfer to drop',  lambda: self.move_to_pose(drop_above, 'above-drop')),
                    ('   settle',            lambda: self._settle(0.6)),
                    ('7. Place down',        lambda: self.cartesian_move(drop, transit_z - drop[2], 'place')),
                    ('   settle',            lambda: self._settle(0.5)),
                    ('8. Open gripper',      self.open_gripper),
                    ('   settle',            lambda: self._settle(0.6)),
                    ('9. Retreat up',        lambda: self.cartesian_move(drop_above, transit_z - drop[2], 'retreat')),
                ]

            steps = pick_steps + place_steps
            if self.return_home:
                steps.append(('Return home', lambda: self.move_to_joints(HOME, 'home')))

            if self._run_steps(steps):
                self.get_logger().info('=== Pick-and-place complete ===')

        finally:
            self._busy = False
            self.publish_state("HOME")

    def run_pick_only(self):
        if self._busy: return
        self._busy = True
        self.publish_state("BUSY")
        try:
            self._wait_for(lambda: self._latest_joints is not None, 5.0, 'joint_states')
            if self.grip_home_start and self._homing_client is not None: self.home_gripper()
            block = self.localize_block(pose_xy=self._pick_only_pose, skip_bounds=self._pick_only_skip_bounds, custom_gz=self._pick_only_grasp_z)
            if block is None: return
            bx, by, _ = block
            travel_z  = self.table_z + self.cube_half + self.lift_height
            hover     = (bx, by, self.table_z + self.cube_half + self.approach_height)
            grasp     = (bx, by, self.grasp_z)
            lift      = (bx, by, travel_z)
            pick_yaw = self._grasp_yaw_override
            self._grasp_yaw_override = None

            steps = [
                ('1. Hover above block', lambda: self.move_to_pose(hover, 'hover', yaw=pick_yaw)),
                ('   settle',            lambda: self._settle(0.8)),
                ('2. Open gripper',      self.open_gripper),
                ('   settle',            lambda: self._settle(0.6)),
                ('3. Descend to grasp',  lambda: self.cartesian_move(grasp, hover[2] - grasp[2], 'descend', yaw=pick_yaw)),
                ('   settle',            lambda: self._settle(0.5)),
                ('4. Grasp block',       self.close_gripper),
                ('   settle',            lambda: self._settle(0.6)),
                ('5. Lift block',        lambda: self.cartesian_move(lift, travel_z - grasp[2], 'lift', yaw=pick_yaw)),
            ]
            if self._run_steps(steps): self._is_holding = True
            else: self._is_holding = False
        finally:
            self._busy = False
            self.publish_state("HOME")

    def run_place_only(self):
        if self._busy: return
        self._busy = True
        self.publish_state("BUSY")
        try:
            if not self._is_holding: return
            if self._place_only_pose:
                x, y, z = self._place_only_pose
                self.drop = (x, y, z if z > 0.001 else self.drop[2])
            drop = self.drop
            behind_robot = drop[0] < BEHIND_ROBOT_X
            transit_z = (self.table_z + self.cube_half + self.lift_height + (BEHIND_EXTRA_Z if behind_robot else 0.0))
            drop_above = (drop[0], drop[1], transit_z)

            if behind_robot:
                steps = [
                    ('6. Via-point (overhead)',   lambda: self.move_to_pose(BEHIND_OVERHEAD, 'overhead-via', extra_time=5.0)),
                    ('   settle',                 lambda: self._settle(0.6)),
                    ('7. Transfer to drop',       lambda: self.move_to_pose(drop_above, 'above-drop', extra_time=5.0)),
                    ('   settle',                 lambda: self._settle(0.6)),
                    ('8. Place down (jnt-space)', lambda: self.move_to_pose(drop, 'place', extra_time=5.0)),
                    ('   settle',                 lambda: self._settle(0.5)),
                    ('9. Open gripper',           self.open_gripper),
                    ('   settle',                 lambda: self._settle(0.6)),
                    ('10. Retreat up',            lambda: self.move_to_pose(drop_above, 'retreat-up', extra_time=5.0)),
                    ('    settle',                lambda: self._settle(0.4)),
                    ('11. Via-point (return)',    lambda: self.move_to_pose(BEHIND_OVERHEAD, 'overhead-return', extra_time=5.0)),
                ]
            else:
                steps = [
                    ('6. Transfer to drop', lambda: self.move_to_pose(drop_above, 'above-drop')),
                    ('   settle',           lambda: self._settle(0.6)),
                    ('7. Place down',       lambda: self.cartesian_move(drop, transit_z - drop[2], 'place')),
                    ('   settle',           lambda: self._settle(0.5)),
                    ('8. Open gripper',     self.open_gripper),
                    ('   settle',           lambda: self._settle(0.6)),
                    ('9. Retreat up',       lambda: self.cartesian_move(drop_above, transit_z - drop[2], 'retreat')),
                ]
            if self.return_home: steps.append(('Return home', lambda: self.move_to_joints(HOME, 'home')))
            if self._run_steps(steps): self._is_holding = False
        finally:
            self._busy = False
            self.publish_state("HOME")


def main(args=None):
    rclpy.init(args=args)
    node = RealPickAndPlace()

    def reset_full():
        node._latest_block_pose = None
        node._block_pose_stamp  = 0.0
        node._skip_bounds_check = False
        node._custom_grasp_z    = None

    def reset_pick():
        node._pick_only_pose  = None
        node._pick_only_stamp = 0.0

    def reset_place():
        node._place_only_pose  = None
        node._place_only_stamp = 0.0

    try:
        while rclpy.ok():
            t = time.time()
            node.get_logger().info('=== Ready — waiting for voice command ===')

            def any_trigger():
                return (
                    node._block_pose_stamp  >= t or
                    node._pick_only_stamp   >= t or
                    node._place_only_stamp  >= t
                )

            while rclpy.ok() and not any_trigger():
                rclpy.spin_once(node, timeout_sec=0.1)

            if not rclpy.ok(): break

            if node._place_only_stamp >= t:
                node.run_place_only()
                reset_place()
            elif node._pick_only_stamp >= t:
                node.run_pick_only()
                reset_pick()
            elif node._block_pose_stamp >= t:
                # FIX: Drain callbacks to guarantee /place_target and /grasp_yaw are ingested
                drain_deadline = time.time() + 0.15
                while time.time() < drain_deadline:
                    rclpy.spin_once(node, timeout_sec=0.02)
                
                node.run()
                reset_full()

            time.sleep(0.5)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()