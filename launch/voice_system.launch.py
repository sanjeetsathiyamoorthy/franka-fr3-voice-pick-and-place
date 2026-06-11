from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node

def generate_launch_description():

    whisper = Node(
        package='fr3_delivery_sim',
        executable='whisper_node.py',
        name='whisper_node',
        output='screen',
    )

    smart_parser = Node(
        package='fr3_delivery_sim',
        executable='smart_parser_node.py',
        name='smart_parser_node',
        output='screen',
    )

    color_selector = Node(
        package='fr3_delivery_sim',
        executable='color_selector_node.py',
        name='color_selector_node',
        output='screen',
    )

    aruco_vision = Node(
        package='fr3_delivery_sim',
        executable='aruco_vision_detector.py',
        name='aruco_vision_detector',
        output='screen',
    )

    real_pick_and_place = Node(
        package='fr3_delivery_sim',
        executable='real_pick_and_place.py',
        name='real_pick_and_place',
        output='screen',
        parameters=[{
            'gripper_ns':            '/franka_gripper',
            'vel_scale':             0.15,
            'acc_scale':             0.15,
            'grasp_z':               0.025,
            'grasp_x_offset':        0.016,
            'grasp_y_offset':        0.003,
            'block_topic':           '/voice_pick_target',
            'detection_timeout':     300.0,
            'gripper_grasp_force':   40.0,
            'gripper_epsilon_inner': 0.03,
            'gripper_epsilon_outer': 0.03,
            'gripper_close_pos':     0.045,
        }]
    )

    return LaunchDescription([
        LogInfo(msg='=== FR3 Voice Pipeline Starting ==='),
        LogInfo(msg='Nodes: whisper -> smart_parser -> color_selector -> real_pick_and_place'),
        LogInfo(msg='Vision: aruco_vision_detector (red/green/blue poses)'),
        whisper,
        smart_parser,
        color_selector,
        aruco_vision,
        real_pick_and_place,
    ])
