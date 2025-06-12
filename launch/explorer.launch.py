from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='macadamia_explorer',
            executable='explorer_node',
            name='macadamia_explorer',
            output='screen',
        ),
    ])
