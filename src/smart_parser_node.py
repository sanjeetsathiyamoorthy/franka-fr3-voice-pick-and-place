#!/usr/bin/env python3
"""
smart_parser_node.py — Refactored for Left/Right zones
"""

import re
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

COLORS = {
    'red': 'RED', 'green': 'GREEN', 'blue': 'BLUE', 'yellow': 'YELLOW',
}

SIDE_PATTERNS = [
    (r'\b(?:on|to|in|into|onto|place|put|drop|move|send|bring|side|zone)\s+(?:the\s+)?left\b', 'LEFT'),
    (r'\bleft\s+(?:side|half|area|section|zone)\b', 'LEFT'),
    (r'^left$', 'LEFT'),
    (r'\b(?:on|to|in|into|onto|place|put|drop|move|send|bring|side|zone)\s+(?:the\s+)?right\b', 'RIGHT'),
    (r'\bright\s+(?:side|half|area|section|zone)\b', 'RIGHT'),
    (r'^right$', 'RIGHT'),
]

HOME_PATTERNS = [r'\bgo\s+home\b', r'\breturn\s+(?:to\s+)?home\b', r'\breset\b']
STOP_PATTERNS = [r'\bstop\b', r'\babort\b', r'\bcancel\b']
STATUS_PATTERNS = [(r'\b(?:status|state)\b', 'STATUS')]

class SmartParserNode(Node):

    def __init__(self):
        super().__init__('smart_parser_node')
        self._sub = self.create_subscription(String, '/voice_raw', self._voice_cb, 10)
        self._pub = self.create_publisher(String, '/pick_command', 10)
        self.get_logger().info('SmartParserNode ready — Left/Right zones active')

    def _voice_cb(self, msg: String):
        raw  = msg.data
        text = raw.lower().strip()
        self.get_logger().info(f'Heard: {raw}')
        
        cmd  = self._parse(text)
        if cmd:
            out = String()
            out.data = cmd
            self._pub.publish(out)
            self.get_logger().info(f'Parsed command: {cmd}')
        else:
            self.get_logger().warn(f'No match for: "{raw}"')

    def _parse(self, text: str) -> str | None:
        if any(re.search(p, text) for p in STOP_PATTERNS): return 'STOP'
        if any(re.search(p, text) for p in HOME_PATTERNS): return 'HOME'
        for pattern, cmd in STATUS_PATTERNS:
            if re.search(pattern, text): return cmd

        color = next((code for word, code in COLORS.items() if re.search(rf'\b{word}\b', text)), None)
        side = next((code for pat, code in SIDE_PATTERNS if re.search(pat, text)), None)

        if color and side:
            return f'PICK_{color}_{side}'
        elif color:
            return f'PICK_{color}'
        elif side:
            return f'PLACE_{side}'

        return None

def main(args=None):
    rclpy.init(args=args)
    node = SmartParserNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()