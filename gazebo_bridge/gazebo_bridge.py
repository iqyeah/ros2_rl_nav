#!/usr/bin/env python3
"""ROS2 bridge: run the RL nav policy on live /scan + /odom, publish /cmd_vel.

Pure numpy inference (weights come from an .npz exported by export_policy_npz.py),
so the container needs no torch/SB3. Observation and action contracts are
identical to training (rl_nav/nav_env.py):

  obs  = 36 lidar rays normalized by 3.5 m
       + goal in body frame (sin(ang), cos(ang), dist/8*sqrt(2))
       + last normalized action
  act  = (v_hat in [0,1] -> 0..0.22 m/s, w_hat in [-1,1] -> -2.84..2.84 rad/s)

The goal is expected in the ODOMETER frame (on --goal-topic, default /rl_goal
to avoid clashing with Nav2's /goal_pose), since wheel odometry and gz world
coordinates drift apart across teleports/slips.
"""
import argparse
import math

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

LIDAR_RAYS = 36
LIDAR_RANGE = 3.5
DT = 0.25
MAX_STEPS = 200
GOAL_RADIUS = 0.18
V_MAX = 0.22
W_MAX = 2.84
WORLD_DIAG = 8.0 * math.sqrt(2.0)
ACT_LOW, ACT_HIGH = np.array([0.0, -1.0]), np.array([1.0, 1.0])


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class RlNavBridge(Node):
    def __init__(self, args):
        super().__init__("rl_nav_bridge")
        npz = np.load(args.npz)
        self.mean, self.var = npz["obs_mean"], npz["obs_var"]
        self.weights = tuple(npz[k] for k in ("w1", "b1", "w2", "b2", "w3", "b3"))
        self.goal = np.array([args.x, args.y], dtype=float) if args.x is not None else None
        self.pose = None
        self.scan_idx = None
        self.scan36 = np.full(LIDAR_RAYS, LIDAR_RANGE)
        self.last_a = np.zeros(2, dtype=np.float32)
        self.step = 0
        self.done = True  # start idle until a goal arrives

        self.create_subscription(LaserScan, "/scan", self.on_scan, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(PoseStamped, args.goal_topic, self.on_goal, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_timer(DT, self.tick)
        self.get_logger().info(
            f"bridge up: goal={self.goal} dt={DT}s max_steps={MAX_STEPS}"
        )

    def on_scan(self, msg):
        n = len(msg.ranges)
        if n == 0:
            return
        if self.scan_idx is None:
            inc = msg.angle_increment or (msg.angle_max - msg.angle_min) / max(n - 1, 1)
            self.scan_idx = [
                int(round((2.0 * math.pi * i / LIDAR_RAYS - msg.angle_min) / inc)) % n
                for i in range(LIDAR_RAYS)
            ]
            self.get_logger().info(
                f"scan mapped: {n} beams inc={inc:.4f} -> {LIDAR_RAYS} rays"
            )
        vals = np.asarray(msg.ranges, dtype=float)[self.scan_idx]
        vals = np.nan_to_num(vals, nan=LIDAR_RANGE, posinf=LIDAR_RANGE, neginf=LIDAR_RANGE)
        self.scan36 = np.clip(vals, 0.0, LIDAR_RANGE)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = np.array([p.x, p.y, yaw])

    def on_goal(self, msg):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y])
        self.step = 0
        self.done = False
        self.last_a = np.zeros(2, dtype=np.float32)
        self.get_logger().info(
            f"goal (odom frame): x={self.goal[0]:.2f} y={self.goal[1]:.2f}"
        )

    def act(self, obs):
        x = np.clip((obs - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0)
        w1, b1, w2, b2, w3, b3 = self.weights
        h = np.tanh(x @ w1.T + b1)
        h = np.tanh(h @ w2.T + b2)
        return np.clip(h @ w3.T + b3, ACT_LOW, ACT_HIGH)

    def stop(self):
        self.cmd_pub.publish(Twist())

    def tick(self):
        if self.done or self.pose is None or self.goal is None:
            return
        dx, dy = self.goal - self.pose[:2]
        dist = math.hypot(dx, dy)
        if dist <= GOAL_RADIUS:
            self.done = True
            self.stop()
            self.get_logger().info(f"RESULT: success steps={self.step} dist={dist:.3f}")
            return
        self.step += 1
        if self.step > MAX_STEPS:
            self.done = True
            self.stop()
            self.get_logger().info(f"RESULT: timeout dist_left={dist:.3f}")
            return
        ang = wrap(math.atan2(dy, dx) - self.pose[2])
        obs = np.concatenate(
            [self.scan36 / LIDAR_RANGE,
             [math.sin(ang), math.cos(ang), min(dist / WORLD_DIAG, 1.0)],
             self.last_a]
        ).astype(np.float32)
        a = self.act(obs)
        self.last_a = a.astype(np.float32)
        t = Twist()
        t.linear.x = float(a[0] * V_MAX)
        t.angular.z = float(a[1] * W_MAX)
        self.cmd_pub.publish(t)
        if self.step % 8 == 0:
            self.get_logger().info(
                f"step {self.step:3d} dist={dist:.2f} yaw_err={ang:+.2f} "
                f"v={t.linear.x:.2f} w={t.angular.z:+.2f}"
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--goal-topic", default="/rl_goal")
    p.add_argument("--x", type=float, default=None, help="goal x in odom frame")
    p.add_argument("--y", type=float, default=None, help="goal y in odom frame")
    args = p.parse_args()
    rclpy.init()
    node = RlNavBridge(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
