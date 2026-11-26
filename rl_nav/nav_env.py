import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .scenarios import ROBOT_RADIUS, WORLD_SIZE, generate_scenario, scenario_to_world
from .world import World

MAX_LIN_VEL = 0.22   # m/s, TurtleBot3 Burger limit -> maps to /cmd_vel Twist.linear.x
MAX_ANG_VEL = 2.84   # rad/s                      -> maps to /cmd_vel Twist.angular.z
LIDAR_RAYS = 36      # 360-degree scan downsampled every 10 degrees (like /scan)
LIDAR_RANGE = 3.5    # m
DT = 0.25            # s, one control step
MAX_STEPS = 200      # 50 s per episode
GOAL_RADIUS = 0.18   # m


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class NavEnv(gym.Env):
    """Differential-drive navigation with a 36-beam lidar.

    Interface mirrors ROS2: observation = /scan + goal pose in body frame,
    action = (v, w) command as in geometry_msgs/Twist.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 12}

    def __init__(self, seed=None, scenario_pool=None, render_mode=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.scenario_pool = scenario_pool  # list of scenario dicts; None = procedural
        self.pool_idx = 0
        self.render_mode = render_mode

        obs_dim = LIDAR_RAYS + 3 + 2  # scan + goal(sin, cos, dist) + last action
        self.observation_space = spaces.Box(-1.0, 1.0, (obs_dim,), np.float32)
        self.action_space = spaces.Box(
            np.array([0.0, -1.0], np.float32),
            np.array([1.0, 1.0], np.float32),
        )

        self.world = World(WORLD_SIZE)
        self.scan_angles = np.linspace(0.0, 2.0 * np.pi, LIDAR_RAYS, endpoint=False)
        self.pose = np.zeros(3)
        self.goal = np.zeros(2)
        self.traj = []
        self.last_action = np.zeros(2, np.float32)
        self.steps = 0
        self.min_clearance = np.inf
        self.prev_dist = np.inf
        self._reset_state()

    def _reset_state(self):
        if self.scenario_pool is not None:
            s = dict(self.scenario_pool[self.pool_idx % len(self.scenario_pool)])
            self.pool_idx += 1
        else:
            s = generate_scenario(self.rng)
        self.world = scenario_to_world(s)
        self.start = np.asarray(s["start"], dtype=np.float64)
        self.goal = np.asarray(s["goal"], dtype=np.float64)
        ang = np.arctan2(self.goal[1] - self.start[1], self.goal[0] - self.start[0])
        heading = ang + self.rng.uniform(-np.pi / 3, np.pi / 3)
        self.pose = np.array([self.start[0], self.start[1], _wrap(heading)])
        self.steps = 0
        self.min_clearance = np.inf
        self.traj = [self.pose[:2].copy()]
        self.prev_dist = float(np.linalg.norm(self.goal - self.pose[:2]))
        self.last_action = np.zeros(2, np.float32)

    def scan(self):
        x, y, th = self.pose
        return self.world.raycast(x, y, self.scan_angles + th, LIDAR_RANGE)

    def _obs(self):
        x, y, th = self.pose
        scan = np.clip(self.scan() / LIDAR_RANGE, 0.0, 1.0)
        dx, dy = self.goal[0] - x, self.goal[1] - y
        dist = np.clip(np.hypot(dx, dy) / (WORLD_SIZE * np.sqrt(2.0)), 0.0, 1.0)
        ang = _wrap(np.arctan2(dy, dx) - th)
        obs = np.concatenate([scan, [np.sin(ang), np.cos(ang), dist], self.last_action])
        return obs.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float64).reshape(2),
            self.action_space.low,
            self.action_space.high,
        )
        self.last_action = action.astype(np.float32)
        v = action[0] * MAX_LIN_VEL
        w = action[1] * MAX_ANG_VEL

        x, y, th = self.pose
        th_mid = th + 0.5 * w * DT
        x2 = x + v * np.cos(th_mid) * DT
        y2 = y + v * np.sin(th_mid) * DT
        self.pose = np.array([x2, y2, _wrap(th + w * DT)])
        self.steps += 1
        self.traj.append(self.pose[:2].copy())

        dist = float(np.linalg.norm(self.goal - self.pose[:2]))
        clearance = self.world.clearance(x2, y2)
        self.min_clearance = min(self.min_clearance, clearance)

        reward = 5.0 * (self.prev_dist - dist) - 0.01
        terminated = truncated = False
        info = {}
        if dist <= GOAL_RADIUS:
            reward += 10.0
            terminated = True
            info["outcome"] = "success"
        elif clearance <= ROBOT_RADIUS:
            reward = -10.0
            terminated = True
            info["outcome"] = "collision"
        elif self.steps >= MAX_STEPS:
            truncated = True
            info["outcome"] = "timeout"
        self.prev_dist = dist

        obs = self._obs()
        if self.render_mode == "rgb_array":
            info["frame"] = self.render()
        return obs, float(reward), terminated, truncated, info

    def render(self):
        from . import render

        return render.frame_rgb(
            self.world, self.goal, self.pose, self.traj, self.scan(), LIDAR_RANGE
        )
