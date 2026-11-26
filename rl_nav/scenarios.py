"""Scenario generation: procedural maps + a fixed evaluation set.

场景 = 地图（障碍物）+ 起点 + 终点。训练时每个 episode 程序化随机生成；
评估集用固定种子生成一次后存盘，保证结果可复现。
"""
import json
from collections import deque
from pathlib import Path

import numpy as np

from .world import World

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WORLD_SIZE = 8.0
ROBOT_RADIUS = 0.105  # TurtleBot3 Burger footprint radius
GRID_RES = 0.10
EVAL_SET_PATH = DATA_DIR / "eval_scenarios.json"


def world_to_scenario(world, start, goal):
    return {
        "size": world.size,
        "start": [float(start[0]), float(start[1])],
        "goal": [float(goal[0]), float(goal[1])],
        "circles": [list(map(float, c)) for c in world.circles],
        "rects": [list(map(float, r)) for r in world.rects],
    }


def scenario_to_world(s):
    w = World(s.get("size", WORLD_SIZE))
    for c in s["circles"]:
        w.add_circle(*c)
    for r in s["rects"]:
        w.add_rect(*r)
    return w


def _occupancy_free(world, margin):
    n = int(round(world.size / GRID_RES)) + 1
    g = np.linspace(0.0, world.size, n)
    X, Y = np.meshgrid(g, g, indexing="ij")
    m = margin
    free = (X > m) & (Y > m) & (X < world.size - m) & (Y < world.size - m)
    if world.circles:
        c = np.asarray(world.circles)
        d = np.hypot(X[..., None] - c[None, None, :, 0], Y[..., None] - c[None, None, :, 1]) - c[None, None, :, 2]
        free &= d.min(axis=-1) >= margin
    for (x0, y0, x1, y1) in world.rects:
        qx = np.clip(X, x0, x1)
        qy = np.clip(Y, y0, y1)
        free &= np.hypot(X - qx, Y - qy) >= margin
    return free


def reachable(world, start, goal, margin):
    """Coarse BFS on an inflated occupancy grid: does a collision-free path exist?"""
    free = _occupancy_free(world, margin)
    n = free.shape[0]

    def to_idx(v):
        return int(np.clip(round(v / GRID_RES), 0, n - 1))

    s = (to_idx(start[0]), to_idx(start[1]))
    t = (to_idx(goal[0]), to_idx(goal[1]))
    if not free[s] or not free[t]:
        return False
    q = deque([s])
    seen = np.zeros_like(free, dtype=bool)
    seen[s] = True
    while q:
        i, j = q.popleft()
        if (i, j) == t:
            return True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n and free[ni, nj] and not seen[ni, nj]:
                seen[ni, nj] = True
                q.append((ni, nj))
    return False


def generate_scenario(rng):
    """Random valid scenario: obstacles fit, start/goal clear, path reachable."""
    size = WORLD_SIZE
    for _ in range(50):
        start = rng.uniform(0.8, size - 0.8, 2)
        goal = rng.uniform(0.8, size - 0.8, 2)
        if np.linalg.norm(goal - start) < 4.0:
            continue
        w = World(size)
        for _ in range(int(rng.integers(2, 7))):
            r = rng.uniform(0.2, 0.5)
            for _t in range(40):
                c = rng.uniform(r + 0.05, size - r - 0.05, 2)
                if w.clearance(c[0], c[1]) >= r + 0.3:
                    w.add_circle(c[0], c[1], r)
                    break
        for _ in range(int(rng.integers(1, 4))):
            hw = rng.uniform(0.2, 0.6)
            hh = rng.uniform(0.2, 0.6)
            for _t in range(40):
                cxy = rng.uniform(hw + 0.05, size - hw - 0.05, 2)
                x0, y0 = cxy[0] - hw, cxy[1] - hh
                x1, y1 = cxy[0] + hw, cxy[1] + hh
                probes = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (cxy[0], cxy[1])]
                if min(w.clearance(px, py) for px, py in probes) >= 0.3:
                    w.add_rect(x0, y0, x1, y1)
                    break
        if w.clearance(start[0], start[1]) < 0.5 or w.clearance(goal[0], goal[1]) < 0.5:
            continue
        if reachable(w, start, goal, ROBOT_RADIUS + 0.05):
            return world_to_scenario(w, start, goal)
    return world_to_scenario(World(size), [0.8, 0.8], [size - 0.8, size - 0.8])


def make_eval_set(n=50, seed=2024):
    rng = np.random.default_rng(seed)
    scenarios = [generate_scenario(rng) for _ in range(n)]
    DATA_DIR.mkdir(exist_ok=True)
    EVAL_SET_PATH.write_text(json.dumps(scenarios))
    return scenarios


def load_eval_set():
    if EVAL_SET_PATH.exists():
        return json.loads(EVAL_SET_PATH.read_text())
    return make_eval_set()
