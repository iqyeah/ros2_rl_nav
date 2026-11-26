import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle


def draw_scene(ax, world, goal, pose=None, traj=None, scan=None, lidar_range=3.5, title=None):
    ax.clear()
    s = world.size
    ax.set_xlim(-0.3, s + 0.3)
    ax.set_ylim(-0.3, s + 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)

    ax.add_patch(Rectangle((0, 0), s, s, fill=False, ec="k", lw=2))
    for (x0, y0, x1, y1) in world.rects:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#8d6e63", ec="k", lw=0.8))
    for (cx, cy, r) in world.circles:
        ax.add_patch(Circle((cx, cy), r, fc="#8d6e63", ec="k", lw=0.8))

    if scan is not None and pose is not None:
        angs = np.linspace(0.0, 2.0 * np.pi, len(scan), endpoint=False) + pose[2]
        ex = pose[0] + scan * np.cos(angs)
        ey = pose[1] + scan * np.sin(angs)
        segs = np.stack(
            [np.stack([np.full(len(scan), pose[0]), ex], axis=-1),
             np.stack([np.full(len(scan), pose[1]), ey], axis=-1)],
            axis=1,
        )
        ax.add_collection(LineCollection(segs, colors="#42a5f5", lw=0.5, alpha=0.5))
        hit = scan < lidar_range - 1e-6
        ax.plot(ex[hit], ey[hit], "r.", ms=2)

    if traj is not None and len(traj) > 0:
        t = np.asarray(traj)
        ax.plot(t[:, 0], t[:, 1], "-", c="#1e88e5", lw=1.5)

    ax.plot(goal[0], goal[1], marker="*", ms=16, c="#43a047", mec="k")

    if pose is not None:
        ax.add_patch(Circle(pose[:2], 0.105, fc="#e53935", ec="k", lw=0.8))
        ax.arrow(
            pose[0], pose[1],
            0.20 * np.cos(pose[2]), 0.20 * np.sin(pose[2]),
            width=0.02, head_width=0.06, fc="k", zorder=5,
        )


def frame_rgb(world, goal, pose=None, traj=None, scan=None, lidar_range=3.5, title=None):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    draw_scene(ax, world, goal, pose, traj, scan, lidar_range, title)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def save_gif(frames, path, fps=12):
    from PIL import Image

    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
