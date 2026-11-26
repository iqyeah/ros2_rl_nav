import numpy as np


class World:
    """Square 2D world: outer walls + circular and box obstacles."""

    def __init__(self, size=8.0):
        self.size = float(size)
        self.circles = []  # (cx, cy, r)
        self.rects = []    # (x0, y0, x1, y1)

    def add_circle(self, cx, cy, r):
        self.circles.append((float(cx), float(cy), float(r)))

    def add_rect(self, x0, y0, x1, y1):
        x0, x1 = sorted((float(x0), float(x1)))
        y0, y1 = sorted((float(y0), float(y1)))
        self.rects.append((x0, y0, x1, y1))

    def _segments(self):
        s = self.size
        segs = [
            (0.0, 0.0, s, 0.0),
            (s, 0.0, s, s),
            (s, s, 0.0, s),
            (0.0, s, 0.0, 0.0),
        ]
        for (x0, y0, x1, y1) in self.rects:
            segs += [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]
        return np.asarray(segs, dtype=np.float64).reshape(-1, 4)

    def raycast(self, px, py, angles, max_range):
        """Distance to first hit along each absolute angle (max_range if no hit)."""
        angles = np.atleast_1d(np.asarray(angles, dtype=np.float64))
        dx, dy = np.cos(angles), np.sin(angles)
        dist = np.full(angles.shape, float(max_range))

        segs = self._segments()
        if len(segs):
            ax, ay, bx, by = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
            sx, sy = bx - ax, by - ay
            denom = dx[:, None] * sy[None, :] - dy[:, None] * sx[None, :]
            parallel = np.abs(denom) < 1e-12
            denom = np.where(parallel, 1.0, denom)
            t = ((ax[None, :] - px) * sy[None, :] - (ay[None, :] - py) * sx[None, :]) / denom
            u = ((ax[None, :] - px) * dy[:, None] - (ay[None, :] - py) * dx[:, None]) / denom
            hit = (~parallel) & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
            t = np.where(hit, t, np.inf)
            if np.isfinite(t).any():
                dist = np.minimum(dist, t.min(axis=1))

        if self.circles:
            c = np.asarray(self.circles)  # (C, 3)
            ox, oy = px - c[:, 0], py - c[:, 1]
            b = ox[None, :] * dx[:, None] + oy[None, :] * dy[:, None]
            cc = ox ** 2 + oy ** 2 - c[:, 2] ** 2
            disc = b ** 2 - cc[None, :]
            valid = disc >= 0.0
            t = -b - np.sqrt(np.where(valid, disc, 0.0))
            hit = valid & (t >= 0.0)
            t = np.where(hit, t, np.inf)
            if np.isfinite(t).any():
                dist = np.minimum(dist, t.min(axis=1))

        return dist

    def clearance(self, px, py):
        """Distance from a point to the nearest obstacle surface (walls included)."""
        d = min(float(px), float(py), self.size - float(px), self.size - float(py))
        if self.circles:
            c = np.asarray(self.circles)
            dc = np.hypot(px - c[:, 0], py - c[:, 1]) - c[:, 2]
            d = min(d, float(dc.min()))
        for (x0, y0, x1, y1) in self.rects:
            qx = np.clip(px, x0, x1)
            qy = np.clip(py, y0, y1)
            d = min(d, float(np.hypot(px - qx, py - qy)))
        return d
