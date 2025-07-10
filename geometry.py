import math
import numpy as np
from scipy.interpolate import CubicSpline


def arc_geom_points(a, b, R, *, centers=None):
    """Return (arc_mid, start, end) tuples for each corner arc."""
    if centers is None:
        a2, b2 = a / 2, b / 2
        centers = [
            (-a2 + R, b2 - R),
            (-a2 + R, -b2 + R),
            (a2 - R, -b2 + R),
            (a2 - R, b2 - R),
        ]

    ang = [
        (math.pi / 2, math.pi),
        (math.pi, 3 * math.pi / 2),
        (3 * math.pi / 2, 2 * math.pi),
        (0.0, math.pi / 2),
    ]
    arcs = []
    for (cx, cy), (a0, a1) in zip(centers, ang):
        amid = (a0 + a1) / 2
        start = (cx + R * math.cos(a0), cy + R * math.sin(a0))
        end = (cx + R * math.cos(a1), cy + R * math.sin(a1))
        mid = (cx + R * math.cos(amid), cy + R * math.sin(amid))
        arcs.append((mid, start, end))
    return arcs


def rounded_rect_points(a, b, R, *, step=5.0, n_arc=180, n_line=200, centers=None):
    if centers is None:
        a2, b2 = a / 2.0, b / 2.0
        centers = [
            (-a2 + R, b2 - R),
            (-a2 + R, -b2 + R),
            (a2 - R, -b2 + R),
            (a2 - R, b2 - R),
        ]
    else:
        centers = [tuple(c) for c in centers]

    def arc(xc, yc, ang0, ang1):
        t = np.linspace(ang0, ang1, n_arc, endpoint=False)
        return np.column_stack((xc + R * np.cos(t), yc + R * np.sin(t)))

    def line(p0, p1):
        p0, p1 = map(np.asarray, (p0, p1))
        t = np.linspace(0, 1, n_line, endpoint=False)[:, None]
        return p0 + t * (p1 - p0)

    ang = [
        (math.pi / 2, math.pi),
        (math.pi, 3 * math.pi / 2),
        (3 * math.pi / 2, 2 * math.pi),
        (0.0, math.pi / 2),
    ]
    arcs = [arc(cx, cy, a0, a1) for (cx, cy), (a0, a1) in zip(centers, ang)]
    lines = [
        line(arcs[0][-1], arcs[1][0]),
        line(arcs[1][-1], arcs[2][0]),
        line(arcs[2][-1], arcs[3][0]),
        line(arcs[3][-1], arcs[0][0]),
    ]
    dense = np.vstack([
        arcs[0], lines[0],
        arcs[1], lines[1],
        arcs[2], lines[2],
        arcs[3], lines[3],
    ])
    seg = np.linalg.norm(np.diff(dense, axis=0, append=dense[:1]), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg[:-1])))
    total = s[-1] + seg[-1]
    m = max(4, int(total / step))
    su = np.linspace(0.0, total, m, endpoint=False)
    x = np.interp(su, s, dense[:, 0])
    y = np.interp(su, s, dense[:, 1])
    return np.column_stack((x, y))


def cubic_spline_closed(points: np.ndarray, samples_per_seg: int = 24) -> np.ndarray:
    P = np.asarray(points, dtype=float)
    N = len(P)
    t = np.arange(N + 1)
    xy = np.vstack([P, P[0]])
    ts_dense = np.linspace(0, N, N * samples_per_seg, endpoint=False)
    cs_x = CubicSpline(t, xy[:, 0], bc_type='periodic')
    cs_y = CubicSpline(t, xy[:, 1], bc_type='periodic')
    return np.column_stack([cs_x(ts_dense), cs_y(ts_dense)])
