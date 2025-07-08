import math
import numpy as np
from scipy.interpolate import CubicSpline


def arc_geom_points_from_centers(centers, R):
    """Return (center, start, end) tuples for each arc using explicit centers."""
    c0, c1, c2, c3 = centers
    return [
        (c0, (c0[0], c0[1] + R), (c0[0] - R, c0[1])),
        (c1, (c1[0] - R, c1[1]), (c1[0], c1[1] - R)),
        (c2, (c2[0], c2[1] - R), (c2[0] + R, c2[1])),
        (c3, (c3[0] + R, c3[1]), (c3[0], c3[1] + R)),
    ]


def arc_geom_points(a, b, R):
    """Return (center, start, end) tuples for each arc."""
    a2, b2 = a / 2, b / 2
    r2 = math.sqrt(2) / 2
    arcs = []
    arcs.append(((-a2 + R - R * r2, b2 - R + R * r2),
                 (-a2 + R, b2),
                 (-a2, b2 - R)))
    arcs.append(((-a2 + R - R * r2, -b2 + R - R * r2),
                 (-a2, -b2 + R),
                 (-a2 + R, -b2)))
    arcs.append(((a2 - R + R * r2, -b2 + R - R * r2),
                 (a2 - R, -b2),
                 (a2, -b2 + R)))
    arcs.append(((a2 - R + R * r2, b2 - R + R * r2),
                 (a2, b2 - R),
                 (a2 - R, b2)))
    return arcs


def rounded_rect_points(a, b, R, *, step=5.0, n_arc=180, n_line=200):
    a2, b2 = a / 2.0, b / 2.0

    def arc(xc, yc, ang0, ang1):
        t = np.linspace(ang0, ang1, n_arc, endpoint=False)
        return np.column_stack((xc + R * np.cos(t), yc + R * np.sin(t)))

    def line(p0, p1):
        p0, p1 = map(np.asarray, (p0, p1))
        t = np.linspace(0, 1, n_line, endpoint=False)[:, None]
        return p0 + t * (p1 - p0)

    dense = np.vstack([
        arc(-a2 + R, b2 - R, math.pi / 2, math.pi),
        line([-a2, b2 - R], [-a2, -b2 + R]),
        arc(-a2 + R, -b2 + R, math.pi, 3 * math.pi / 2),
        line([-a2 + R, -b2], [a2 - R, -b2]),
        arc(a2 - R, -b2 + R, 3 * math.pi / 2, 2 * math.pi),
        line([a2, -b2 + R], [a2, b2 - R]),
        arc(a2 - R, b2 - R, 0.0, math.pi / 2),
        line([a2 - R, b2], [-a2 + R, b2]),
    ])
    seg = np.linalg.norm(np.diff(dense, axis=0, append=dense[:1]), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg[:-1])))
    total = s[-1] + seg[-1]
    m = max(4, int(total / step))
    su = np.linspace(0.0, total, m, endpoint=False)
    x = np.interp(su, s, dense[:, 0])
    y = np.interp(su, s, dense[:, 1])
    return np.column_stack((x, y))


def rounded_rect_points_from_centers(centers, R, *, step=5.0, n_arc=180, n_line=200):
    """Return contour points for a rounded shape defined by centers."""

    def arc(xc, yc, ang0, ang1):
        t = np.linspace(ang0, ang1, n_arc, endpoint=False)
        return np.column_stack((xc + R * np.cos(t), yc + R * np.sin(t)))

    def line(p0, p1):
        p0, p1 = map(np.asarray, (p0, p1))
        t = np.linspace(0, 1, n_line, endpoint=False)[:, None]
        return p0 + t * (p1 - p0)

    c0, c1, c2, c3 = centers
    dense = np.vstack([
        arc(c0[0], c0[1], math.pi / 2, math.pi),
        line([c0[0] - R, c0[1]], [c1[0] - R, c1[1]]),
        arc(c1[0], c1[1], math.pi, 3 * math.pi / 2),
        line([c1[0], c1[1] - R], [c2[0], c2[1] - R]),
        arc(c2[0], c2[1], 3 * math.pi / 2, 2 * math.pi),
        line([c2[0] + R, c2[1]], [c3[0] + R, c3[1]]),
        arc(c3[0], c3[1], 0.0, math.pi / 2),
        line([c3[0], c3[1] + R], [c0[0], c0[1] + R]),
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
