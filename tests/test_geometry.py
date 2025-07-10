import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from geometry import rounded_rect_points, arc_geom_points


def default_centers(a, b, R):
    a2, b2 = a / 2, b / 2
    return [
        (-a2 + R, b2 - R),
        (-a2 + R, -b2 + R),
        (a2 - R, -b2 + R),
        (a2 - R, b2 - R),
    ]


def test_default_bounds():
    a, b, R = 200, 100, 20
    pts = rounded_rect_points(a, b, R, step=1.0)
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    assert np.isclose(x_min, -a / 2)
    assert np.isclose(x_max, a / 2)
    assert np.isclose(y_min, -b / 2)
    assert np.isclose(y_max, b / 2)


def test_trapezoid_width_increase():
    a, b, R = 200, 100, 20
    centers = default_centers(a, b, R)
    delta = 30
    centers[1] = (centers[1][0] - delta, centers[1][1])
    centers[2] = (centers[2][0] + delta, centers[2][1])
    pts_wide = rounded_rect_points(a, b, R, step=1.0, centers=centers)
    width_default = a
    width_new = pts_wide[:, 0].max() - pts_wide[:, 0].min()
    assert width_new > width_default


def test_arc_midpoints_follow_centers():
    a, b, R = 200, 100, 20
    centers = default_centers(a, b, R)
    arcs = arc_geom_points(a, b, R, centers=centers)
    angs = [
        (np.pi / 2, np.pi),
        (np.pi, 3 * np.pi / 2),
        (3 * np.pi / 2, 2 * np.pi),
        (0.0, np.pi / 2),
    ]
    for center, arc, ang_pair in zip(centers, arcs, angs):
        mid, start, end = arc
        cx, cy = center
        a0, a1 = ang_pair
        exp_mid = (cx + R * np.cos((a0 + a1) / 2), cy + R * np.sin((a0 + a1) / 2))
        exp_start = (cx + R * np.cos(a0), cy + R * np.sin(a0))
        exp_end = (cx + R * np.cos(a1), cy + R * np.sin(a1))
        assert np.allclose(mid, exp_mid)
        assert np.allclose(start, exp_start)
        assert np.allclose(end, exp_end)
