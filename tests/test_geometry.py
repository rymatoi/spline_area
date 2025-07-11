import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from geometry import (
    rounded_rect_points,
    arc_geom_points,
    _arc_angles_from_centers,
    rounded_rect_area,
)


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
    angs = _arc_angles_from_centers(centers)
    for center, arc, ang_pair in zip(centers, arcs, angs):
        mid, start, end = arc
        cx, cy = center
        a0, a1 = ang_pair
        exp_mid = (
            cx + R * np.cos((a0 + a1) / 2),
            cy + R * np.sin((a0 + a1) / 2),
        )
        exp_start = (cx + R * np.cos(a0), cy + R * np.sin(a0))
        exp_end = (cx + R * np.cos(a1), cy + R * np.sin(a1))
        assert np.allclose(mid, exp_mid)
        assert np.allclose(start, exp_start)
        assert np.allclose(end, exp_end)


def test_lines_are_tangents():
    a, b, R = 200, 100, 20
    centers = default_centers(a, b, R)
    centers[0] = (centers[0][0] - 10, centers[0][1] + 5)
    centers[1] = (centers[1][0] - 20, centers[1][1] - 5)
    arcs = arc_geom_points(a, b, R, centers=centers)
    for i in range(4):
        end_pt = np.array(arcs[i][2])
        start_pt_next = np.array(arcs[(i + 1) % 4][1])
        line_vec = start_pt_next - end_pt
        c1 = np.array(centers[i])
        c2 = np.array(centers[(i + 1) % 4])
        r1 = end_pt - c1
        r2 = start_pt_next - c2
        assert np.isclose(np.dot(line_vec, r1), 0.0)
        assert np.isclose(np.dot(line_vec, r2), 0.0)


def test_rounded_area_matches_formula():
    a, b, R = 200, 100, 20
    area_calc = rounded_rect_area(a, b, R)
    expected = a * b - (4 - np.pi) * (R ** 2)
    assert np.isclose(area_calc, expected)


def test_area_changes_with_moved_centers():
    a, b, R = 200, 100, 20
    centers = default_centers(a, b, R)
    default_area = rounded_rect_area(a, b, R, centers=centers)
    centers[0] = (centers[0][0] - 30, centers[0][1] + 15)
    changed = rounded_rect_area(a, b, R, centers=centers)
    assert not np.isclose(changed, default_area)
