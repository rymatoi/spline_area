import math
from typing import List, Sequence

import numpy as np
from scipy.interpolate import CubicSpline


def _arc_angles_from_centers(centers):
    """Return start/end angles for arcs so lines stay tangent."""
    c = [np.asarray(p, dtype=float) for p in centers]
    dirs = []
    for i in range(4):
        d = c[(i + 1) % 4] - c[i]
        v = d / np.linalg.norm(d)
        n = np.array([v[1], -v[0]])  # outward normal for tangent
        dirs.append(n)
    ang_pairs = []
    for i in range(4):
        a0 = math.atan2(dirs[(i - 1) % 4][1], dirs[(i - 1) % 4][0])
        a1 = math.atan2(dirs[i][1], dirs[i][0])
        if a1 <= a0:
            a1 += 2 * math.pi
        ang_pairs.append((a0, a1))
    return ang_pairs


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

    ang = _arc_angles_from_centers(centers)
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

    ang = _arc_angles_from_centers(centers)
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


def _chord_param(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return np.array([], dtype=float)
    diffs = np.diff(pts, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    t = np.concatenate(([0.0], np.cumsum(seg)))
    if t[-1] == 0:
        t = np.arange(len(pts), dtype=float)
    return t


def _collect_segments(points: np.ndarray, seam_indices: Sequence[int]) -> List[np.ndarray]:
    P = np.asarray(points, dtype=float)
    N = len(P)
    if N == 0:
        return []
    if not seam_indices:
        return [P]
    seams = np.unique(np.mod(seam_indices, N))
    seams.sort()
    if len(seams) < 2:
        return [P]

    segments = []
    for i in range(len(seams)):
        start = int(seams[i])
        end = int(seams[(i + 1) % len(seams)])
        seg_pts = [P[start]]
        idx = (start + 1) % N
        while idx != end:
            seg_pts.append(P[idx])
            idx = (idx + 1) % N
        seg_pts.append(P[end])
        segments.append(np.array(seg_pts, dtype=float))
    return segments


def _ensure_shared_endpoints(segments: List[np.ndarray]) -> List[np.ndarray]:
    if not segments:
        return []
    shared = [np.array(seg, dtype=float, copy=True) for seg in segments]
    n = len(shared)
    if n <= 1:
        return shared
    for i in range(n):
        j = (i + 1) % n
        if len(shared[i]) == 0 or len(shared[j]) == 0:
            continue
        # Copy the seam point verbatim instead of averaging so that only
        # designated joints are affected by the C¹ stitching.
        shared[j][0] = shared[i][-1]
    return shared


def _segment_tangent(seg: np.ndarray, *, forward: bool) -> np.ndarray:
    if len(seg) < 2:
        return np.zeros(2, dtype=float)
    if forward:
        return seg[1] - seg[0]
    return seg[-1] - seg[-2]


def _project_to_direction(vec: np.ndarray, direction: np.ndarray) -> np.ndarray:
    dir_norm = np.linalg.norm(direction)
    if dir_norm == 0:
        return vec
    unit = direction / dir_norm
    return np.dot(vec, unit) * unit


def build_c1_closed_spline(points: np.ndarray, seam_indices: Sequence[int] | None = None):
    """Return per-segment cubic splines stitched with C¹ continuity."""

    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("points must be an array of shape (N, 2)")
    N = len(P)
    if N == 0:
        return {
            "segments": [],
            "params": [],
            "Sx": [],
            "Sy": [],
            "joint_tangents": [],
        }

    segments = _collect_segments(P, seam_indices or [])
    segments = _ensure_shared_endpoints(segments)
    if len(segments) == 1:
        t = np.arange(N + 1)
        xy = np.vstack([P, P[0]])
        cs_x = CubicSpline(t, xy[:, 0], bc_type="periodic")
        cs_y = CubicSpline(t, xy[:, 1], bc_type="periodic")
        return {
            "segments": [P],
            "params": [t],
            "Sx": [cs_x],
            "Sy": [cs_y],
            "joint_tangents": [],
        }

    params = [_chord_param(seg) for seg in segments]
    joint_tangents = []
    for i in range(len(segments)):
        prev_seg = segments[(i - 1) % len(segments)]
        curr_seg = segments[i]
        incoming = _segment_tangent(prev_seg, forward=False)
        outgoing = _segment_tangent(curr_seg, forward=True)
        tangent = 0.5 * (incoming + outgoing)
        if len(prev_seg) <= 2:
            tangent = _project_to_direction(tangent, incoming)
        if len(curr_seg) <= 2:
            tangent = _project_to_direction(tangent, outgoing)
        joint_tangents.append(tangent)

    Sx, Sy = [], []
    for i, seg in enumerate(segments):
        t = params[i]
        start_tangent = joint_tangents[i]
        end_tangent = joint_tangents[(i + 1) % len(joint_tangents)]
        Sx.append(CubicSpline(t, seg[:, 0], bc_type=((1, start_tangent[0]), (1, end_tangent[0]))))
        Sy.append(CubicSpline(t, seg[:, 1], bc_type=((1, start_tangent[1]), (1, end_tangent[1]))))

    return {
        "segments": segments,
        "params": params,
        "Sx": Sx,
        "Sy": Sy,
        "joint_tangents": joint_tangents,
    }


def _sample_segments(spline_data, samples_per_seg: int) -> np.ndarray:
    if samples_per_seg <= 0:
        raise ValueError("samples_per_seg must be positive")

    pts = []
    for cs_x, cs_y, t in zip(spline_data["Sx"], spline_data["Sy"], spline_data["params"]):
        if len(t) == 0:
            continue
        n_interval = max(1, len(t) - 1)
        u = np.linspace(t[0], t[-1], n_interval * samples_per_seg, endpoint=False)
        pts.append(np.column_stack([cs_x(u), cs_y(u)]))
    if not pts:
        return np.zeros((0, 2))
    return np.vstack(pts)


def cubic_spline_closed(points: np.ndarray, samples_per_seg: int = 24, *, seam_indices: Sequence[int] | None = None) -> np.ndarray:
    data = build_c1_closed_spline(points, seam_indices)
    return _sample_segments(data, samples_per_seg)


def rounded_rect_area(a: float, b: float, R: float, *, centers=None) -> float:
    """Exact area of the rounded figure defined by ``centers``."""
    if centers is None:
        a2, b2 = a / 2, b / 2
        centers = [
            (-a2 + R, b2 - R),
            (-a2 + R, -b2 + R),
            (a2 - R, -b2 + R),
            (a2 - R, b2 - R),
        ]
    ang_pairs = _arc_angles_from_centers(centers)
    area = 0.0
    for i in range(4):
        cx, cy = centers[i]
        a0, a1 = ang_pairs[i]
        # Circular arc contribution
        area += 0.5 * (
            R * (cx * (math.sin(a1) - math.sin(a0)) - cy * (math.cos(a1) - math.cos(a0)))
            + R * R * (a1 - a0)
        )
        j = (i + 1) % 4
        p1 = (cx + R * math.cos(a1), cy + R * math.sin(a1))
        p2 = (
            centers[j][0] + R * math.cos(ang_pairs[j][0]),
            centers[j][1] + R * math.sin(ang_pairs[j][0]),
        )
        # Tangent line contribution
        area += 0.5 * (p1[0] * p2[1] - p1[1] * p2[0])
    return abs(area)
