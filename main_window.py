import math
import numpy as np
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPen, QBrush, QPainterPath, QColor, QFont, QTransform, QPainter
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QMainWindow, QDockWidget

from geometry import arc_geom_points, rounded_rect_points, cubic_spline_closed
from scipy.interpolate import CubicSpline
from points import GroupOfPoints, FreePoint, CenterPoint
from inspector import InspectorWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Интерактивный инспектор: фигура и визуал (позиции сохраняются)")
        self.a, self.b, self.R = 400, 200, 60
        self.scale = 1.5
        self.point_radius = 7
        self.line_width = 3
        self.step = 1.0
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.setCentralWidget(self.view)
        self.groups = []
        self.free_points = []
        self.spline_path = None
        self.view.viewport().installEventFilter(self)
        self.reset_arc_centers()
        self._inspector = InspectorWidget(self)
        dock = QDockWidget("Параметры", self)
        dock.setWidget(self._inspector)
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self._marker_states = None
        self.center_points = []
        self.redraw_all()
        self._apply_transform()

    def reset_arc_centers(self):
        self.arc_centers = [
            np.array([-self.a / 2 + self.R, self.b / 2 - self.R]),
            np.array([-self.a / 2 + self.R, -self.b / 2 + self.R]),
            np.array([self.a / 2 - self.R, -self.b / 2 + self.R]),
            np.array([self.a / 2 - self.R, self.b / 2 - self.R]),
        ]

    def update_free_points_radius(self):
        for fp in self.free_points:
            fp.update_radius()
        for cp in self.center_points:
            cp.update_radius()

    def move_center(self, index, pos):
        """Update a circle center and redraw without recursion."""
        self.arc_centers[index] = np.array(pos)
        for cp in self.center_points:
            cp._syncing = True
        self.redraw_all(preserve_markers=True)
        for cp in self.center_points:
            cp._syncing = False

    def eventFilter(self, obj, event):
        from PySide6.QtGui import QMouseEvent
        if isinstance(event, QMouseEvent) and event.type() == QMouseEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton and event.modifiers() & Qt.ControlModifier:
                pos_view = event.position()
                pos_scene = self.view.mapToScene(int(pos_view.x()), int(pos_view.y()))
                contour = self.get_contour()
                arr = np.array([pos_scene.x(), pos_scene.y()])

                for fp in self.free_points:
                    if (fp.pos() - pos_scene).manhattanLength() < self.point_radius + 3:
                        self.scene.removeItem(fp)
                        self.free_points.remove(fp)
                        self._draw_spline()
                        return True

                all_percents = []
                for grp in self.groups:
                    for pt in grp.points:
                        p = pt.pos()
                        arrp = np.array([p.x(), p.y()])
                        idx = int(np.argmin(np.linalg.norm(contour - arrp, axis=1)))
                        all_percents.append(idx / len(contour))
                for fp in self.free_points:
                    all_percents.append(fp.percent)
                all_percents = sorted([p % 1.0 for p in all_percents])

                min_dist = float('inf')
                insert_index = 0
                new_percent = 0
                N = len(all_percents)
                for i in range(N):
                    p1 = all_percents[i]
                    p2 = all_percents[(i + 1) % N]
                    idx1 = int(round(p1 * len(contour))) % len(contour)
                    idx2 = int(round(p2 * len(contour))) % len(contour)
                    xy1 = contour[idx1]
                    xy2 = contour[idx2]
                    v = xy2 - xy1
                    u = arr - xy1
                    t = np.clip(np.dot(u, v) / (np.dot(v, v) + 1e-12), 0, 1)
                    proj = xy1 + t * v
                    dist = np.linalg.norm(proj - arr)
                    if dist < min_dist:
                        min_dist = dist
                        proj_idx = int(np.argmin(np.linalg.norm(contour - proj, axis=1)))
                        new_percent = proj_idx / len(contour)
                        insert_index = 0
                        for j, fp in enumerate(self.free_points):
                            if abs(fp.percent - p1) < 1e-5:
                                insert_index = j + 1

                fp = FreePoint(self, new_percent)
                self.scene.addItem(fp)
                self.free_points.insert(insert_index, fp)
                self._draw_spline()
                return True
        return False

    def _apply_transform(self):
        t = QTransform()
        t.scale(self.scale, -self.scale)
        self.view.setTransform(t)

    def get_contour(self):
        return rounded_rect_points(
            self.a,
            self.b,
            self.R,
            step=self.step,
            centers=self.arc_centers,
        )

    def arc_center_indices(self, contour):
        arcs = arc_geom_points(self.a, self.b, self.R, centers=self.arc_centers)
        centers_geom = [arc[0] for arc in arcs]
        return [int(np.argmin(np.linalg.norm(contour - np.array(pt), axis=1))) for pt in centers_geom]

    def _marker_position_for_offset(self, contour, _, offset, offsets, arc_num):
        center_xy, start_xy, end_xy = [np.array(p) for p in arc_geom_points(self.a, self.b, self.R, centers=self.arc_centers)[arc_num]]
        center_idx = np.argmin(np.linalg.norm(contour - center_xy, axis=1))
        start_idx = np.argmin(np.linalg.norm(contour - start_xy, axis=1))
        end_idx = np.argmin(np.linalg.norm(contour - end_xy, axis=1))
        if offset == 0:
            return contour[center_idx]
        elif offset == offsets[1]:
            return contour[end_idx]
        elif offset == -offsets[1]:
            return contour[start_idx]
        elif offset == offsets[0]:
            pos = 0.6 * center_xy + 0.4 * end_xy
        elif offset == -offsets[0]:
            pos = 0.6 * center_xy + 0.4 * start_xy
        else:
            pos = center_xy
        idx = np.argmin(np.linalg.norm(contour - pos, axis=1))
        return contour[idx]

    def redraw_all(self, preserve_markers=False):
        contour = self.get_contour()
        marker_params = None

        if preserve_markers and self.groups:
            old_contour = self._prev_contour if hasattr(self, '_prev_contour') else contour
            marker_params = []
            for grp in self.groups:
                group_param = []
                for pt in grp.points:
                    p = np.array([pt.pos().x(), pt.pos().y()])
                    dists = np.linalg.norm(old_contour - p, axis=1)
                    idx = np.argmin(dists)
                    percent = idx / len(old_contour)
                    group_param.append(percent)
                marker_params.append(group_param)
            self._free_points_percents = []
            for fp in self.free_points:
                p = np.array([fp.pos().x(), fp.pos().y()])
                idx = np.argmin(np.linalg.norm(old_contour - p, axis=1))
                percent = idx / len(old_contour)
                self._free_points_percents.append(percent)
        else:
            self._free_points_percents = [fp.percent for fp in self.free_points]

        self.scene.clear()
        self.spline_path = None
        self.groups.clear()

        offsets = [5, 15]
        col = dict(center=QColor(255, 255, 255), near=QColor(0, 120, 255), far=QColor(0, 200, 80))
        arc_indices = self.arc_center_indices(contour)
        for arc_num, idx in enumerate(arc_indices):
            positions = None
            if marker_params is not None:
                positions = []
                N = len(contour)
                for j, percent in enumerate(marker_params[arc_num]):
                    if j == 2:
                        positions.append(tuple(self._marker_position_for_offset(contour, 0, 0, offsets, arc_num)))
                    else:
                        new_idx = int(round(percent * N)) % N
                        positions.append(tuple(contour[new_idx]))
            grp = GroupOfPoints(self.scene, self, self.get_contour, idx, offsets, col['center'], col['near'], col['far'], positions=positions, arc_num=arc_num)
            self.groups.append(grp)

        self.free_points = []
        for percent in self._free_points_percents:
            fp = FreePoint(self, percent)
            self.scene.addItem(fp)
            self.free_points.append(fp)

        self.center_points = []
        for i in range(4):
            cp = CenterPoint(self, i)
            self.scene.addItem(cp)
            cp.finish_init()
            self.center_points.append(cp)

        self._prev_contour = contour.copy()
        self._draw_background(contour)
        self._draw_contour(contour)
        self._draw_spline()

    def get_all_marker_positions(self):
        contour = self.get_contour()
        points = []
        for grp in self.groups:
            for pt in grp.points:
                p = pt.pos()
                arr = np.array([p.x(), p.y()])
                idx = int(np.argmin(np.linalg.norm(contour - arr, axis=1)))
                percent = idx / len(contour)
                points.append((percent, (p.x(), p.y())))
        for fp in self.free_points:
            percent = fp.percent % 1.0
            p = fp.pos()
            points.append((percent, (p.x(), p.y())))
        points.sort(key=lambda x: x[0])
        return [xy for percent, xy in points]

    def _draw_spline(self):
        pts = self.get_all_marker_positions()
        if len(pts) < 4:
            return
        spline_pts = cubic_spline_closed(np.array(pts), samples_per_seg=24)
        path = QPainterPath()
        path.moveTo(*spline_pts[0])
        for x, y in spline_pts[1:]:
            path.lineTo(x, y)
        path.closeSubpath()
        if self.spline_path is not None:
            try:
                self.scene.removeItem(self.spline_path)
            except RuntimeError:
                pass
        self.spline_path = self.scene.addPath(path, QPen(Qt.red, self.line_width))
        self._inspector.update_error()

    def _draw_contour(self, contour):
        path = QPainterPath()
        path.moveTo(*contour[0])
        for x, y in contour[1:]:
            path.lineTo(x, y)
        path.closeSubpath()
        self.scene.addPath(path, QPen(Qt.blue, self.line_width))

    def _draw_background(self, contour):
        a2, b2, R, m = self.a / 2, self.b / 2, self.R, 40
        pen_axis = QPen(Qt.darkGray, 1)
        pen_axis.setCosmetic(True)
        self.scene.addLine(-a2 - m, 0, a2 + m, 0, pen_axis)
        self.scene.addLine(0, -b2 - m, 0, b2 + m, pen_axis)
        self._arrow(a2 + m, 0)
        self._arrow(0, b2 + m, vertical=True)
        f = QFont("Tahoma", 10)

        def add_text(txt, x, y):
            item = self.scene.addText(txt, f)
            item.setPos(x, y)
            item.setTransform(QTransform().scale(1, -1))

        add_text("x", a2 + m - 15, -18)
        add_text("y", 8, b2 + m - 20)
        add_text(" a", a2 - 6, 2)
        add_text("-a", -a2 - 22, 2)
        add_text(" b", 4, b2 - 14)
        add_text("-b", 4, -b2 - 18)
        arc_info = [
            (*self.arc_centers[0], math.pi / 2, math.pi),
            (*self.arc_centers[1], math.pi, 3 * math.pi / 2),
            (*self.arc_centers[2], 3 * math.pi / 2, 2 * math.pi),
            (*self.arc_centers[3], 0.0, math.pi / 2),
        ]
        pen_c = QPen(Qt.darkGray, 1, Qt.DotLine)
        pen_c.setCosmetic(True)
        pen_r = QPen(Qt.red, 1, Qt.DashLine)
        pen_r.setCosmetic(True)
        for cx, cy, a0, a1 in arc_info:
            s = 6
            self.scene.addLine(cx - s, cy, cx + s, cy, pen_c)
            self.scene.addLine(cx, cy - s, cx, cy + s, pen_c)
            p0 = QPointF(cx + R * math.cos(a0), cy + R * math.sin(a0))
            p1 = QPointF(cx + R * math.cos(a1), cy + R * math.sin(a1))
            self.scene.addLine(cx, cy, p0.x(), p0.y(), pen_r)
            self.scene.addLine(cx, cy, p1.x(), p1.y(), pen_r)

    def _arrow(self, x, y, *, vertical=False):
        pen = QPen(Qt.darkGray, 1)
        pen.setCosmetic(True)
        if vertical:
            self.scene.addLine(x, y, x - 5, y - 10, pen)
            self.scene.addLine(x, y, x + 5, y - 10, pen)
        else:
            self.scene.addLine(x, y, x - 10, y - 5, pen)
            self.scene.addLine(x, y, x - 10, y + 5, pen)

    def propagate_move(self, src_group, moved_offset, delta):
        contour = self.get_contour()
        N = len(contour)
        for grp in self.groups:
            if grp is src_group:
                continue
            c = grp.center_idx
            tgt_idx = (c + delta) % N
            p_idx = grp.offset_list.index(moved_offset)
            pt = grp.points[p_idx]
            pt._syncing = True
            pt.setPos(*contour[tgt_idx])
            if moved_offset != 0:
                pair_off = -moved_offset
                pair_idx = grp.offset_list.index(pair_off)
                q = grp.points[pair_idx]
                mir_idx = (2 * c - tgt_idx) % N
                q._syncing = True
                q.setPos(*contour[mir_idx])
                q._syncing = False
            pt._syncing = False

    def spline_area(self):
        pts = self.get_all_marker_positions()
        if len(pts) < 4:
            return 0.0
        pts = np.array(pts)
        N = len(pts)
        t = np.arange(N + 1)
        xy = np.vstack([pts, pts[0]])
        ts_dense = np.linspace(0, N, N, endpoint=True)
        cs_x = CubicSpline(t, xy[:, 0], bc_type='periodic')
        cs_y = CubicSpline(t, xy[:, 1], bc_type='periodic')
        bezier_segments = self.spline_to_bezier(cs_x, cs_y, t)
        area = sum(self.eval_segment_area(seg) for seg in bezier_segments)
        return abs(area)

    def spline_to_bezier(self, cs_x, cs_y, t_range):
        bezier_segments = []
        for i in range(len(t_range) - 1):
            t_start = t_range[i]
            t_end = t_range[i + 1]
            dt = t_end - t_start
            p0_x = cs_x(t_start)
            p0_y = cs_y(t_start)
            p1_x = p0_x + dt * cs_x.derivative()(t_start) / 3
            p1_y = p0_y + dt * cs_y.derivative()(t_start) / 3
            p3_x = cs_x(t_end)
            p3_y = cs_y(t_end)
            p2_x = p3_x - dt * cs_x.derivative()(t_end) / 3
            p2_y = cs_y(t_end) - dt * cs_y.derivative()(t_end) / 3
            bezier_segment = np.array([[p0_x, p0_y], [p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y]])
            bezier_segments.append(bezier_segment)
        return bezier_segments

    def eval_segment_area(self, segment):
        x = segment[:, 0]
        y = segment[:, 1]
        return 0.5 * (1 / 20) * (
                12 * x[0] * y[1] + 6 * x[0] * y[2] + 2 * x[0] * y[3] -
                12 * x[1] * y[0] + 6 * x[1] * y[2] + 6 * x[1] * y[3] -
                6 * x[2] * y[0] - 6 * x[2] * y[1] + 12 * x[2] * y[3] -
                2 * x[3] * y[0] - 6 * x[3] * y[1] - 12 * x[3] * y[2]
        )

    def theoretical_area(self):
        a, b, R = self.a, self.b, self.R
        return a * b - (4 - math.pi) * (R ** 2)

    def area_error_percent(self):
        theory = self.theoretical_area()
        if theory == 0:
            return 0.0
        area = self.spline_area()
        return 100.0 * abs(area - theory) / theory
