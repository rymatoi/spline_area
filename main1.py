import sys, math, numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
    QDockWidget, QWidget, QFormLayout, QDoubleSpinBox, QSpinBox, QLabel
)
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPen, QBrush, QPainterPath, QColor, QFont, QTransform, QPainter
from scipy.interpolate import CubicSpline


def arc_geom_points(a, b, R):
    """Возвращает [(центр, начало, конец)] для каждой дуги"""
    a2, b2 = a / 2, b / 2
    r2 = math.sqrt(2) / 2
    # Считаем (центр дуги, начало дуги, конец дуги) в порядке UL, LL, LR, UR
    arcs = []
    # UL
    arcs.append((
        (-a2 + R - R * r2, b2 - R + R * r2),  # центр
        (-a2 + R, b2),  # начало
        (-a2, b2 - R)  # конец
    ))
    # LL
    arcs.append((
        (-a2 + R - R * r2, -b2 + R - R * r2),
        (-a2, -b2 + R),
        (-a2 + R, -b2)
    ))
    # LR
    arcs.append((
        (a2 - R + R * r2, -b2 + R - R * r2),
        (a2 - R, -b2),
        (a2, -b2 + R)
    ))
    # UR
    arcs.append((
        (a2 - R + R * r2, b2 - R + R * r2),
        (a2, b2 - R),
        (a2 - R, b2)
    ))
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
        arc(-a2 + R, b2 - R, math.pi / 2, math.pi),  # UL
        line([-a2, b2 - R], [-a2, -b2 + R]),  # left
        arc(-a2 + R, -b2 + R, math.pi, 3 * math.pi / 2),  # LL
        line([-a2 + R, -b2], [a2 - R, -b2]),  # bottom
        arc(a2 - R, -b2 + R, 3 * math.pi / 2, 2 * math.pi),  # LR
        line([a2, -b2 + R], [a2, b2 - R]),  # right
        arc(a2 - R, b2 - R, 0.0, math.pi / 2),  # UR
        line([a2 - R, b2], [-a2 + R, b2]),  # top
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
    t = np.arange(N + 1)  # closed loop
    xy = np.vstack([P, P[0]])
    ts_dense = np.linspace(0, N, N * samples_per_seg, endpoint=False)
    cs_x = CubicSpline(t, xy[:, 0], bc_type='periodic')
    cs_y = CubicSpline(t, xy[:, 1], bc_type='periodic')
    return np.column_stack([cs_x(ts_dense), cs_y(ts_dense)])


class GroupPoint(QGraphicsEllipseItem):
    def __init__(self, group, offset, color, movable=True, radius=6):
        super().__init__(-radius, -radius, 2 * radius, 2 * radius)
        self.group, self.offset = group, offset
        self.movable, self._syncing = movable, False
        self.setBrush(QBrush(color))
        self.setPen(QPen(Qt.black, 1))
        self.setZValue(2)
        if self.movable:
            self.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)
            self.setFlag(QGraphicsEllipseItem.ItemSendsScenePositionChanges, True)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.ItemPositionChange and self.movable:
            if self._syncing or not self.group.ready:
                return value
            contour = self.group.get_contour()
            p = np.array([value.x(), value.y()])
            idx = int(np.argmin(np.linalg.norm(contour - p, axis=1)))
            if self.offset != 0:
                pair_off = -self.offset
                pair_idx = self.group.offset_list.index(pair_off)
                pair_pt = self.group.points[pair_idx]
                N, c = len(contour), self.group.center_idx
                mir_idx = (2 * c - idx) % N
                pair_pt._syncing = True
                pair_pt.setPos(*contour[mir_idx])
                pair_pt._syncing = False
            N, c = len(contour), self.group.center_idx
            delta = (idx - c + N) % N
            if delta > N / 2:
                delta -= N
            self.group.main_window.propagate_move(self.group, self.offset, int(delta))
            self.group.main_window._draw_spline()
            return QPointF(*contour[idx])
        return super().itemChange(change, value)


class GroupOfPoints:
    def __init__(self, scene, main_window, get_contour, center_idx, offsets, col_c, col_n, col_f, positions=None,
                 arc_num=0):
        self.scene, self.main_window = scene, main_window
        self._contour = get_contour
        self.center_idx = center_idx
        self.offset_list = [-offsets[1], -offsets[0], 0, offsets[0], offsets[1]]
        self.points, self.ready = [], False
        contour = get_contour()
        for i, off in enumerate(self.offset_list):
            if off == 0:
                col, mov = col_c, False
            elif abs(off) == offsets[0]:
                col, mov = col_n, True
            else:
                col, mov = col_f, True
            pt = GroupPoint(self, off, col, mov, radius=main_window.point_radius)
            self.points.append(pt)
            if positions is not None:
                pt.setPos(*positions[i])
            else:
                pos = main_window._marker_position_for_offset(contour, center_idx, off, offsets, arc_num=arc_num)
                pt.setPos(*pos)
            scene.addItem(pt)
        self.ready = True

    def get_contour(self):
        return self._contour()


class InspectorWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.layout = QFormLayout()
        self.setLayout(self.layout)
        self.a = self._add_dspinbox("a (ширина)", 50, 2000, main_window.a, self._change_a)
        self.b = self._add_dspinbox("b (высота)", 50, 2000, main_window.b, self._change_b)
        self.R = self._add_dspinbox("R (радиус скругления)", 1, 1000, main_window.R, self._change_R)
        self.scale = self._add_dspinbox("Масштаб", 0.1, 4.0, main_window.scale, self._change_scale, decimals=2,
                                        step=0.05)
        self.step = self._add_dspinbox("Шаг дискретизации", 0.1, 100, main_window.step, self._change_step, decimals=2,
                                       step=0.1)
        self.layout.addRow(QLabel("<b>Визуализация</b>"))
        self.point_radius = self._add_spinbox("Размер точек", 1, 50, main_window.point_radius,
                                              self._change_point_radius)
        self.line_width = self._add_spinbox("Толщина линий", 1, 15, main_window.line_width, self._change_line_width)

        self.error_label = QLabel()

        self.area_label = QLabel()
        self.theory_area_label = QLabel()
        self.layout.addRow("Площадь сплайна:", self.area_label)
        self.layout.addRow("Теоретическая площадь:", self.theory_area_label)
        self.layout.addRow("Ошибка площади:", self.error_label)

        self.update_error()  # начальное обновление

    def update_error(self):
        err = self.main_window.area_error_percent()
        area = self.main_window.spline_area()
        theory = self.main_window.theoretical_area()
        self.area_label.setText(f"{area:.2f}")
        self.theory_area_label.setText(f"{theory:.2f}")
        self.error_label.setText(f"{err:.2f}%")

    def _add_dspinbox(self, label, mn, mx, val, slot, decimals=1, step=1.0):
        w = QDoubleSpinBox()
        w.setRange(mn, mx)
        w.setDecimals(decimals)
        w.setSingleStep(step)
        w.setValue(val)
        w.valueChanged.connect(slot)
        self.layout.addRow(label, w)
        return w

    def _add_spinbox(self, label, mn, mx, val, slot):
        w = QSpinBox()
        w.setRange(mn, mx)
        w.setValue(val)
        w.valueChanged.connect(slot)
        self.layout.addRow(label, w)
        return w

    def _change_a(self, v):
        self.main_window.a = v
        self.main_window.redraw_all(preserve_markers=True)
        self.update_error()

    def _change_b(self, v):
        self.main_window.b = v
        self.main_window.redraw_all(preserve_markers=True)
        self.update_error()

    def _change_R(self, v):
        self.main_window.R = v
        self.main_window.redraw_all(preserve_markers=True)
        self.update_error()

    def _change_scale(self, v):
        self.main_window.scale = v
        self.main_window._apply_transform()
        self.update_error()

    def _change_point_radius(self, v):
        self.main_window.point_radius = v
        self.main_window.redraw_all(preserve_markers=True)
        self.main_window.update_free_points_radius()
        self.update_error()

    def _change_line_width(self, v):
        self.main_window.line_width = v
        self.main_window.redraw_all(preserve_markers=True)
        self.update_error()

    def _change_step(self, v):
        self.main_window.step = v
        self.main_window.redraw_all(preserve_markers=True)
        self.update_error()


# --- Свободная точка (двигается только по контуру, участвует в сплайне) ---
class FreePoint(QGraphicsEllipseItem):
    COLOR = QColor(255, 160, 0)

    # убрали фиксированный RADIUS

    def __init__(self, main_window, percent):
        self.main_window = main_window
        self.percent = percent  # процент по длине контура (0..1)
        radius = main_window.point_radius
        super().__init__(-radius, -radius, 2 * radius, 2 * radius)
        self.setBrush(QBrush(self.COLOR))
        self.setPen(QPen(Qt.black, 1))
        self.setZValue(10)
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.ItemSendsScenePositionChanges, True)
        self.update_position()

    def update_position(self):
        contour = self.main_window.get_contour()
        idx = int(round(self.percent * len(contour))) % len(contour)
        self.setPos(*contour[idx])

    def update_radius(self):
        r = self.main_window.point_radius
        self.setRect(-r, -r, 2 * r, 2 * r)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.ItemPositionChange:
            contour = self.main_window.get_contour()
            p = np.array([value.x(), value.y()])
            idx = int(np.argmin(np.linalg.norm(contour - p, axis=1)))
            self.percent = idx / len(contour)
            self.main_window._draw_spline()
            return QPointF(*contour[idx])
        return super().itemChange(change, value)


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
        self._inspector = InspectorWidget(self)
        dock = QDockWidget("Параметры", self)
        dock.setWidget(self._inspector)
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self._marker_states = None  # для сохранения позиций при изменениях
        self.redraw_all()

        self._apply_transform()

    def update_free_points_radius(self):
        for fp in self.free_points:
            fp.update_radius()

    def eventFilter(self, obj, event):
        from PySide6.QtGui import QMouseEvent
        if isinstance(event, QMouseEvent) and event.type() == QMouseEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton and event.modifiers() & Qt.ControlModifier:
                pos_view = event.position()
                pos_scene = self.view.mapToScene(int(pos_view.x()), int(pos_view.y()))
                contour = self.get_contour()
                arr = np.array([pos_scene.x(), pos_scene.y()])

                # Удаление free точки если кликнули по ней
                for fp in self.free_points:
                    if (fp.pos() - pos_scene).manhattanLength() < self.point_radius + 3:
                        self.scene.removeItem(fp)
                        self.free_points.remove(fp)
                        self._draw_spline()
                        return True

                # Составляем все точки (percent) — маркеры и free
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

                # Ищем ближайший отрезок
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
                        # Найти в self.free_points тот, чей percent == p1
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
        return rounded_rect_points(self.a, self.b, self.R, step=self.step)

    def arc_center_indices(self, contour):
        a2, b2, R = self.a / 2, self.b / 2, self.R
        r2 = math.sqrt(2) / 2
        # центры дуг (геометрически, в массиве точек ищем ближайшие)
        centers_geom = [(-a2 + R - R * r2, b2 - R + R * r2),
                        (-a2 + R - R * r2, -b2 + R - R * r2),
                        (a2 - R + R * r2, -b2 + R - R * r2),
                        (a2 - R + R * r2, b2 - R + R * r2)]
        return [int(np.argmin(np.linalg.norm(contour - np.array(pt), axis=1)))
                for pt in centers_geom]

    def _marker_position_for_offset(self, contour, _, offset, offsets, arc_num):
        """
        arc_num — номер дуги (0..3).
        Белая — центр дуги;
        Зелёные — концы дуги;
        Синие — между центром и концами (40 % к краю).
        """

        # --- 1. Берём геометрию дуги и СРАЗУ превращаем в numpy-вектора ------------
        center_xy, start_xy, end_xy = [
            np.array(p) for p in arc_geom_points(self.a, self.b, self.R)[arc_num]
        ]

        # --- 2. Индексы ближайших дискретных точек контура -------------------------
        center_idx = np.argmin(np.linalg.norm(contour - center_xy, axis=1))
        start_idx = np.argmin(np.linalg.norm(contour - start_xy, axis=1))
        end_idx = np.argmin(np.linalg.norm(contour - end_xy, axis=1))

        # --- 3. Выбор позиции по offset -------------------------------------------
        if offset == 0:  # белая
            return contour[center_idx]

        elif offset == offsets[1]:  # дальняя (+)
            return contour[end_idx]
        elif offset == -offsets[1]:  # дальняя (–)
            return contour[start_idx]

        elif offset == offsets[0]:  # ближняя (+)
            pos = 0.6 * center_xy + 0.4 * end_xy
        elif offset == -offsets[0]:  # ближняя (–)
            pos = 0.6 * center_xy + 0.4 * start_xy
        else:  # запасной случай
            pos = center_xy

        idx = np.argmin(np.linalg.norm(contour - pos, axis=1))
        return contour[idx]

    def redraw_all(self, preserve_markers=False):
        contour = self.get_contour()
        marker_params = None

        if preserve_markers and self.groups:
            # Сохраняем "процент" вдоль контура для каждой точки
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
            # --- сохраняем проценты свободных точек
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
        col = dict(center=QColor(255, 255, 255),
                   near=QColor(0, 120, 255),
                   far=QColor(0, 200, 80))
        arc_indices = self.arc_center_indices(contour)
        for arc_num, idx in enumerate(arc_indices):
            positions = None
            if marker_params is not None:
                positions = []
                N = len(contour)
                for j, percent in enumerate(marker_params[arc_num]):
                    if j == 2:  # индекс 2 → белая точка
                        # белая точка — всегда новый центр, НЕ используем percent
                        positions.append(
                            tuple(self._marker_position_for_offset(
                                contour, 0, 0, offsets, arc_num)))
                    else:
                        new_idx = int(round(percent * N)) % N
                        positions.append(tuple(contour[new_idx]))
            grp = GroupOfPoints(self.scene, self, self.get_contour,
                                idx, offsets,
                                col['center'], col['near'], col['far'],
                                positions=positions, arc_num=arc_num)
            self.groups.append(grp)

        self.free_points = []
        for percent in self._free_points_percents:
            fp = FreePoint(self, percent)
            self.scene.addItem(fp)
            self.free_points.append(fp)

        self._prev_contour = contour.copy()
        self._draw_background(contour)
        self._draw_contour(contour)
        self._draw_spline()

    def get_all_marker_positions(self):
        # Получаем все точки (опорные и free), отмечая их percent вдоль контура
        contour = self.get_contour()
        points = []
        # Опорные точки (всегда первыми, чтобы не терять индексы)
        for grp in self.groups:
            for pt in grp.points:
                p = pt.pos()
                arr = np.array([p.x(), p.y()])
                idx = int(np.argmin(np.linalg.norm(contour - arr, axis=1)))
                percent = idx / len(contour)
                points.append((percent, (p.x(), p.y())))
        # Free points
        for fp in self.free_points:
            percent = fp.percent % 1.0
            p = fp.pos()
            points.append((percent, (p.x(), p.y())))
        # Сортируем по percent вдоль обхода
        points.sort(key=lambda x: x[0])
        # Возвращаем только координаты (x, y) в правильном порядке
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
        arc_info = [(-a2 + R, b2 - R, math.pi / 2, math.pi),
                    (-a2 + R, -b2 + R, math.pi, 3 * math.pi / 2),
                    (a2 - R, -b2 + R, 3 * math.pi / 2, 0.0),
                    (a2 - R, b2 - R, 0.0, math.pi / 2)]
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
        # Получаем контрольные точки
        pts = self.get_all_marker_positions()
        if len(pts) < 4:
            return 0.0

        pts = np.array(pts)
        N = len(pts)
        t = np.arange(N + 1)
        xy = np.vstack([pts, pts[0]])
        ts_dense = np.linspace(0, N, N, endpoint=True)  # т. к. будет N сегментов

        # Строим сплайн
        cs_x = CubicSpline(t, xy[:, 0], bc_type='periodic')
        cs_y = CubicSpline(t, xy[:, 1], bc_type='periodic')

        # Формируем Безье-сегменты
        bezier_segments = self.spline_to_bezier(cs_x, cs_y, t)

        # Суммируем площади всех сегментов
        area = sum(self.eval_segment_area(seg) for seg in bezier_segments)
        return abs(area) / (self.scale ** 2)  # делим на квадрат масштаба, если рисуем с масштабом

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
            p2_y = p3_y - dt * cs_y.derivative()(t_end) / 3

            bezier_segment = np.array([
                [p0_x, p0_y],
                [p1_x, p1_y],
                [p2_x, p2_y],
                [p3_x, p3_y]
            ])
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
        area = (a - 2 * R) * (b - 2 * R) + math.pi * R * R
        return area

    def area_error_percent(self):
        theory = self.theoretical_area()
        if theory == 0:
            return 0.0
        area = self.spline_area()
        return 100.0 * abs(area - theory) / theory


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
