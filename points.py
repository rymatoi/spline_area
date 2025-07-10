import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QBrush, QPen, QColor
from PySide6.QtWidgets import QGraphicsEllipseItem


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
            if self.offset == 0:
                contour = self.group.get_contour()
                p = np.array([value.x(), value.y()])
                N = len(contour)
                idx_new = int(np.argmin(np.linalg.norm(contour - p, axis=1)))
                delta = (idx_new - self.group.center_idx + N) % N
                if delta > N / 2:
                    delta -= N
                self.group.main_window.move_group_by_delta(self.group, delta, contour)
                self.group.main_window.arc_centers[self.group.arc_num] = (
                    value.x(),
                    value.y(),
                )
                self.group.main_window.redraw_all(preserve_markers=True)
                return value
            contour = self.group.get_contour()
            p = np.array([value.x(), value.y()])
            idx = int(np.argmin(np.linalg.norm(contour - p, axis=1)))
            pair_off = -self.offset
            pair_idx = self.group.offset_list.index(pair_off)
            pair_pt = self.group.points[pair_idx]
            N, c = len(contour), self.group.center_idx
            mir_idx = (2 * c - idx) % N
            pair_pt._syncing = True
            pair_pt.setPos(*contour[mir_idx])
            pair_pt._syncing = False
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
        self.arc_num = arc_num
        self.center_idx = center_idx
        self.offset_list = [-offsets[1], -offsets[0], 0, offsets[0], offsets[1]]
        self.points, self.ready = [], False
        contour = get_contour()
        for i, off in enumerate(self.offset_list):
            if off == 0:
                col, mov = col_c, True
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


class FreePoint(QGraphicsEllipseItem):
    COLOR = QColor(255, 160, 0)

    def __init__(self, main_window, percent):
        self.main_window = main_window
        self.percent = percent
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


class ArcCenterPoint(QGraphicsEllipseItem):
    def __init__(self, main_window, index, radius=4):
        super().__init__(-radius, -radius, 2 * radius, 2 * radius)
        self.main_window = main_window
        self.index = index
        self._syncing = False
        self.setBrush(QBrush(QColor(200, 0, 200)))
        self.setPen(QPen(Qt.black, 1))
        self.setZValue(1)
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.ItemSendsScenePositionChanges, True)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.ItemPositionHasChanged and not self._syncing:
            pos = value if isinstance(value, QPointF) else self.pos()
            self.main_window.arc_centers[self.index] = (pos.x(), pos.y())
            self.main_window.schedule_redraw()
            return value
        return super().itemChange(change, value)
