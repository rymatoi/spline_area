import numpy as np
from PySide6.QtCore import QPointF, Qt
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
        self.offsets = offsets
        self.arc_num = arc_num
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

    def update_positions(self, contour):
        for pt, off in zip(self.points, self.offset_list):
            pos = self.main_window._marker_position_for_offset(
                contour, self.center_idx, off, self.offsets, arc_num=self.arc_num
            )
            pt._syncing = True
            pt.setPos(*pos)
            pt._syncing = False


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

    def finish_init(self):
        """Enable normal behavior after initial creation."""
        self._syncing = False

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


class CenterPoint(QGraphicsEllipseItem):
    COLOR = QColor(200, 50, 200)

    def __init__(self, main_window, index):
        radius = main_window.point_radius
        super().__init__(-radius, -radius, 2 * radius, 2 * radius)
        self.main_window = main_window
        self.index = index
        self._syncing = True
        self._initialized = False
        self.setBrush(QBrush(self.COLOR))
        self.setPen(QPen(Qt.black, 1))
        self.setZValue(5)
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.ItemSendsScenePositionChanges, True)
        self.update_position()

    def finish_init(self):
        """Enable normal behavior after initial creation."""
        self._syncing = False
        self._initialized = True

    def update_position(self):
        cx, cy = self.main_window.arc_centers[self.index]
        self._syncing = True
        self.setPos(cx, cy)
        self._syncing = False

    def update_radius(self):
        r = self.main_window.point_radius
        self.setRect(-r, -r, 2 * r, 2 * r)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.ItemPositionChange:
            if self._syncing or not self._initialized:
                return value
            pos = (value.x(), value.y())
            self.main_window.move_center(self.index, pos)
            return QPointF(*self.main_window.arc_centers[self.index])
        return super().itemChange(change, value)
