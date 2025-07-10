import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main_window import MainWindow

app = QApplication.instance()
if app is None:
    app = QApplication([])


def test_spline_area_close_to_theory():
    win = MainWindow()
    assert win.area_error_percent() < 5.0


def test_arc_center_move_updates():
    win = MainWindow()
    before_contour = win.get_contour().copy()
    new_center = (win.arc_centers[0][0] + 20, win.arc_centers[0][1] + 10)
    win.center_points[0].setPos(*new_center)
    moved_pos = win.center_points[0].pos()
    assert np.allclose([moved_pos.x(), moved_pos.y()], new_center)
    assert not np.array_equal(win.get_contour(), before_contour)


def test_center_marker_move_shifts_group():
    win = MainWindow()
    before_contour = win.get_contour().copy()
    center = win.groups[0].points[2]
    new_pos = center.pos() + QPointF(10, 5)
    center.setPos(new_pos)
    moved_pos = win.groups[0].points[2].pos()
    assert np.allclose([moved_pos.x(), moved_pos.y()], [new_pos.x(), new_pos.y()])
    assert not np.array_equal(win.get_contour(), before_contour)
