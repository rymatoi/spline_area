import numpy as np
from PySide6.QtWidgets import QApplication
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main_window import MainWindow

app = QApplication.instance()
if app is None:
    app = QApplication([])

def idx_from_pos(contour, pt):
    arr = np.array([pt.pos().x(), pt.pos().y()])
    return int(np.argmin(np.linalg.norm(contour - arr, axis=1)))


def test_spline_area_close_to_theory():
    win = MainWindow()
    err = win.area_error_percent()
    assert err < 1.0


def test_center_move_propagates():
    win = MainWindow()
    contour = win.get_contour()
    grp0 = win.groups[0]
    delta = 10
    indices_before = [idx_from_pos(contour, pt) for pt in grp0.points]
    centers_before = [g.center_idx for g in win.groups]
    win.move_group_by_delta(grp0, delta)
    win.propagate_move(grp0, 0, delta)
    contour = win.get_contour()
    indices_after = [idx_from_pos(contour, pt) for pt in grp0.points]
    N = len(contour)
    for before, after in zip(indices_before, indices_after):
        assert after == (before + delta) % N
    # other groups centers moved
    for grp, before_c in zip(win.groups[1:], centers_before[1:]):
        assert grp.center_idx == (before_c + delta) % N

