import os
import sys
import numpy as np
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PySide6.QtWidgets import QApplication
from main_window import MainWindow


def test_move_center_updates_geometry():
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    orig = win.arc_centers[0].copy()
    win.move_center(0, (orig[0] + 10, orig[1] + 20))
    assert np.allclose(win.arc_centers[0], [orig[0] + 10, orig[1] + 20])
