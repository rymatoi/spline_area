from PySide6.QtWidgets import QWidget, QFormLayout, QDoubleSpinBox, QSpinBox, QLabel


class InspectorWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.layout = QFormLayout()
        self.setLayout(self.layout)
        self.a = self._add_dspinbox("a (ширина)", 50, 2000, main_window.a, self._change_a)
        self.b = self._add_dspinbox("b (высота)", 50, 2000, main_window.b, self._change_b)
        self.R = self._add_dspinbox("R (радиус скругления)", 1, 1000, main_window.R, self._change_R)
        self.scale = self._add_dspinbox("Масштаб", 0.1, 4.0, main_window.scale, self._change_scale, decimals=2, step=0.05)
        self.step = self._add_dspinbox("Шаг дискретизации", 0.1, 100, main_window.step, self._change_step, decimals=2, step=0.1)
        self.layout.addRow(QLabel("<b>Визуализация</b>"))
        self.point_radius = self._add_spinbox("Размер точек", 1, 50, main_window.point_radius, self._change_point_radius)
        self.line_width = self._add_spinbox("Толщина линий", 1, 15, main_window.line_width, self._change_line_width)

        self.error_label = QLabel()
        self.area_label = QLabel()
        self.theory_area_label = QLabel()
        self.layout.addRow("Площадь сплайна:", self.area_label)
        self.layout.addRow("Теоретическая площадь:", self.theory_area_label)
        self.layout.addRow("Ошибка площади:", self.error_label)

        self.update_error()

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
