from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QLabel
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QMouseEvent


class MouseLineEdit(QLineEdit):
    mousePressed = pyqtSignal(QMouseEvent)
    mouseMoved = pyqtSignal(QMouseEvent)
    mouseReleased = pyqtSignal(QMouseEvent)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mouse_is_pressed = False
        self.mouse_is_moving = False

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.mouse_is_pressed = True
        self.mousePressed.emit(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self.mouse_is_pressed:
            self.mouse_is_moving = True
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.mouseMoved.emit(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self.mouse_is_moving:
            self.setCursor(Qt.CursorShape.IBeamCursor)
        self.mouse_is_pressed = False
        self.mouse_is_moving = False
        self.mouseReleased.emit(event)


class ImageControl(QWidget):
    value_changed = pyqtSignal(float)

    def __init__(self, text: str, initial_value: float, precision: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mouse_pressed = False
        self.last_mouse_position = None

        self.text = text
        self.value = initial_value
        self.precision = precision
        self.multiplier = 10**-precision

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()

        info_label = QLabel(self.text)
        main_layout.addWidget(info_label)

        self.mouse_line_edit = MouseLineEdit("{: .{precision}f}".format(self.value, precision=self.precision))
        self.mouse_line_edit.textChanged.connect(self.on_value_changed)
        self.mouse_line_edit.mousePressed.connect(self.on_mouse_pressed)
        self.mouse_line_edit.mouseMoved.connect(self.on_mouse_moved)
        self.mouse_line_edit.mouseReleased.connect(self.on_mouse_released)
        main_layout.addWidget(self.mouse_line_edit)

        self.setLayout(main_layout)

    def on_mouse_pressed(self, event):
        self.last_mouse_position = event.pos()

    def on_mouse_moved(self, event):
        if self.last_mouse_position is not None:
            delta = event.pos().x() - self.last_mouse_position.x()
            sender = self.sender()
            self.value = float(sender.text()) + delta * self.multiplier
            sender.setText("{: .{precision}f}".format(self.value, precision=self.precision))
            self.last_mouse_position = event.pos()

    def on_mouse_released(self, _event):
        self.last_mouse_position = None

    def on_value_changed(self):
        self.value_changed.emit(self.value)
