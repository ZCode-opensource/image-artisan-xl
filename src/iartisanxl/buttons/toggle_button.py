from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPainter
from PyQt6.QtWidgets import QFrame


class ToggleButton(QFrame):
    clicked = pyqtSignal()

    def __init__(self, icon, button_width: int = 20, button_height: int = 20):
        super().__init__()

        self.button_width = button_width
        self.button_height = button_height
        self.toggled_on = False
        self.setFixedSize(self.button_width, self.button_height)
        self.icon = icon

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.TextAntialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )

        icon = QIcon(str(self.icon))
        icon.paint(
            painter,
            QRectF(0, 0, self.button_width, self.button_height).toRect(),
            Qt.AlignmentFlag.AlignCenter,
            state=QIcon.State.Off,
        )

    def set_toggle(self, value: bool):
        self.toggled_on = value

        if self.toggled_on:
            self.setStyleSheet(
                "color: rgba(100, 100, 100, 0.5); background-color: rgba(57, 35, 98, 1); border-radius: 0px"
            )
        else:
            self.setStyleSheet(
                "color: rgba(100, 100, 100, 0.5); background-color: rgba(0, 0, 0, 0); border-radius: 0px"
            )

    def enterEvent(self, event):
        self.setStyleSheet(
            "color: rgba(100, 100, 100, 0.5); background-color: rgba(112, 70, 192, 1); border-radius: 0px"
        )
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self.toggled_on:
            self.setStyleSheet(
                "color: rgba(100, 100, 100, 0.5); background-color: rgba(57, 35, 98, 1); border-radius: 0px"
            )
        else:
            self.setStyleSheet(
                "color: rgba(100, 100, 100, 0.5); background-color: rgba(0, 0, 0, 0); border-radius: 0px"
            )
        self.unsetCursor()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.clicked.emit()
