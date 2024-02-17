from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPainter
from PyQt6.QtWidgets import QFrame


class TransparentButton(QFrame):
    clicked = pyqtSignal(bool)

    def __init__(self, icon, button_width: int = 20, button_height: int = 20):
        super().__init__()

        self.button_width = button_width
        self.button_height = button_height

        self.setFixedSize(self.button_width, self.button_height)
        self.icon = icon
        self.setStyleSheet("background-color: rgba(0, 0, 0, 0); border-radius: 0px;")

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

    def enterEvent(self, event):
        self.setStyleSheet("color: rgba(100, 100, 100, 0.5); border-radius: 0px")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet("background-color: rgba(0, 0, 0, 0);border-radius: 0px;")
        self.unsetCursor()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.clicked.emit(True)
