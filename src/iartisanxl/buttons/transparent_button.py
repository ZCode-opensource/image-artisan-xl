from PyQt6.QtWidgets import QFrame
from PyQt6.QtGui import QIcon, QPainter
from PyQt6.QtCore import pyqtSignal, QRectF, Qt


class TransparentButton(QFrame):
    clicked = pyqtSignal(bool)

    def __init__(self, icon, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFixedSize(20, 20)
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
            QRectF(0, 0, 20, 20).toRect(),
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
