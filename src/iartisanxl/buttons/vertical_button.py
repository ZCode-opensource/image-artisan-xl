from PyQt6.QtWidgets import QWidget, QVBoxLayout, QStyleOption
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath
from PyQt6.QtCore import Qt, QRectF, pyqtSignal

from .vertical_label import VerticalLabel


class VerticalButton(QWidget):
    clicked = pyqtSignal()

    def __init__(self, text=""):
        super().__init__()
        self.text = text
        self.hover = False

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        self.label = VerticalLabel(self.text)
        main_layout.addWidget(self.label)

        self.setLayout(main_layout)

    def enterEvent(self, event):
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.label.hover_in()
        self.hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.unsetCursor()
        self.label.hover_out()
        self.hover = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def paintEvent(self, _event):
        opt = QStyleOption()
        opt.initFrom(self)
        painter = QPainter(self)

        painter.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.TextAntialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )

        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), 4, 4)

        if self.hover:
            painter.fillPath(path, QColor("#3a4046"))

        pen = QPen(QColor("#ada7c9"), 1)
        painter.setPen(pen)
        painter.drawPath(path)
