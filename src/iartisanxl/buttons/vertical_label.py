from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QPainter, QPalette
from PyQt6.QtCore import QSize


class VerticalLabel(QLabel):
    def __init__(self, text=""):
        super().__init__(text)

        self.setStyleSheet("color: #ada7c9;")

    def hover_in(self):
        self.setStyleSheet("color: #ffffff; background-color: #3a4046;")

    def hover_out(self):
        self.setStyleSheet("color: #ada7c9;")

    def paintEvent(self, _event):
        painter = QPainter(self)

        painter.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.TextAntialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )

        painter.setPen(self.palette().color(QPalette.ColorRole.Text))
        painter.translate(-15, self.height() - 1)
        painter.rotate(-90)
        painter.drawText(0, 30, self.text())
        painter.end()

    def minimumSizeHint(self):
        size = super().minimumSizeHint()
        return QSize(size.height(), size.width())

    def sizeHint(self):
        size = super().sizeHint()
        return QSize(size.height(), size.width())
