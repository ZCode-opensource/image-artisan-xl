from importlib.resources import files

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QBrush, QColor, QPixmap, QCursor


class RemoveButton(QWidget):
    clicked = pyqtSignal()

    CLOSE_IMG = files("iartisanxl.theme.icons").joinpath("close.png")
    red_color = QColor("#b40808")
    white_color = QColor("#ffffff")
    hover_color = QColor("#c52222")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hovered = False
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.hovered:
            painter.setBrush(QBrush(self.hover_color))
        else:
            painter.setBrush(QBrush(self.red_color))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(1, 1, self.width() - 1, self.height() - 2, 5, 5)

        pixmap = QPixmap(str(self.CLOSE_IMG))
        pixmap = pixmap.scaled(
            20,
            20,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawPixmap(
            (self.width() - pixmap.width()) // 2,
            (self.height() - pixmap.height()) // 2,
            pixmap,
        )

    def enterEvent(self, _event):
        self.hovered = True
        self.update()

    def leaveEvent(self, _event):
        self.hovered = False
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
