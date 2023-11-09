from importlib.resources import files
from PyQt6 import QtGui
from PyQt6.QtCore import Qt, QRectF

from iartisanxl.buttons.base_menu_button import BaseMenuButton


class ExpandButton(BaseMenuButton):
    CONTRACT_ICON = files("iartisanxl.theme.icons").joinpath("chevron_left.png")
    HAMBURGUER_ICON = files("iartisanxl.theme.icons").joinpath("hamburguer.png")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setMinimumWidth(40)
        self.extended = True

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.TextAntialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )

        if self.extended:
            icon = QtGui.QIcon(str(self.CONTRACT_ICON))

        else:
            icon = QtGui.QIcon(str(self.HAMBURGUER_ICON))

        icon.paint(
            painter,
            QRectF(self.width() - 40, 2, 40, 40).toRect(),
            Qt.AlignmentFlag.AlignCenter,
            state=QtGui.QIcon.State.Off,
        )

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.clicked.emit(True)
        self.extended = not self.extended
        self.update()
