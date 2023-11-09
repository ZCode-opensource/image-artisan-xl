from importlib.resources import files
from PyQt6 import QtGui
from PyQt6.QtCore import Qt, QRectF

from iartisanxl.buttons.base_menu_button import BaseMenuButton


class ExpandRightButton(BaseMenuButton):
    CONTRACT_ICON = files("iartisanxl.theme.icons").joinpath("chevron_right.png")
    EXPAND_ICON = files("iartisanxl.theme.icons").joinpath("chevron_left.png")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.extended = True
        self.setFixedSize(40, 44)

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
            icon = QtGui.QIcon(str(self.EXPAND_ICON))
        icon.paint(
            painter,
            QRectF(0, 2, 40, 40).toRect(),
            Qt.AlignmentFlag.AlignCenter,
            state=QtGui.QIcon.State.Off,
        )

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.clicked.emit(True)
        self.extended = not self.extended
        self.update()
