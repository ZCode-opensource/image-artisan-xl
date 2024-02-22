from importlib.resources import files

from PyQt6 import QtGui
from PyQt6.QtCore import QRectF, Qt

from iartisanxl.buttons.base_menu_button import BaseMenuButton


class ExpandContractButton(BaseMenuButton):
    CONTRACT_ICON = files("iartisanxl.theme.icons").joinpath("chevron_right.png")
    EXPAND_ICON = files("iartisanxl.theme.icons").joinpath("chevron_left.png")

    def __init__(self, button_width: int = 40, button_height: int = 40, extended: bool = True, inverted: bool = False):
        super().__init__()

        self.button_width = button_width
        self.button_height = button_height
        self.inverted = inverted
        self.extended = extended
        self.setFixedSize(self.button_width, self.button_height + 4)

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.TextAntialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )

        if self.extended:
            if self.inverted:
                icon = QtGui.QIcon(str(self.CONTRACT_ICON))
            else:
                icon = QtGui.QIcon(str(self.EXPAND_ICON))
        else:
            if self.inverted:
                icon = QtGui.QIcon(str(self.EXPAND_ICON))
            else:
                icon = QtGui.QIcon(str(self.CONTRACT_ICON))
        icon.paint(
            painter,
            QRectF(0, 2, self.button_width, self.button_height).toRect(),
            Qt.AlignmentFlag.AlignCenter,
            state=QtGui.QIcon.State.Off,
        )

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.clicked.emit(True)
        self.extended = not self.extended
        self.update()
