from PyQt6 import QtGui
from PyQt6.QtCore import Qt, QRectF, QRect

from iartisanxl.buttons.base_menu_button import BaseMenuButton


class MenuButton(BaseMenuButton):
    __slots__ = ["_icon", "_label", "_module_name"]

    def __init__(self, *args, icon: str, label: str, **kwargs):
        super().__init__(*args, **kwargs)

        self._icon = QtGui.QIcon(str(icon))
        self._label = label

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.TextAntialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )

        icon = QtGui.QIcon(self._icon)
        icon.paint(
            painter,
            QRectF(0, 2, 40, 40).toRect(),
            Qt.AlignmentFlag.AlignCenter,
            state=QtGui.QIcon.State.Off,
        )

        painter.setFont(self.font())
        painter.setPen(QtGui.QColor(255, 255, 255))
        painter.drawText(
            QRect(
                45,
                0,
                self.width(),
                self.height(),
            ),
            Qt.AlignmentFlag.AlignVCenter,
            self._label,
        )
