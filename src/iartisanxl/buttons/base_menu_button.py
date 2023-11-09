from abc import abstractmethod, ABCMeta

from PyQt6.QtWidgets import QFrame
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QSizePolicy


class ABCQFrameMeta(ABCMeta, type(QFrame)):
    pass


class BaseMenuButton(QFrame, metaclass=ABCQFrameMeta):
    clicked = pyqtSignal(bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.extended = False

        self.setMinimumSize(40, 44)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )

    @abstractmethod
    def paintEvent(self, event):
        super().paintEvent(event)

    def enterEvent(self, event):
        self.setStyleSheet("background-color: #3a4046; border-color: #2f3338")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet("")
        self.unsetCursor()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.clicked.emit(True)
