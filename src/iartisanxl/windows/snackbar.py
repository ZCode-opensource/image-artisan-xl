from importlib.resources import files
from PyQt6 import QtWidgets, QtGui
from PyQt6.QtCore import pyqtSignal

from iartisanxl.buttons.transparent_button import TransparentButton


class SnackBar(QtWidgets.QFrame):
    CLOSE_IMG = files("iartisanxl.theme.icons").joinpath("close.png")
    closed = pyqtSignal()
    __slots__ = ["_message"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setVisible(False)
        self.setStyleSheet(
            "background-color: rgba(150, 30, 30, 0.5); border-radius: 10px;"
        )

        self.setMinimumHeight(50)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)

        text_layout = QtWidgets.QHBoxLayout()
        text_layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        self.label = QtWidgets.QLabel()
        self.label.setStyleSheet("padding: 10px; background-color: rgba(0, 0, 0, 0);")
        text_layout.addWidget(self.label)
        main_layout.addLayout(text_layout)

        close_button = TransparentButton(self.CLOSE_IMG)
        close_button.clicked.connect(self.closed.emit)
        text_layout.addWidget(close_button)

    @property
    def message(self):
        return self._message

    @message.setter
    def message(self, value):
        self._message = value
        self.label.setText(value)

        # Calculate the width of the text
        font_metrics = QtGui.QFontMetrics(self.label.font())
        text_width = font_metrics.horizontalAdvance(value)

        new_width = min(text_width, 350)
        if text_width > 350:
            self.label.setWordWrap(True)

        # Set the width of the SnackBar
        self.setMinimumWidth(new_width)  # Add some padding
        self.adjustSize()
