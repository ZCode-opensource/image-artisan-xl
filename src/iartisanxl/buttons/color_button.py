from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget, QPushButton
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QCursor
from vcolorpicker import getColor


class ColorButton(QWidget):
    color_changed = pyqtSignal(tuple)

    def __init__(self, text: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.text = text
        self.color = (0, 0, 0)

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()

        select_color_button = QLabel(self.text)
        main_layout.addWidget(select_color_button)

        self.color_frame = QPushButton()
        self.color_frame.setFixedSize(QSize(50, 22))
        self.color_frame.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.color_frame.clicked.connect(self.on_select_color)
        main_layout.addWidget(self.color_frame)

        self.color_frame.setStyleSheet(f"background-color: rgb{self.color};")

        self.setLayout(main_layout)

    def on_select_color(self):
        self.set_color(getColor())

    def set_color(self, color):
        self.color = color
        self.color_frame.setStyleSheet(f"background-color: rgb{self.color};")
        self.color_changed.emit(self.color)
