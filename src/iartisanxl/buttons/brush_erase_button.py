from importlib.resources import files

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QWidget

from .toggle_button import ToggleButton


class BrushEraseButton(QWidget):
    BRUSH_IMG = files("iartisanxl.theme.icons").joinpath("brush.png")
    ERASER_IMG = files("iartisanxl.theme.icons").joinpath("eraser.png")

    brush_selected = pyqtSignal(bool)

    def __init__(self):
        super().__init__()

        self.erase_mode = False

        self.init_ui()
        self.on_brush_clicked()

    def init_ui(self):
        main_layout = QHBoxLayout()

        self.brush_button = ToggleButton(self.BRUSH_IMG, 25, 25)
        self.brush_button.clicked.connect(self.on_brush_clicked)
        main_layout.addWidget(self.brush_button)
        self.erase_button = ToggleButton(self.ERASER_IMG, 25, 25)
        self.erase_button.clicked.connect(self.on_erase_clicked)
        main_layout.addWidget(self.erase_button)

        self.setLayout(main_layout)

    def on_brush_clicked(self):
        self.brush_button.set_toggle(True)
        self.erase_button.set_toggle(False)
        self.erase_mode = False
        self.brush_selected.emit(self.erase_mode)

    def on_erase_clicked(self):
        self.brush_button.set_toggle(False)
        self.erase_button.set_toggle(True)
        self.erase_mode = True
        self.brush_selected.emit(self.erase_mode)
