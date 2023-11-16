from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox
from PyQt6.QtCore import pyqtSignal

from iartisanxl.generation.controlnet_data_object import ControlNetDataObject
from iartisanxl.buttons.remove_button import RemoveButton


class ControlNetAddedItem(QWidget):
    remove_clicked = pyqtSignal()
    enabled = pyqtSignal()

    def __init__(self, controlnet: ControlNetDataObject, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.controlnet = controlnet
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        upper_layout = QHBoxLayout()

        self.enabled_checkbox = QCheckBox(self.controlnet.name)
        self.enabled_checkbox.setChecked(self.controlnet.enabled)
        self.enabled_checkbox.stateChanged.connect(self.on_check_enabled)
        upper_layout.addWidget(self.enabled_checkbox)

        remove_button = RemoveButton()
        remove_button.setFixedSize(20, 20)
        remove_button.clicked.connect(self.remove_clicked.emit)
        upper_layout.addWidget(remove_button)

        upper_layout.setStretch(0, 1)
        upper_layout.setStretch(1, 0)

        main_layout.addLayout(upper_layout)

        self.setLayout(main_layout)

    def on_check_enabled(self):
        self.enabled.emit()
