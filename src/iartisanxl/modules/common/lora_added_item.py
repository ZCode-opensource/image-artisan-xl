from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QCheckBox
from PyQt6.QtCore import Qt, pyqtSignal
from superqt import QLabeledDoubleSlider

from iartisanxl.generation.lora_data_object import LoraDataObject
from iartisanxl.buttons.remove_button import RemoveButton


class LoraAddedItem(QFrame):
    remove_clicked = pyqtSignal()
    weight_changed = pyqtSignal()
    enabled = pyqtSignal()

    def __init__(self, lora: LoraDataObject, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora = lora
        self.init_ui()
        self.weight_slider.valueChanged.connect(self.on_slider_value_changed)

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(3, 3, 3, 3)
        main_layout.setSpacing(1)
        upper_layout = QHBoxLayout()

        lora_name = self.lora.name
        if len(self.lora.version) > 0:
            lora_name = f"{self.lora.name} v{self.lora.version}"

        self.enabled_checkbox = QCheckBox(lora_name)
        self.enabled_checkbox.setChecked(self.lora.enabled)
        self.enabled_checkbox.stateChanged.connect(self.on_check_enabled)
        upper_layout.addWidget(self.enabled_checkbox)

        remove_button = RemoveButton()
        remove_button.setFixedSize(20, 20)
        remove_button.clicked.connect(self.remove_clicked.emit)
        upper_layout.addWidget(remove_button)

        upper_layout.setStretch(0, 1)
        upper_layout.setStretch(1, 0)

        main_layout.addLayout(upper_layout)

        bottom_layout = QHBoxLayout()
        self.weight_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.weight_slider.setRange(-8.0, 8.0)
        self.weight_slider.setValue(self.lora.weight)
        main_layout.addWidget(self.weight_slider)
        bottom_layout.addWidget(self.weight_slider)

        main_layout.addLayout(bottom_layout)
        self.setLayout(main_layout)

    def on_slider_value_changed(self, value):
        self.lora.weight = value

    def get_slider_value(self):
        return self.weight_slider.value()

    def on_check_enabled(self):
        self.enabled.emit()
