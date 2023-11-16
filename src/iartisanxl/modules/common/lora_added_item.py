from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal

from iartisanxl.generation.lora_data_object import LoraDataObject
from iartisanxl.buttons.remove_button import RemoveButton


class LoraAddedItem(QWidget):
    remove_clicked = pyqtSignal()
    weight_changed = pyqtSignal()
    enabled = pyqtSignal()

    def __init__(self, lora: LoraDataObject, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora = lora
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

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
        self.weight_slider = QSlider()
        self.weight_slider.setRange(-100, 100)
        self.weight_slider.setSingleStep(1)
        self.weight_slider.setValue(int(self.lora.weight * 10))
        self.weight_slider.setOrientation(Qt.Orientation.Horizontal)
        bottom_layout.addWidget(self.weight_slider)

        self.lbl_added_lora_weight = QLabel(f"{self.lora.weight:.1f}")
        bottom_layout.addWidget(self.lbl_added_lora_weight)

        self.weight_slider.valueChanged.connect(self.on_slider_value_changed)

        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

    def on_slider_value_changed(self, value):
        float_value = value / 10.0
        self.lora.weight = float_value
        self.lbl_added_lora_weight.setText(f"{float_value:.1f}")

    def get_slider_value(self):
        return self.weight_slider.value() / 10.0

    def on_check_enabled(self):
        self.enabled.emit()
