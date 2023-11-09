from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
)
from PyQt6.QtCore import Qt, pyqtSignal

from iartisanxl.generation.lora_data_object import LoraDataObject


class LoraAddedItem(QWidget):
    remove_clicked = pyqtSignal()
    weight_changed = pyqtSignal()

    def __init__(self, lora: LoraDataObject, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora = lora
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        upper_layout = QHBoxLayout()

        lora_name = self.lora.name
        if len(self.lora.version) > 0:
            lora_name = f"{self.lora.name}_{self.lora.version}"

        lora_label = QLabel(lora_name)
        upper_layout.addWidget(lora_label)

        remove_button = QPushButton("X")
        remove_button.setMinimumSize(20, 20)
        remove_button.setMaximumSize(20, 20)
        remove_button.setStyleSheet("color: white; background-color: red;")
        remove_button.clicked.connect(self.remove_clicked.emit)
        upper_layout.addWidget(remove_button)

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
