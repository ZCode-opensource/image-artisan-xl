from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout
from superqt import QLabeledDoubleSlider

from iartisanxl.buttons.remove_button import RemoveButton
from iartisanxl.modules.common.lora.lora_data_object import LoraDataObject


class LoraAddedItem(QFrame):
    remove_clicked = pyqtSignal(object)
    weight_changed = pyqtSignal()
    enabled = pyqtSignal(int, bool)

    def __init__(self, lora: LoraDataObject, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora = lora
        self.linked = True

        self.init_ui()

        self.unet_weight_slider.valueChanged.connect(lambda weight: self.on_slider_value_changed(0, weight))
        self.text_encoder_one_weight_slider.valueChanged.connect(
            lambda weight: self.on_slider_value_changed(1, weight)
        )
        self.text_encoder_two_weight_slider.valueChanged.connect(
            lambda weight: self.on_slider_value_changed(2, weight)
        )

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
        remove_button.clicked.connect(lambda: self.remove_clicked.emit(self))
        upper_layout.addWidget(remove_button)

        upper_layout.setStretch(0, 1)
        upper_layout.setStretch(1, 0)

        main_layout.addLayout(upper_layout)

        sliders_layout = QGridLayout()

        unet_label = QLabel("Unet: ")
        sliders_layout.addWidget(unet_label, 0, 0)
        self.unet_weight_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.unet_weight_slider.setRange(-8.0, 8.0)
        self.unet_weight_slider.setValue(self.lora.unet_weight)
        sliders_layout.addWidget(self.unet_weight_slider, 0, 1)

        text_encoder_one_label = QLabel("Text 1: ")
        sliders_layout.addWidget(text_encoder_one_label, 1, 0)
        self.text_encoder_one_weight_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.text_encoder_one_weight_slider.setRange(-8.0, 8.0)
        self.text_encoder_one_weight_slider.setValue(self.lora.text_encoder_one_weight)
        sliders_layout.addWidget(self.text_encoder_one_weight_slider, 1, 1)

        text_encoder_two_label = QLabel("Text 2: ")
        sliders_layout.addWidget(text_encoder_two_label, 2, 0)
        self.text_encoder_two_weight_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.text_encoder_two_weight_slider.setRange(-8.0, 8.0)
        self.text_encoder_two_weight_slider.setValue(self.lora.text_encoder_two_weight)
        sliders_layout.addWidget(self.text_encoder_two_weight_slider, 2, 1)

        main_layout.addLayout(sliders_layout)

        bottom_layout = QHBoxLayout()
        advanced_button = QPushButton("Advanced")
        bottom_layout.addWidget(advanced_button, alignment=Qt.AlignmentFlag.AlignCenter)
        self.linked_checkbox = QCheckBox("Linked")
        self.linked_checkbox.setChecked(self.linked)
        self.linked_checkbox.stateChanged.connect(self.on_linked_changed)
        bottom_layout.addWidget(self.linked_checkbox, alignment=Qt.AlignmentFlag.AlignCenter)

        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

    def on_slider_value_changed(self, weight_type: int, value: float):
        if self.linked:
            self.lora.unet_weight = value
            self.lora.text_encoder_one_weight = value
            self.lora.text_encoder_two_weight = value

            self.unet_weight_slider.setValue(value)
            self.text_encoder_one_weight_slider.setValue(value)
            self.text_encoder_two_weight_slider.setValue(value)
        else:
            if weight_type == 0:
                self.lora.unet_weight = value
            elif weight_type == 1:
                self.lora.text_encoder_one_weight = value
            elif weight_type == 2:
                self.lora.text_encoder_two_weight = value

    def get_slider_value(self):
        return self.unet_weight_slider.value()

    def on_check_enabled(self):
        self.enabled.emit(self.lora.lora_id, self.enabled_checkbox.isChecked())

    def on_linked_changed(self):
        self.linked = self.linked_checkbox.isChecked()
