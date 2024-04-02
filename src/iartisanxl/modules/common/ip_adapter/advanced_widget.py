from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from superqt import QLabeledDoubleSlider

from iartisanxl.modules.common.ip_adapter.ip_adapter_data_object import IPAdapterDataObject


class AdvancedWidget(QWidget):
    advanced_canceled = pyqtSignal()
    granular_enabled = pyqtSignal(bool)

    def __init__(self, ip_adapter: IPAdapterDataObject):
        super().__init__()

        self.ip_adapter = ip_adapter
        self.attention_values = {
            "down_1": [1.0, 1.0],
            "down_2": [1.0, 1.0],
            "mid": [1.0],
            "up_0": [1.0, 1.0, 1.0],
            "up_1": [1.0, 1.0, 1.0],
        }

        self.sliders = {}
        self.frames = []

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        granular_scales_checkbox = QCheckBox("Enable granular scales")
        granular_scales_checkbox.stateChanged.connect(self.on_granular)
        main_layout.addWidget(granular_scales_checkbox)

        sections_layout = QHBoxLayout()

        for section, values in self.attention_values.items():
            frame = QFrame()
            frame.setDisabled(True)
            frame.setObjectName("block_frame")

            blocks_layout = QVBoxLayout()
            section_label = QLabel(f"{section.capitalize()} Blocks")
            blocks_layout.addWidget(section_label)

            # Loop and create all the sliders for the section
            for i, value in enumerate(values):
                attention_layout = QHBoxLayout()
                attention_label = QLabel(
                    f"Attention {i+1}"
                )  # this is the number of the count in the total attention vars
                attention_layout.addWidget(attention_label)
                attention_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
                attention_slider.setRange(0.0, 1.0)
                attention_slider.setValue(value)
                attention_slider.valueChanged.connect(lambda val, sec=section, idx=i: self.update_scale(val, sec, idx))
                attention_layout.addWidget(attention_slider)
                blocks_layout.addLayout(attention_layout)

                self.sliders.setdefault(section, []).append(attention_slider)

            frame.setLayout(blocks_layout)
            sections_layout.addWidget(frame)
            self.frames.append(frame)

        main_layout.addLayout(sections_layout)
        main_layout.addStretch()

        button_layout = QHBoxLayout()
        save_button = QPushButton("Set scales")
        save_button.clicked.connect(self.on_save)
        button_layout.addWidget(save_button)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.on_cancel)
        button_layout.addWidget(cancel_button)

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def update_scale(self, value, section, index):
        self.attention_values[section][index] = value

    def on_cancel(self):
        self.advanced_canceled.emit()

    def on_save(self):
        scales = {}
        for section, values in self.attention_values.items():
            if section.startswith("down_"):
                block = "down_blocks"
                block_num = section.split("_")[1]
            elif section == "mid":
                block = "mid_block"
                block_num = ""
            elif section.startswith("up_"):
                block = "up_blocks"
                block_num = section.split("_")[1]

            for i, value in enumerate(values):
                key = f"{block}.{block_num}.attentions.{i}"
                scales[key] = value

        self.ip_adapter.granular_scale = scales
        self.advanced_canceled.emit()

    def on_granular(self, state):
        is_enabled = state != Qt.CheckState.Unchecked.value

        for frame in self.frames:
            frame.setEnabled(is_enabled)

        self.granular_enabled.emit(is_enabled)
