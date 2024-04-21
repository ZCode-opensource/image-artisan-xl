from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizeGrip,
    QSpacerItem,
    QVBoxLayout,
)
from superqt import QLabeledDoubleSlider

from iartisanxl.app.title_bar import TitleBar
from iartisanxl.modules.common.lora.lora_data_object import LoraDataObject


class CustomSizeGrip(QSizeGrip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFixedSize(5, 5)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QColor(0, 0, 0, 0))
        painter.setPen(QPen(QColor(0, 0, 0, 0)))
        painter.drawRect(event.rect())


class LoraAdvancedDialog(QDialog):
    border_color = QColor("#ff6b6b6b")
    closed = pyqtSignal()

    def __init__(self, lora: LoraDataObject):
        super().__init__()
        self.setWindowTitle("LoRA Advanced Dialog")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setMinimumSize(1000, 250)

        self.lora = lora

        self.sliders = {}
        self.frames = []

        self.init_ui()

        self.unet_weight_slider.valueChanged.connect(lambda weight: self.on_slider_value_changed(0, weight))
        self.text_encoder_one_weight_slider.valueChanged.connect(
            lambda weight: self.on_slider_value_changed(1, weight)
        )
        self.text_encoder_two_weight_slider.valueChanged.connect(
            lambda weight: self.on_slider_value_changed(2, weight)
        )

    def init_ui(self):
        self.dialog_layout = QVBoxLayout()
        self.dialog_layout.setContentsMargins(0, 0, 0, 0)
        self.dialog_layout.setSpacing(0)

        title_bar = TitleBar(title=f"LoRA Advanced Dialog - {self.lora.name}", is_dialog=True)
        self.dialog_layout.addWidget(title_bar)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        self.dialog_layout.addLayout(main_layout)

        sliders_layout = QGridLayout()

        self.unet_label = QLabel("Unet: ")
        self.unet_label.setDisabled(self.lora.granular_unet_weights_enabled)
        sliders_layout.addWidget(self.unet_label, 0, 0)
        self.unet_weight_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.unet_weight_slider.setRange(0.0, 1.0)
        self.unet_weight_slider.setValue(self.lora.unet_weight)
        self.unet_weight_slider.setDisabled(self.lora.granular_unet_weights_enabled)
        sliders_layout.addWidget(self.unet_weight_slider, 0, 1)

        text_encoder_one_label = QLabel("Text 1: ")
        sliders_layout.addWidget(text_encoder_one_label, 1, 0)
        self.text_encoder_one_weight_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.text_encoder_one_weight_slider.setRange(0.0, 1.0)
        self.text_encoder_one_weight_slider.setValue(self.lora.text_encoder_one_weight)
        sliders_layout.addWidget(self.text_encoder_one_weight_slider, 1, 1)

        text_encoder_two_label = QLabel("Text 2: ")
        sliders_layout.addWidget(text_encoder_two_label, 2, 0)
        self.text_encoder_two_weight_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.text_encoder_two_weight_slider.setRange(0.0, 1.0)
        self.text_encoder_two_weight_slider.setValue(self.lora.text_encoder_two_weight)
        sliders_layout.addWidget(self.text_encoder_two_weight_slider, 2, 1)

        main_layout.addLayout(sliders_layout)

        self.locked_checkbox = QCheckBox("Locked")
        self.locked_checkbox.setChecked(self.lora.locked)
        self.locked_checkbox.stateChanged.connect(self.on_lock_changed)
        main_layout.addWidget(self.locked_checkbox, alignment=Qt.AlignmentFlag.AlignCenter)

        granular_scales_checkbox = QCheckBox("Enable granular scales")
        granular_scales_checkbox.setChecked(self.lora.granular_unet_weights_enabled)
        granular_scales_checkbox.stateChanged.connect(self.on_granular)
        main_layout.addWidget(granular_scales_checkbox)

        sections_layout = QHBoxLayout()

        # Refactored loop to create sliders based on the new block_values structure
        for section, blocks in self.lora.granular_unet_weights.items():
            frame = QFrame()
            frame.setEnabled(self.lora.granular_unet_weights_enabled)
            frame.setObjectName("block_frame")

            blocks_layout = QVBoxLayout()
            section_label = QLabel(f"{section.capitalize()} Blocks")
            blocks_layout.addWidget(section_label)

            if isinstance(blocks, dict):
                for block, values in blocks.items():
                    for i, value in enumerate(values):
                        attention_layout = QHBoxLayout()
                        attention_label = QLabel(f"{block} Attention {i+1}")
                        attention_layout.addWidget(attention_label)
                        attention_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
                        attention_slider.setRange(0.0, 1.0)
                        attention_slider.setValue(value)
                        attention_slider.valueChanged.connect(
                            lambda val, sec=section, blk=block, idx=i: self.update_scale(val, sec, blk, idx)
                        )
                        attention_layout.addWidget(attention_slider)
                        blocks_layout.addLayout(attention_layout)

                        self.sliders.setdefault(section, {}).setdefault(block, []).append(attention_slider)
                    blocks_layout.addSpacerItem(QSpacerItem(0, 15))
                blocks_layout.addStretch()
            else:
                # Handle the 'mid' section which is not a dictionary
                attention_layout = QHBoxLayout()
                attention_label = QLabel(f"{section.capitalize()} Attention")
                attention_layout.addWidget(attention_label)
                attention_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
                attention_slider.setRange(0.0, 1.0)
                attention_slider.setValue(blocks)
                attention_slider.valueChanged.connect(lambda val, sec=section: self.update_scale(val, sec, None, None))
                attention_layout.addWidget(attention_slider)
                blocks_layout.addLayout(attention_layout)
                blocks_layout.addStretch()

                self.sliders[section] = attention_slider

            frame.setLayout(blocks_layout)
            sections_layout.addWidget(frame)
            self.frames.append(frame)
        main_layout.addLayout(sections_layout)
        main_layout.addStretch()

        size_grip = CustomSizeGrip(self)
        self.dialog_layout.addWidget(
            size_grip,
            alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight,
        )

        self.setLayout(self.dialog_layout)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(self.border_color)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(0, 0, 0, self.height())
        painter.drawLine(self.width(), 0, self.width(), self.height())
        painter.drawLine(0, self.height(), self.width(), self.height())

    def on_slider_value_changed(self, weight_type: int, value: float):
        if weight_type == 0:
            diff = value - self.lora.unet_weight
            self.lora.unet_weight = value

            if self.lora.locked:
                self.lora.text_encoder_one_weight += diff
                self.lora.text_encoder_two_weight += diff
        elif weight_type == 1:
            diff = value - self.lora.text_encoder_one_weight
            self.lora.text_encoder_one_weight = value

            if self.lora.locked:
                if not self.lora.granular_unet_weights_enabled:
                    self.lora.unet_weight += diff
                self.lora.text_encoder_two_weight += diff
        elif weight_type == 2:
            diff = value - self.lora.text_encoder_two_weight
            self.lora.text_encoder_two_weight = value

            if self.lora.locked:
                if not self.lora.granular_unet_weights_enabled:
                    self.lora.unet_weight += diff
                self.lora.text_encoder_one_weight += diff

        self.unet_weight_slider.setValue(self.lora.unet_weight)
        self.text_encoder_one_weight_slider.setValue(self.lora.text_encoder_one_weight)
        self.text_encoder_two_weight_slider.setValue(self.lora.text_encoder_two_weight)

    def on_lock_changed(self):
        self.lora.locked = self.locked_checkbox.isChecked()

    def on_granular(self, state):
        self.lora.granular_unet_weights_enabled = state != Qt.CheckState.Unchecked.value

        for frame in self.frames:
            frame.setEnabled(self.lora.granular_unet_weights_enabled)

        self.unet_label.setDisabled(self.lora.granular_unet_weights_enabled)
        self.unet_weight_slider.setDisabled(self.lora.granular_unet_weights_enabled)

    def update_scale(self, value, section, block, index):
        if isinstance(self.lora.granular_unet_weights[section], dict):
            self.lora.granular_unet_weights[section][block][index] = value
        else:
            self.lora.granular_unet_weights[section] = value

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)
