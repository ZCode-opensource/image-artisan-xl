from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
)
from PyQt6.QtCore import Qt

from iartisanxl.modules.common.panels.base_panel import BasePanel
from iartisanxl.modules.common.dialogs.lora_dialog import LoraDialog
from iartisanxl.modules.common.lora_added_item import LoraAddedItem


class LoraPanel(BasePanel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.loras = []
        self.lora_scale = 1.0
        self.lora_dialog = None
        self.init_ui()
        self.update_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        add_lora_button = QPushButton("Add LoRA")
        add_lora_button.clicked.connect(self.open_lora_dialog)
        main_layout.addWidget(add_lora_button)

        lora_scale_layout = QHBoxLayout()
        lbl_lscale_text = QLabel("Lora scale")
        lora_scale_layout.addWidget(lbl_lscale_text, 1, Qt.AlignmentFlag.AlignLeft)
        self.lbl_lora_scale = QLabel(f"{self.lora_scale:.1f}")
        lora_scale_layout.addWidget(self.lbl_lora_scale, 1, Qt.AlignmentFlag.AlignRight)
        main_layout.addLayout(lora_scale_layout)

        self.lora_slider = QSlider()
        self.lora_slider.setRange(-100, 100)
        self.lora_slider.setSingleStep(1)
        self.lora_slider.setValue(int(self.lora_scale * 10))
        self.lora_slider.setOrientation(Qt.Orientation.Horizontal)
        self.lora_slider.valueChanged.connect(self.on_lora_scale_changed)
        main_layout.addWidget(self.lora_slider)

        added_loras_widget = QWidget()
        self.loras_layout = QVBoxLayout(added_loras_widget)
        main_layout.addWidget(added_loras_widget)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def on_lora_scale_changed(self):
        self.lora_scale = self.lora_slider.value() / 10.0
        self.lbl_lora_scale.setText(f"{self.lora_scale:.1f}")
        self.image_generation_data.lora_scale = self.lora_scale

    def open_lora_dialog(self):
        self.dialog_opened.emit(LoraDialog, "LoRAs")

    def on_remove_lora(self, lora_widget: LoraAddedItem):
        index = self.loras_layout.indexOf(lora_widget)
        if index != -1:
            self.loras_layout.takeAt(index)
            lora_widget.deleteLater()
            self.image_generation_data.remove_lora(self.loras[index].lora)
            del self.loras[index]

    def clear_loras(self):
        self.loras = []

        while self.loras_layout.count():
            item = self.loras_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def update_ui(self):
        self.lora_scale = self.image_generation_data.lora_scale
        self.lora_slider.setValue(int(self.lora_scale * 10))
        self.lbl_lora_scale.setText(f"{self.lora_scale:.1f}")
        loras = self.image_generation_data.loras
        self.clear_loras()

        if len(loras) > 0:
            for lora in loras:
                lora_widget = LoraAddedItem(lora)
                lora_widget.remove_clicked.connect(
                    lambda lw=lora_widget: self.on_remove_lora(lw)
                )
                self.loras_layout.addWidget(lora_widget)
                self.loras.append(lora_widget)
