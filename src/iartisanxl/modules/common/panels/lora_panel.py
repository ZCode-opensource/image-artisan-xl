from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt
from superqt import QDoubleSlider

from iartisanxl.app.event_bus import EventBus
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
        self.dialog = None
        self.init_ui()

        self.event_bus = EventBus()
        self.event_bus.subscribe("lora", self.on_lora)
        self.event_bus.subscribe("update_from_json", self.update_ui)

        self.update_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        add_lora_button = QPushButton("Add LoRA")
        add_lora_button.clicked.connect(self.open_lora_dialog)
        main_layout.addWidget(add_lora_button)

        lora_scale_layout = QHBoxLayout()
        lbl_lscale_text = QLabel("Lora scale")
        lora_scale_layout.addWidget(lbl_lscale_text, 1, Qt.AlignmentFlag.AlignLeft)
        self.lbl_lora_scale = QLabel()
        lora_scale_layout.addWidget(self.lbl_lora_scale, 1, Qt.AlignmentFlag.AlignRight)
        main_layout.addLayout(lora_scale_layout)

        self.lora_scale_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        self.lora_scale_slider.setRange(0.0, 1.0)
        self.lora_scale_slider.setValue(self.lora_scale)
        self.lora_scale_slider.valueChanged.connect(self.on_lora_scale_changed)
        main_layout.addWidget(self.lora_scale_slider)

        added_loras_widget = QWidget()
        self.loras_layout = QVBoxLayout(added_loras_widget)
        main_layout.addWidget(added_loras_widget)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def on_lora_scale_changed(self, value):
        self.lora_scale = value
        self.lbl_lora_scale.setText(f"{self.lora_scale:.2f}")
        self.image_generation_data.lora_scale = self.lora_scale

    def open_lora_dialog(self):
        self.parent().open_dialog(
            "lora",
            LoraDialog,
            self.directories,
            self.preferences,
            "LoRAs",
            self.show_error,
            self.image_generation_data,
            self.image_viewer,
            self.prompt_window,
        )

        self.parent().lora_dialog.loading_loras = True
        self.parent().lora_dialog.lora_items_view.load_items()

    def clear_loras(self):
        self.loras = []

        while self.loras_layout.count():
            item = self.loras_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def update_ui(self, _data=None):
        self.lora_scale = self.image_generation_data.lora_scale
        self.lora_scale_slider.setValue(self.lora_scale)
        self.lbl_lora_scale.setText(f"{self.lora_scale:.2f}")
        loras = self.lora_list.loras
        self.clear_loras()

        if len(loras) > 0:
            for lora in loras:
                lora_widget = LoraAddedItem(lora)
                lora_widget.remove_clicked.connect(lambda lw=lora_widget: self.on_remove_lora(lw))
                lora_widget.enabled.connect(lambda lw=lora_widget: self.on_lora_enabled(lw))
                self.loras_layout.addWidget(lora_widget)
                self.loras.append(lora_widget)

    def on_remove_lora(self, lora_widget: LoraAddedItem):
        index = self.loras_layout.indexOf(lora_widget)
        if index != -1:
            self.loras_layout.takeAt(index)
            lora_widget.deleteLater()
            self.lora_list.remove(self.loras[index].lora)
            del self.loras[index]

    def on_lora_enabled(self, lora_widget: LoraAddedItem):
        self.lora_list.update_lora(
            lora_widget.lora.filename,
            {"enabled": lora_widget.enabled_checkbox.isChecked()},
        )

    def on_lora(self, data):
        if data["action"] == "add":
            lora_widget = LoraAddedItem(data["lora"])
            lora_widget.remove_clicked.connect(lambda lw=lora_widget: self.on_remove_lora(lw))
            lora_widget.enabled.connect(lambda lw=lora_widget: self.on_lora_enabled(lw))
            self.loras_layout.addWidget(lora_widget)
            self.loras.append(lora_widget)

    def clean_up(self):
        self.event_bus.unsubscribe("lora", self.on_lora)
        self.event_bus.unsubscribe("update_from_json", self.update_ui)
