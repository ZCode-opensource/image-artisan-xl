from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from superqt import QDoubleSlider

from iartisanxl.app.event_bus import EventBus
from iartisanxl.modules.common.lora.lora_added_item import LoraAddedItem
from iartisanxl.modules.common.lora.lora_dialog import LoraDialog
from iartisanxl.modules.common.panels.base_panel import BasePanel


class LoraPanel(BasePanel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lora_scale = 1.0
        self.init_ui()

        self.event_bus = EventBus()
        self.event_bus.subscribe("update_from_json", self.update_from_json)

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

    def update_ui(self):
        self.lora_scale = self.image_generation_data.lora_scale
        self.lora_scale_slider.setValue(self.lora_scale)
        self.lbl_lora_scale.setText(f"{self.lora_scale:.2f}")

        if len(self.lora_list.loras) > 0:
            for lora in self.lora_list.loras:
                lora_widget = LoraAddedItem(lora)
                lora_widget.remove_clicked.connect(self.on_remove_clicked)
                lora_widget.enabled.connect(self.on_enabled)
                lora_widget.sliders_locked.connect(self.on_locked)
                self.loras_layout.addWidget(lora_widget)

    def update_from_json(self, _data):
        while self.loras_layout.count():
            item = self.loras_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.update_ui()

    def on_remove_clicked(self, lora_widget: LoraAddedItem):
        self.lora_list.remove(lora_widget.lora)
        self.loras_layout.removeWidget(lora_widget)
        lora_widget.deleteLater()

    def on_enabled(self, lora_id, enabled):
        self.lora_list.update_lora_by_id(lora_id, {"enabled": enabled})

    def on_locked(self, lora_id, locked):
        self.lora_list.update_lora_by_id(lora_id, {"locked": locked})

    def clean_up(self):
        self.event_bus.unsubscribe("update_from_json", self.update_from_json)
