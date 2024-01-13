import torch

from PyQt6.QtWidgets import QVBoxLayout, QPushButton, QWidget
from PyQt6.QtGui import QPixmap

from iartisanxl.app.event_bus import EventBus
from iartisanxl.modules.common.panels.base_panel import BasePanel
from iartisanxl.modules.common.controlnet.controlnet_dialog import ControlNetDialog
from iartisanxl.modules.common.controlnet.controlnet_added_item import ControlNetAddedItem
from iartisanxl.modules.common.controlnet.controlnet_data_object import ControlNetDataObject


class ControlNetPanel(BasePanel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_bus = EventBus()
        self.event_bus.subscribe("controlnet", self.on_controlnet)
        self.controlnets = []

        self.init_ui()
        self.update_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        add_controlnet_button = QPushButton("Add ControlNet")
        add_controlnet_button.clicked.connect(self.open_controlnet_dialog)
        main_layout.addWidget(add_controlnet_button)

        added_controlnets_widget = QWidget()
        self.controlnets_layout = QVBoxLayout(added_controlnets_widget)
        main_layout.addWidget(added_controlnets_widget)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def update_ui(self):
        if len(self.controlnet_list.adapters) > 0:
            for controlnet in self.controlnet_list.adapters:
                controlnet_widget = ControlNetAddedItem(controlnet)
                controlnet_widget.remove_clicked.connect(self.on_remove_clicked)
                controlnet_widget.edit_clicked.connect(self.on_edit_clicked)
                controlnet_widget.enabled.connect(self.on_controlnet_enabled)
                self.controlnets_layout.addWidget(controlnet_widget)

    def open_controlnet_dialog(self):
        self.parent().open_dialog(
            "controlnet",
            ControlNetDialog,
            self.directories,
            self.preferences,
            "ControlNet",
            self.show_error,
            self.image_generation_data,
            self.image_viewer,
            self.prompt_window,
        )

    def on_dialog_closed(self):
        self.parent().controlnet_dialog.depth_estimator = None
        self.parent().controlnet_dialog = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def on_controlnet(self, data):
        if data["action"] == "add":
            adapter_id = self.controlnet_list.add(data["controlnet"])
            data["controlnet"].adapter_id = adapter_id
            controlnet_widget = ControlNetAddedItem(data["controlnet"])
            controlnet_widget.remove_clicked.connect(self.on_remove_clicked)
            controlnet_widget.edit_clicked.connect(self.on_edit_clicked)
            controlnet_widget.enabled.connect(self.on_controlnet_enabled)
            self.controlnets_layout.addWidget(controlnet_widget)
        elif data["action"] == "update":
            controlnet = data["controlnet"]
            self.controlnet_list.update_with_adapter_data_object(controlnet)
            for i in range(self.controlnets_layout.count()):
                widget = self.controlnets_layout.itemAt(i).widget()
                if widget.controlnet.adapter_id == controlnet.adapter_id:
                    widget.enabled_checkbox.setText(controlnet.adapter_type)
                    source_thumb_pixmap = QPixmap(controlnet.source_image_thumb)
                    widget.source_thumb.setPixmap(source_thumb_pixmap)
                    annotator_thumb_pixmap = QPixmap(controlnet.annotator_image_thumb)
                    widget.annotator_thumb.setPixmap(annotator_thumb_pixmap)
                    widget.controlnet = data["controlnet"]
                    break

    def on_remove_clicked(self, controlnet_widget: ControlNetAddedItem):
        self.controlnet_list.remove(controlnet_widget.controlnet)
        self.controlnets_layout.removeWidget(controlnet_widget)
        controlnet_widget.deleteLater()

        if self.parent().controlnet_dialog is not None:
            self.parent().controlnet_dialog.controlnet = None
            self.parent().controlnet_dialog.reset_ui()

    def on_edit_clicked(self, controlnet: ControlNetDataObject):
        if self.parent().controlnet_dialog is None:
            self.open_controlnet_dialog()

        self.parent().controlnet_dialog.controlnet = controlnet
        self.parent().controlnet_dialog.update_ui()
        self.parent().controlnet_dialog.raise_()

    def clean_up(self):
        self.event_bus.unsubscribe("controlnet", self.on_controlnet)
        super().clean_up()

    def on_controlnet_enabled(self, controlet_id, enabled):
        self.controlnet_list.update_adapter(controlet_id, {"enabled": enabled})
