import os
import shutil

from typing import cast, Optional
from importlib.resources import files
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
    QFrame,
)
from PyQt6.QtCore import QSettings, pyqtSignal, Qt, QSize
from PyQt6.QtGui import QPixmap

from iartisanxl.app.event_bus import EventBus
from iartisanxl.generation.lora_data_object import LoraDataObject
from iartisanxl.modules.common.dialogs.base_dialog import BaseDialog
from iartisanxl.modules.common.dialogs.lora_info_widget import LoraInfoWidget
from iartisanxl.modules.common.dialogs.lora_edit_widget import LoraEditWidget
from iartisanxl.modules.common.dialogs.model_items_view import ModelItemsView
from iartisanxl.modules.common.model_item import ModelItem


class LoraDialog(BaseDialog):
    LORA_IMG = files("iartisanxl.theme.images").joinpath("lora.webp")
    lora_selected = pyqtSignal(LoraDataObject)

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.setWindowTitle("LoRAs")
        self.setMinimumSize(1160, 800)

        self.loading_loras = False
        self.selected_lora = None
        self.event_bus = EventBus()

        self.settings = QSettings("ZCode", "ImageArtisanXL")
        self.settings.beginGroup("lora_dialog")
        self.load_settings()

        loras_directory = {
            "path": self.directories.models_loras,
            "type": "safetensors",
        }
        self.loras_directories = (loras_directory,)

        self.init_ui()

    def init_ui(self):
        content_layout = QHBoxLayout()

        self.lora_items_view = ModelItemsView(self.loras_directories, self.preferences, self.LORA_IMG)
        self.lora_items_view.item_imported.connect(self.on_lora_imported)
        self.lora_items_view.model_item_clicked.connect(self.on_lora_item_clicked)
        self.lora_items_view.finished_loading.connect(self.on_finished_loading_loras)
        content_layout.addWidget(self.lora_items_view)

        lora_frame = QFrame()
        self.lora_frame_layout = QVBoxLayout()

        lora_frame.setLayout(self.lora_frame_layout)
        lora_frame.setFixedWidth(350)
        lora_frame.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        content_layout.addWidget(lora_frame)
        self.main_layout.addLayout(content_layout)

    def load_settings(self):
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def save_settings(self):
        self.settings.setValue("geometry", self.saveGeometry())

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

    def clear_selected_lora(self):
        self.selected_lora = None

        for i in reversed(range(self.lora_frame_layout.count())):
            widget_to_remove = self.lora_frame_layout.itemAt(i).widget()
            self.lora_frame_layout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)

    def on_lora_item_clicked(self, data):
        self.clear_selected_lora()

        self.selected_lora = LoraDataObject(
            enabled=True,
            name=data["name"],
            filename=data["root_filename"],
            version=data["version"],
            path=data["filepath"],
        )

        lora_info_widget = LoraInfoWidget(data)
        lora_info_widget.lora_selected.connect(self.on_lora_selected)
        lora_info_widget.lora_edit.connect(self.on_lora_edit_clicked)
        lora_info_widget.trigger_clicked.connect(self.on_trigger_clicked)
        lora_info_widget.example_prompt_clicked.connect(self.on_example_prompt_clicked)
        self.lora_frame_layout.addWidget(lora_info_widget)

    def on_lora_edit_clicked(self, data):
        self.clear_selected_lora()

        lora_edit_widget = LoraEditWidget(
            self.image_viewer,
            data["filepath"],
            lora_name=data["name"],
            lora_version=data["version"],
            lora_image=data["image"],
            default_image=data["default_image"],
            description=data["description"],
            tags=data["tags"],
            triggers=data["triggers"],
            example_prompt=data["example_prompt"],
        )
        lora_edit_widget.lora_info_saved.connect(self.on_info_saved)
        self.lora_frame_layout.addWidget(lora_edit_widget)

    def on_info_saved(
        self,
        name: str,
        filepath: str,
        directory: str,
        version: str,
        pixmap: Optional[QPixmap],
    ):
        filename = os.path.basename(filepath)
        root_filename, _ = os.path.splitext(filename)

        data = {
            "name": name,
            "root_filename": root_filename,
            "directory": directory,
            "version": version,
            "filepath": filepath,
        }

        lora_items = self.lora_items_view.flow_layout.items()
        edited_item = None
        for i, lora in enumerate(lora_items):
            lora_item = cast(ModelItem, lora.widget())
            if lora_item.data["root_filename"] == root_filename:
                edited_item = self.lora_items_view.flow_layout.itemAt(i)
                break

        if edited_item is not None:
            edited_item = cast(ModelItem, edited_item.widget())
            edited_item.data = data
            edited_item.image_widget.name_label.setText(name)
            edited_item.image_widget.set_model_version(version)

            if pixmap is not None:
                scaled_pixmap = pixmap.scaled(
                    QSize(
                        self.lora_items_view.thumb_width,
                        self.lora_items_view.thumb_height,
                    ),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                edited_item.image_widget.image_label.setPixmap(scaled_pixmap)

        self.on_lora_item_clicked(data)

    def on_finished_loading_loras(self):
        self.loading_loras = False

    def on_trigger_clicked(self, trigger):
        self.prompt_window.positive_prompt.insertTriggerAtCursor(trigger)

    def on_example_prompt_clicked(self, prompt):
        self.prompt_window.positive_prompt.insertTextAtCursor(prompt)

    def on_lora_selected(self):
        self.event_bus.publish("lora", {"action": "add", "lora": self.selected_lora})

    def on_lora_imported(self, path: str):
        if path.endswith(".safetensors"):
            file_name = os.path.basename(path)
            model_new_path = os.path.join(self.directories.models_loras, file_name)

            shutil.move(path, model_new_path)
            self.lora_items_view.add_single_item_from_path(model_new_path, "safetensors")
