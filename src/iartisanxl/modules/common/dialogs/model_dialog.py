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
from iartisanxl.generation.model_data_object import ModelDataObject
from iartisanxl.modules.common.dialogs.model_info_widget import ModelInfoWidget
from iartisanxl.modules.common.dialogs.modeL_edit_widget import ModelEditWidget
from iartisanxl.modules.common.dialogs.base_dialog import BaseDialog
from iartisanxl.modules.common.dialogs.model_items_view import ModelItemsView
from iartisanxl.modules.common.model_item import ModelItem


class ModelDialog(BaseDialog):
    MODEL_IMG = files("iartisanxl.theme.images").joinpath("model.webp")
    model_selected = pyqtSignal(ModelDataObject)

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.setWindowTitle("Base model")
        self.setMinimumSize(1160, 800)

        self.loading_models = False
        self.selected_model = None

        self.settings = QSettings("ZCode", "ImageArtisanXL")
        self.settings.beginGroup("model_dialog")
        self.load_settings()

        diffusers_directory = {
            "path": self.directories.models_diffusers,
            "type": "diffusers",
        }

        safetensors_directory = {
            "path": self.directories.models_safetensors,
            "type": "safetensors",
        }
        self.model_directories = (diffusers_directory, safetensors_directory)

        self.event_bus = EventBus()

        self.init_ui()

    def init_ui(self):
        content_layout = QHBoxLayout()

        self.model_items_view = ModelItemsView(self.model_directories, self.MODEL_IMG)
        self.model_items_view.model_item_clicked.connect(self.on_model_item_clicked)
        self.model_items_view.item_imported.connect(self.on_model_imported)
        self.model_items_view.finished_loading.connect(self.on_finished_loading_models)
        content_layout.addWidget(self.model_items_view)

        model_frame = QFrame()
        self.model_frame_layout = QVBoxLayout()

        model_frame.setLayout(self.model_frame_layout)
        model_frame.setFixedWidth(350)
        model_frame.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        )
        content_layout.addWidget(model_frame)
        self.main_layout.addLayout(content_layout)

    def dialog_raised(self):
        super().dialog_raised()
        self.loading_models = True
        self.model_items_view.load_items()

    def load_settings(self):
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def save_settings(self):
        self.settings.setValue("geometry", self.saveGeometry())

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

    def clear_selected_model(self):
        self.selected_model = None

        for i in reversed(range(self.model_frame_layout.count())):
            widget_to_remove = self.model_frame_layout.itemAt(i).widget()
            self.model_frame_layout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)

    def on_finished_loading_models(self):
        self.loading_models = False

    def on_model_item_clicked(self, data: dict):
        self.clear_selected_model()

        self.selected_model = ModelDataObject(
            name=data["name"],
            path=data["filepath"],
            version=data["version"],
            type=data["type"],
        )

        model_info_widget = ModelInfoWidget(data, self.directories.models_diffusers)
        self.model_frame_layout.addWidget(model_info_widget)
        model_info_widget.model_selected.connect(self.on_model_selected)
        model_info_widget.model_edit.connect(self.on_model_edit_clicked)
        model_info_widget.model_converted.connect(self.on_model_converted)

    def on_model_edit_clicked(self, data):
        self.clear_selected_model()

        model_edit_widget = ModelEditWidget(
            self.image_viewer,
            data["filepath"],
            model_name=data["name"],
            model_type=data["type"],
            model_version=data["version"],
            model_image=data["image"],
            default_image=data["default_image"],
            description=data["description"],
            tags=data["tags"],
        )
        model_edit_widget.model_info_saved.connect(self.on_info_saved)
        self.model_frame_layout.addWidget(model_edit_widget)

    def on_info_saved(
        self,
        name: str,
        filepath: str,
        version: str,
        model_type: str,
        pixmap: Optional[QPixmap],
    ):
        filename = os.path.basename(filepath)

        if model_type == "diffusers":
            root_filename = filename
        else:
            root_filename, _ = os.path.splitext(filename)

        data = {
            "name": name,
            "filepath": filepath,
            "version": version,
            "root_filename": root_filename,
            "type": model_type,
        }

        model_items = self.model_items_view.flow_layout.items()
        edited_item = None
        for i, model in enumerate(model_items):
            model_item = cast(ModelItem, model.widget())
            if model_item.data["root_filename"] == root_filename:
                edited_item = self.model_items_view.flow_layout.itemAt(i)
                break

        if edited_item is not None:
            edited_item = cast(ModelItem, edited_item.widget())
            edited_item.data = data
            edited_item.image_widget.name_label.setText(name)
            edited_item.image_widget.set_model_version(version)

            if pixmap is not None:
                scaled_pixmap = pixmap.scaled(
                    QSize(
                        self.model_items_view.thumb_width,
                        self.model_items_view.thumb_height,
                    ),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                edited_item.image_widget.image_label.setPixmap(scaled_pixmap)

        self.on_model_item_clicked(data)

    def on_model_converted(self, model_name: str):
        model_items = self.model_items_view.flow_layout.items()

        converted_item = None
        for i, model in enumerate(model_items):
            model_item = cast(ModelItem, model.widget())
            if model_item.data["root_filename"] == model_name:
                converted_item = self.model_items_view.flow_layout.itemAt(i)
                break

        if converted_item is not None:
            converted_item = cast(ModelItem, converted_item.widget())
            data = {
                "name": model_name,
                "filepath": os.path.join(self.directories.models_diffusers, model_name),
                "version": "",
                "root_filename": model_name,
                "type": "diffusers",
            }
            converted_item.data = data
            converted_item.image_widget.name_label.setText(model_name)
            converted_item.image_widget.type_label.setText("diffusers")
            self.on_model_item_clicked(data)

    def on_model_selected(self):
        self.image_generation_data.model = self.selected_model
        self.event_bus.publish("selected_model", {"model": self.selected_model})

    def on_model_imported(self, path: str):
        if path.endswith(".safetensors"):
            file_name = os.path.basename(path)
            model_new_path = os.path.join(
                self.directories.models_safetensors, file_name
            )

            shutil.move(path, model_new_path)
            self.model_items_view.add_single_item_from_path(
                model_new_path, "safetensors"
            )
        else:
            if os.path.isdir(path) and os.path.exists(
                os.path.join(path, "model_index.json")
            ):
                shutil.move(path, self.directories.models_diffusers)
                dir_name = os.path.basename(path)
                self.model_items_view.add_single_item_from_path(
                    os.path.join(self.directories.models_diffusers, dir_name),
                    "diffusers",
                )
