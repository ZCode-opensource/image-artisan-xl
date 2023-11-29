import io
import base64
import json
import os

import torch
from importlib.resources import files
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QSpacerItem,
    QHBoxLayout,
    QProgressBar,
)

from iartisanxl.app.event_bus import EventBus
from iartisanxl.modules.common.model_utils import get_metadata_from_safetensors
from iartisanxl.threads.convert_safetensors_thread import ConvertSafetensorsThread


class ModelInfoWidget(QWidget):
    MODEL_IMG = files("iartisanxl.theme.images").joinpath("model.webp")
    model_selected = pyqtSignal()
    model_edit = pyqtSignal(dict)
    model_converted = pyqtSignal(str)

    def __init__(
        self,
        data: dict,
        diffusers_directory: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.data = {}
        self.model_name = data["name"]
        self.root_filename = data["root_filename"]
        self.filepath = data["filepath"]
        self.model_type = data["type"]
        self.generation_data = None
        self.default_image = True
        self.diffusers_directory = diffusers_directory
        self.convert_safetensors_thread = None

        self.event_bus = EventBus()

        self.init_ui()
        self.load_info()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(3, 0, 3, 0)

        self.model_image_label = QLabel()
        self.model_image_label.setFixedWidth(345)
        self.model_image_label.setMaximumHeight(480)
        self.model_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.model_image_label)

        self.model_name_label = QLabel(self.model_name)
        self.model_name_label.setObjectName("model_name")
        self.model_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.model_name_label)

        self.version_layout = QHBoxLayout()
        self.version_layout.setSpacing(5)
        self.version_layout.addStretch()
        self.version_title_label = QLabel("Version")
        self.version_title_label.setObjectName("version_title")
        self.version_layout.addWidget(self.version_title_label)
        self.version_label = QLabel()
        self.version_layout.addWidget(self.version_label)
        main_layout.addLayout(self.version_layout)

        self.model_description_label = QLabel()
        self.model_description_label.setObjectName("description")
        self.model_description_label.setWordWrap(True)
        main_layout.addWidget(self.model_description_label)

        spacer = QSpacerItem(0, 10)
        main_layout.addSpacerItem(spacer)

        self.tags_title_label = QLabel("Tags:")
        self.tags_title_label.setObjectName("tags_title")
        self.tags_label = QLabel()
        self.tags_label.setObjectName("tags")
        main_layout.addWidget(self.tags_title_label)
        main_layout.addWidget(self.tags_label)

        if self.model_type == "diffusers":
            main_layout.addStretch()
            self.edit_model_button = QPushButton("Edit")
            self.edit_model_button.clicked.connect(self.on_edit_clicked)
            main_layout.addWidget(self.edit_model_button)
        else:
            safetensors_label = QLabel(
                "You can't edit the metadata of a model in safetensors format. This is because we store this "
                "information in the same file, which makes it slow and requires a lot of RAM each time."
                "\n\nTo edit and save the metadata of your model, you need to convert it to the diffusers format."
            )
            safetensors_label.setWordWrap(True)
            main_layout.addWidget(safetensors_label)
            main_layout.addStretch()
            self.progress_label = QLabel()
            main_layout.addWidget(
                self.progress_label, alignment=Qt.AlignmentFlag.AlignCenter
            )
            self.progress_label.setVisible(False)

            main_layout.addSpacerItem(QSpacerItem(0, 5))

            self.convert_progress_bar = QProgressBar()
            self.convert_progress_bar.setMinimum(0)
            self.convert_progress_bar.setMaximum(8)
            self.convert_progress_bar.setValue(0)
            main_layout.addWidget(self.convert_progress_bar)
            self.convert_progress_bar.setVisible(False)

            main_layout.addSpacerItem(QSpacerItem(0, 5))

            self.convert_model_button = QPushButton("Convert to diffusers")
            self.convert_model_button.clicked.connect(self.on_convert_model)
            main_layout.addWidget(self.convert_model_button)

        self.generate_example_button = QPushButton("Generate example")
        self.generate_example_button.clicked.connect(self.on_generate_example)
        self.generate_example_button.setVisible(False)
        main_layout.addWidget(self.generate_example_button)

        self.select_model_button = QPushButton("Select model")
        self.select_model_button.setObjectName("green_button")
        self.select_model_button.clicked.connect(self.model_selected.emit)
        main_layout.addWidget(self.select_model_button)

        self.setLayout(main_layout)

    def load_info(self):
        if self.model_type == "diffusers":
            model_directory = os.path.join(self.filepath, "text_encoder")
            model_path = os.path.join(model_directory, "model.fp16.safetensors")
        else:
            model_path = self.filepath

        metadata = get_metadata_from_safetensors(model_path)

        image = metadata.get("iartisan_image")
        version = metadata.get("iartisan_version")
        description = metadata.get("iartisan_description")
        tags = metadata.get("iartisan_tags")
        example_generation = metadata.get("iartisan_example_generation")

        if image is not None:
            img_bytes = base64.b64decode(image)
            buffer = io.BytesIO(img_bytes)
            qimage = QImage.fromData(buffer.getvalue())
            pixmap = QPixmap.fromImage(qimage)
            self.default_image = False
        else:
            pixmap = QPixmap(str(self.MODEL_IMG))

        self.model_image_label.setPixmap(pixmap)

        self.version_label.setText(version)
        if version is None:
            self.version_title_label.setVisible(False)

        if description is None:
            self.model_description_label.setVisible(False)
        else:
            self.model_description_label.setText(description)

        self.tags_label.setText(tags)
        if tags is None:
            self.tags_title_label.setVisible(False)

        if example_generation is not None:
            decoded_str = base64.b64decode(example_generation).decode("utf-8")
            self.generation_data = json.loads(decoded_str)
            self.generate_example_button.setVisible(True)

        self.data = {
            "root_filename": self.root_filename,
            "filepath": self.filepath,
            "name": self.model_name,
            "type": self.model_type,
            "image": pixmap,
            "default_image": self.default_image,
            "description": description,
            "version": version,
            "tags": tags,
        }

    def on_edit_clicked(self):
        self.model_edit.emit(self.data)

    def on_generate_example(self):
        self.event_bus.publish(
            "auto_generate", {"generation_data": self.generation_data}
        )

    def on_convert_model(self):
        self.progress_label.setVisible(True)
        self.convert_progress_bar.setVisible(True)
        self.convert_model_button.setVisible(False)
        self.select_model_button.setVisible(False)

        self.convert_safetensors_thread = ConvertSafetensorsThread(
            self.filepath, self.root_filename, self.diffusers_directory
        )
        self.convert_safetensors_thread.status_changed.connect(
            self.udpate_conversion_status
        )
        self.convert_safetensors_thread.finished.connect(self.conversion_finished)
        self.convert_safetensors_thread.start()

    def udpate_conversion_status(self, text: str, step: int):
        self.convert_progress_bar.setValue(step)
        self.progress_label.setText(text)

    def conversion_finished(self):
        self.convert_safetensors_thread = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        self.model_converted.emit(self.root_filename)
