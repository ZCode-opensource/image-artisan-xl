import os
import base64
import copy
import json

from importlib.resources import files
from safetensors.torch import load_file, save_file
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QGridLayout,
)
from PyQt6.QtCore import Qt, pyqtSignal, QByteArray, QBuffer, QIODevice
from PyQt6.QtGui import QPixmap

from iartisanxl.modules.common.dialogs.custom_text_edit import CustomTextEdit
from iartisanxl.modules.common.image_viewer_simple import ImageViewerSimple
from iartisanxl.modules.common.model_utils import get_metadata_from_safetensors


class ModelEditWidget(QWidget):
    MODEL_IMG = files("iartisanxl.theme.images").joinpath("model.png")
    model_info_saved = pyqtSignal(str, str, str, str, object)

    def __init__(
        self,
        image_viewer: ImageViewerSimple,
        filepath: str,
        *args,
        model_image: QPixmap = None,
        default_image: bool = True,
        model_name: str = "",
        model_type: str = "",
        model_version: str = "",
        description: str = "",
        tags: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.image_viewer = image_viewer
        self.filepath = filepath
        self.model_image = model_image
        self.default_image = default_image
        self.model_name = model_name
        self.model_type = model_type
        self.model_version = model_version
        self.description = description
        self.tags = tags

        self.image_width = 345
        self.image_height = 345

        self.model_pixmap = None
        self.image_updated = False
        self.model_serialized_data = None

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(3, 0, 3, 0)

        self.model_image_label = QLabel()
        self.model_image_label.setFixedWidth(self.image_width)
        self.model_image_label.setFixedHeight(self.image_height)
        self.model_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        if self.model_image:
            pixmap = self.model_image
        else:
            pixmap = QPixmap(str(self.MODEL_IMG))

        self.model_image_label.setPixmap(pixmap)
        main_layout.addWidget(self.model_image_label)

        self.set_image_button = QPushButton("Set current image")
        self.set_image_button.clicked.connect(self.set_model_image)
        main_layout.addWidget(self.set_image_button)

        name_version_layout = QGridLayout()

        model_name_label = QLabel("Name: ")
        name_version_layout.addWidget(model_name_label, 0, 0)
        self.name_edit = QLineEdit(self.model_name)
        name_version_layout.addWidget(self.name_edit, 0, 1)
        model_version_label = QLabel("Version:")
        name_version_layout.addWidget(model_version_label, 1, 0)
        self.version_edit = QLineEdit(self.model_version)
        name_version_layout.addWidget(self.version_edit, 1, 1)
        main_layout.addLayout(name_version_layout)

        char_limit = 300
        description_label = QLabel("Description")
        main_layout.addWidget(description_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.description_edit = CustomTextEdit(char_limit=char_limit)
        main_layout.addWidget(self.description_edit)
        self.description_count_label = QLabel(f"0/{char_limit}")
        main_layout.addWidget(
            self.description_count_label, alignment=Qt.AlignmentFlag.AlignRight
        )
        self.description_edit.char_changed.connect(
            lambda x: self.description_count_label.setText(f"{x}/{char_limit}")
        )
        self.description_edit.setPlainText(self.description)

        tags_label = QLabel("Tags:")
        main_layout.addWidget(tags_label)
        self.tags_edit = QLineEdit(self.tags)
        main_layout.addWidget(self.tags_edit)

        main_layout.addStretch()

        if self.model_type == "diffusers":
            self.save_button = QPushButton("Save")
            self.save_button.setObjectName("green_button")
            self.save_button.clicked.connect(self.save_model_info)
            main_layout.addWidget(self.save_button)

        self.setLayout(main_layout)

    def set_model_image(self):
        if self.image_viewer.pixmap_item is not None:
            self.model_pixmap = self.image_viewer.pixmap_item.pixmap()
            self.model_serialized_data = self.image_viewer.serialized_data

            scaled_pixmap = self.model_pixmap.scaled(
                self.image_width,
                self.image_height,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )

            x = (scaled_pixmap.width() - self.image_width) // 2
            y = (scaled_pixmap.height() - self.image_height) // 2
            cropped_pixmap = scaled_pixmap.copy(
                x, y, self.image_width, self.image_height
            )

            self.model_image_label.setPixmap(cropped_pixmap)
            self.image_updated = True

    def save_model_info(self):
        if self.model_type == "diffusers":
            model_directory = os.path.join(self.filepath, "text_encoder")
            model_path = os.path.join(model_directory, "model.fp16.safetensors")

            image = None
            example_generation = None

            metadata = {
                "iartisan_name": self.name_edit.text(),
                "iartisan_description": self.description_edit.toPlainText(),
                "iartisan_version": self.version_edit.text(),
                "iartisan_tags": self.tags_edit.text(),
                "iartisan_image": None,
                "iartisan_example_generation": None,
            }
            safetensors_metadata = get_metadata_from_safetensors(model_path)

            if self.image_updated:
                qimage = self.model_image_label.pixmap().toImage()
                byte_array = QByteArray()
                buffer = QBuffer(byte_array)
                buffer.open(QIODevice.OpenModeFlag.WriteOnly)
                qimage.save(buffer, "WEBP")
                base64_data = base64.b64encode(byte_array).decode("utf-8")
                image = base64_data
                metadata["iartisan_image"] = image

                if self.model_serialized_data is not None:
                    json_str = json.dumps(self.model_serialized_data)
                    example_generation = base64.b64encode(
                        json_str.encode("utf-8")
                    ).decode("utf-8")
                    metadata["iartisan_example_generation"] = example_generation
            else:
                if not self.default_image:
                    metadata["iartisan_image"] = safetensors_metadata.get(
                        "iartisan_image"
                    )
                    metadata["iartisan_example_generation"] = safetensors_metadata.get(
                        "iartisan_example_generation"
                    )

            safetensors_metadata.update(metadata)
            safetensors_metadata = {
                k: v
                for k, v in safetensors_metadata.items()
                if v is not None and v != ""
            }
            converted_state_dict = copy.deepcopy(load_file(model_path))
            save_file(
                converted_state_dict,
                model_path,
                metadata=safetensors_metadata,
            )
            del converted_state_dict

            self.model_info_saved.emit(
                self.name_edit.text(),
                self.filepath,
                self.version_edit.text(),
                self.model_type,
                self.model_image_label.pixmap(),
            )
