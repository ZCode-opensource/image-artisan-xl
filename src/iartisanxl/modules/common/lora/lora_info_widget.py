import io
import base64
import json

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
    QSizePolicy,
)

from iartisanxl.app.event_bus import EventBus
from iartisanxl.layouts.simple_flow_layout import SimpleFlowLayout
from iartisanxl.modules.common.model_utils import get_metadata_from_safetensors
from iartisanxl.modules.common.dialogs.example_prompt import ExamplePrompt


class LoraInfoWidget(QWidget):
    LORA_IMG = files("iartisanxl.theme.images").joinpath("lora.webp")
    lora_selected = pyqtSignal()
    lora_edit = pyqtSignal(dict)
    generate_example = pyqtSignal(str)
    trigger_clicked = pyqtSignal(str)
    example_prompt_clicked = pyqtSignal(str)

    def __init__(
        self,
        data: dict,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.data = {}
        self.lora_name = data["name"]
        self.root_filename = data["root_filename"]
        self.filepath = data["filepath"]
        self.generation_data = None
        self.default_image = True

        self.event_bus = EventBus()

        self.init_ui()
        self.load_info()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(3, 0, 3, 0)

        self.lora_image_label = QLabel()
        self.lora_image_label.setFixedWidth(345)
        self.lora_image_label.setMaximumHeight(480)
        self.lora_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.lora_image_label)

        self.lora_name_label = QLabel(self.lora_name)
        self.lora_name_label.setObjectName("model_name")
        self.lora_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.lora_name_label)

        self.version_layout = QHBoxLayout()
        self.version_layout.setSpacing(5)
        self.version_layout.addStretch()
        self.version_title_label = QLabel("Version")
        self.version_title_label.setObjectName("version_title")
        self.version_layout.addWidget(self.version_title_label)
        self.version_label = QLabel()
        self.version_layout.addWidget(self.version_label)
        main_layout.addLayout(self.version_layout)

        self.lora_description_label = QLabel()
        self.lora_description_label.setObjectName("description")
        self.lora_description_label.setWordWrap(True)
        main_layout.addWidget(self.lora_description_label)

        spacer = QSpacerItem(0, 10)
        main_layout.addSpacerItem(spacer)

        self.tags_title_label = QLabel("Tags:")
        self.tags_title_label.setObjectName("tags_title")
        self.tags_label = QLabel()
        self.tags_label.setObjectName("tags")
        main_layout.addWidget(self.tags_title_label)
        main_layout.addWidget(self.tags_label)

        spacer = QSpacerItem(0, 10)
        main_layout.addSpacerItem(spacer)

        self.trigger_label = QLabel("Trigger words:")
        self.trigger_label.setObjectName("trigger_words")
        main_layout.addWidget(self.trigger_label)

        spacer = QSpacerItem(0, 5)
        main_layout.addSpacerItem(spacer)

        trigger_words_container = QWidget()
        self.triggers_layout = SimpleFlowLayout()
        self.triggers_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.triggers_layout.setSpacing(4)
        trigger_words_container.setLayout(self.triggers_layout)
        main_layout.addWidget(trigger_words_container)

        spacer = QSpacerItem(0, 10)
        main_layout.addSpacerItem(spacer)

        self.prompt_label = QLabel("Example prompt:")
        self.prompt_label.setObjectName("prompt_label")
        main_layout.addWidget(self.prompt_label)

        spacer = QSpacerItem(0, 5)
        main_layout.addSpacerItem(spacer)

        self.example_prompt_text_label = ExamplePrompt()
        self.example_prompt_text_label.clicked.connect(self.on_example_prompt_clicked)
        main_layout.addWidget(self.example_prompt_text_label)

        main_layout.addStretch()

        self.edit_lora_button = QPushButton("Edit")
        self.edit_lora_button.clicked.connect(self.on_edit_clicked)
        main_layout.addWidget(self.edit_lora_button)

        self.generate_example_button = QPushButton("Generate example")
        self.generate_example_button.clicked.connect(self.on_generate_example)
        self.generate_example_button.setVisible(False)
        main_layout.addWidget(self.generate_example_button)

        self.add_lora_button = QPushButton("Add LoRA")
        self.add_lora_button.setObjectName("green_button")
        self.add_lora_button.clicked.connect(self.lora_selected.emit)
        main_layout.addWidget(self.add_lora_button)

        self.setLayout(main_layout)

    def load_info(self):
        metadata = get_metadata_from_safetensors(self.filepath)

        image = metadata.get("iartisan_image", None)
        version = metadata.get("iartisan_version", None)
        description = metadata.get("iartisan_description", None)
        tags = metadata.get("iartisan_tags", None)
        triggers = metadata.get("iartisan_triggers", None)
        example_prompt = metadata.get("iartisan_example_prompt", None)
        example_generation = metadata.get("iartisan_example_generation", None)

        if image is not None:
            img_bytes = base64.b64decode(image)
            buffer = io.BytesIO(img_bytes)
            qimage = QImage.fromData(buffer.getvalue())
            pixmap = QPixmap.fromImage(qimage)
            self.default_image = False
        else:
            pixmap = QPixmap(str(self.LORA_IMG))

        self.lora_image_label.setPixmap(pixmap)

        self.version_label.setText(version)
        if version is None:
            self.version_title_label.setVisible(False)

        if description is None:
            self.lora_description_label.setVisible(False)
        else:
            self.lora_description_label.setText(description)

        self.tags_label.setText(tags)
        if tags is None:
            self.tags_title_label.setVisible(False)

        if triggers is not None:
            triggers_list = [tag.strip() for tag in triggers.split(",")]

            for trigger in triggers_list:
                button = QPushButton(trigger)
                button.setObjectName("trigger_item")
                button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                button.setCursor(Qt.CursorShape.PointingHandCursor)
                button.clicked.connect(self.on_trigger_clicked)
                self.triggers_layout.addWidget(button)
        else:
            self.trigger_label.setVisible(False)

        self.example_prompt_text_label.setText(example_prompt)
        if example_prompt is None:
            self.prompt_label.setVisible(False)
            self.example_prompt_text_label.setVisible(False)

        if example_generation is not None:
            decoded_str = base64.b64decode(example_generation).decode("utf-8")
            self.generation_data = json.loads(decoded_str)
            self.generate_example_button.setVisible(True)

        self.data = {
            "root_filename": self.root_filename,
            "filepath": self.filepath,
            "name": self.lora_name,
            "image": pixmap,
            "default_image": self.default_image,
            "description": description,
            "version": version,
            "tags": tags,
            "triggers": triggers,
            "example_prompt": example_prompt,
        }

    def on_edit_clicked(self):
        self.lora_edit.emit(self.data)

    def on_generate_example(self):
        self.event_bus.publish("auto_generate", {"generation_data": self.generation_data})

    def on_trigger_clicked(self):
        button = self.sender()
        self.trigger_clicked.emit(button.text())

    def on_example_prompt_clicked(self):
        self.example_prompt_clicked.emit(self.example_prompt_text_label.text())
