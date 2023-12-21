import os

from importlib.resources import files
from PyQt6.QtWidgets import QLabel, QPushButton, QHBoxLayout, QSpacerItem
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QPixmap, QFont

from iartisanxl.configuration.base_setup_panel import BaseSetupPanel


class WelcomePanel(BaseSetupPanel):
    HEADER_IMG = str(files("iartisanxl.theme.images").joinpath("welcome.webp"))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.init_ui()

    def init_ui(self):
        welcome_label = QLabel("Welcome to Image Artisan XL")
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        welcome_label.setFont(font)
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(welcome_label)

        self.main_layout.addSpacerItem(QSpacerItem(0, 10))

        header_pixmap = QPixmap(self.HEADER_IMG)
        header_label = QLabel("")
        header_label.setPixmap(header_pixmap)
        self.main_layout.addWidget(header_label)

        self.main_layout.addSpacerItem(QSpacerItem(0, 10))

        welcome_label = QLabel(
            "<html><body>"
            "Since this is the first time you use this application, we need to know where do you want to store your models "
            "and images, don't worry you can still change them afterwards in the configuration.<br><br>"
            "You can choose to set them up now or leave the them to the defaults. Which is a directory inside your Documents "
            "in a subdirectory named <b>Image Artisan XL</b>.<br><br>"
            "Another configuration you can do now is the optimizations in case you don't have a GPU with enough VRAM. The "
            "default for the optimizations will need at least <b>16 GB of RAM</b> and <b>12 GB of VRAM</b>."
            "</body></html>"
        )
        welcome_label.setWordWrap(True)
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignJustify)
        self.main_layout.addWidget(welcome_label)

        self.main_layout.addStretch()

        button_layout = QHBoxLayout()
        default_button = QPushButton("Use defaults")
        default_button.clicked.connect(self.on_defaults)
        button_layout.addWidget(default_button)
        configure_button = QPushButton("Configure")
        configure_button.clicked.connect(self.on_next_step)
        button_layout.addWidget(configure_button)

        self.buttons_widget.setLayout(button_layout)
        self.main_layout.addWidget(self.buttons_widget)

    def on_defaults(self):
        base_dirs = ["Documents", "Image Artisan XL"]

        sub_dirs = {
            "models_diffusers": os.path.join("models", "diffusers"),
            "models_safetensors": os.path.join("models", "safetensors"),
            "vaes": os.path.join("models", "vae"),
            "models_loras": os.path.join("models", "loras"),
            "models_controlnets": os.path.join("models", "controlnet"),
            "models_t2i_adapters": os.path.join("models", "t2i-adapter"),
            "models_ip_adapters": os.path.join("models", "ip-adapter"),
            "outputs_images": os.path.join("outputs", "images"),
            "outputs_loras": os.path.join("outputs", "loras"),
            "datasets": "datasets",
        }

        home_dir = os.path.expanduser("~")

        for directory in base_dirs:
            home_dir = os.path.join(home_dir, directory)
            if not os.path.exists(home_dir):
                os.makedirs(home_dir)

        for key, directory in sub_dirs.items():
            sub_dir = os.path.join(home_dir, directory)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            setattr(self.directories, key, sub_dir)

        settings = QSettings("ZCode", "ImageArtisanXL")
        settings.setValue("models_diffusers", self.directories.models_diffusers)
        settings.setValue("models_safetensors", self.directories.models_safetensors)
        settings.setValue("vaes", self.directories.vaes)
        settings.setValue("models_loras", self.directories.models_loras)
        settings.setValue("models_controlnets", self.directories.models_controlnets)
        settings.setValue("models_t2i_adapters", self.directories.models_t2i_adapters)
        settings.setValue("models_ip_adapters", self.directories.models_ip_adapters)
        settings.setValue("outputs_images", self.directories.outputs_images)
        settings.setValue("outputs_loras", self.directories.outputs_loras)
        settings.setValue("datasets", self.directories.datasets)

        self.finish_setup.emit()
