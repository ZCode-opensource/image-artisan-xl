from PyQt6.QtWidgets import (
    QCheckBox,
    QPushButton,
    QHBoxLayout,
    QFrame,
    QVBoxLayout,
    QSpacerItem,
    QLabel,
)
from PyQt6.QtCore import QSettings, Qt

from iartisanxl.configuration.base_setup_panel import BaseSetupPanel


class OptimizationsPanel(BaseSetupPanel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.init_ui()

    def init_ui(self):
        intermediate_images_frame = QFrame()
        intermediate_images_layout = QVBoxLayout()
        intermediate_images_label = QLabel(
            "<html><body>"
            "Using a Tiny AutoEncoder for Stable Diffusion, we can display the intermediate images "
            "from the start of the denoising process. TAESD can decode Stable Diffusion’s latent "
            "variables into full-size images at (almost) no cost."
            "</body></html>"
        )
        intermediate_images_label.setAlignment(Qt.AlignmentFlag.AlignJustify)
        intermediate_images_label.setWordWrap(True)
        intermediate_images_layout.addWidget(intermediate_images_label)
        intermediate_images_layout.addSpacerItem(QSpacerItem(0, 3))
        self.intermediate_images_checkbox = QCheckBox("Display intermediate images")
        self.intermediate_images_checkbox.stateChanged.connect(
            self.on_checkbox_state_changed
        )
        intermediate_images_layout.addWidget(
            self.intermediate_images_checkbox, alignment=Qt.AlignmentFlag.AlignCenter
        )
        intermediate_images_frame.setLayout(intermediate_images_layout)
        self.main_layout.addWidget(intermediate_images_frame)

        tomes_frame = QFrame()
        tomes_layout = QVBoxLayout()
        tomes_label = QLabel(
            "<html><body>"
            "Token Merging (ToMe) speeds up transformers by merging redundant tokens, which means the transformer has to do less work."
            "<br><br><b>Note:</b> this is a lossy process, so the image will change, ideally not by much."
            "</body></html>"
        )
        tomes_label.setAlignment(Qt.AlignmentFlag.AlignJustify)
        tomes_label.setWordWrap(True)
        tomes_layout.addWidget(tomes_label)
        tomes_layout.addSpacerItem(QSpacerItem(0, 3))
        self.tomes_base_checkbox = QCheckBox("Enable token merging")
        self.tomes_base_checkbox.stateChanged.connect(self.on_checkbox_state_changed)
        tomes_layout.addWidget(
            self.tomes_base_checkbox, alignment=Qt.AlignmentFlag.AlignCenter
        )
        tomes_frame.setLayout(tomes_layout)
        self.main_layout.addWidget(tomes_frame)

        offload_frame = QFrame()
        offload_layout = QVBoxLayout()
        offload_label = QLabel(
            "Full-model offloading moves whole models to the GPU. This results in a negligible impact on "
            "inference time while still providing some memory savings."
        )
        offload_label.setWordWrap(True)
        offload_label.setAlignment(Qt.AlignmentFlag.AlignJustify)
        offload_layout.addWidget(offload_label)
        offload_layout.addSpacerItem(QSpacerItem(0, 3))
        offload_checkboxes_layout = QHBoxLayout()
        self.offload_base_checkbox = QCheckBox("Enable CPU offload")
        self.offload_base_checkbox.stateChanged.connect(self.on_checkbox_state_changed)
        offload_checkboxes_layout.addWidget(
            self.offload_base_checkbox, alignment=Qt.AlignmentFlag.AlignCenter
        )
        offload_layout.addLayout(offload_checkboxes_layout)
        offload_layout.addSpacerItem(QSpacerItem(0, 5))
        sequential_label = QLabel(
            "Sequential CPU offloading preserves a lot of memory but makes inference slower, because submodules are"
            "moved to GPU as needed, and immediately returned to CPU when a new module runs."
        )
        sequential_label.setWordWrap(True)
        offload_layout.addWidget(sequential_label)
        offload_layout.addSpacerItem(QSpacerItem(0, 3))
        sequential_layout = QHBoxLayout()
        self.sequential_offload_checkbox = QCheckBox("Enable sequential")
        self.sequential_offload_checkbox.stateChanged.connect(
            self.on_checkbox_state_changed
        )
        sequential_layout.addWidget(
            self.sequential_offload_checkbox, alignment=Qt.AlignmentFlag.AlignCenter
        )
        offload_layout.addLayout(sequential_layout)
        offload_frame.setLayout(offload_layout)
        self.main_layout.addWidget(offload_frame)

        self.main_layout.addStretch()

        button_layout = QHBoxLayout()
        finish_button = QPushButton("Back")
        finish_button.clicked.connect(self.on_back_step)
        button_layout.addWidget(finish_button)
        next_step_button = QPushButton("Finish")
        next_step_button.clicked.connect(self.finish_setup.emit)
        button_layout.addWidget(next_step_button)

        self.buttons_widget.setLayout(button_layout)
        self.main_layout.addWidget(self.buttons_widget)

    def on_checkbox_state_changed(self):
        sender = self.sender()

        settings = QSettings("ZCode", "ImageArtisanXL")

        if sender == self.intermediate_images_checkbox:
            settings.setValue("intermediate_images", sender.isChecked())
            self.preferences.intermediate_images = sender.isChecked()
        elif sender == self.tomes_base_checkbox:
            settings.setValue("use_tomes", sender.isChecked())
            self.preferences.use_tomes = sender.isChecked()
        elif sender == self.offload_base_checkbox:
            settings.setValue("model_offload", sender.isChecked())
            self.preferences.model_offload = sender.isChecked()
        elif sender == self.sequential_offload_checkbox:
            settings.setValue("sequential_offload", sender.isChecked())
            self.preferences.sequential_offload = sender.isChecked()
