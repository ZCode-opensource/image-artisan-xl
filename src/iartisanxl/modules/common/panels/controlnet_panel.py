from PyQt6.QtWidgets import (
    QVBoxLayout,
    QPushButton,
)

from iartisanxl.modules.common.panels.base_panel import BasePanel
from iartisanxl.modules.common.dialogs.controlnet_dialog import ControlNetDialog


class ControlNetPanel(BasePanel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        add_lora_button = QPushButton("Add ControlNet")
        add_lora_button.clicked.connect(self.open_controlnet_dialog)
        main_layout.addWidget(add_lora_button)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def open_controlnet_dialog(self):
        self.dialog_opened.emit(ControlNetDialog, "ControlNet")
