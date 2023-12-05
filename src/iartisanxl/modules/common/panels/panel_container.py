from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy


class PanelContainer(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setObjectName("panel_container")
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        self.model_dialog = None
        self.lora_dialog = None
        self.controlnet_dialog = None
        self.t2i_adapter_dialog = None
        self.ip_adapter_dialog = None

        self.init_ui()

    def init_ui(self):
        self.panel_layout = QVBoxLayout()
        self.panel_layout.setContentsMargins(0, 0, 0, 0)
        self.panel_layout.setSpacing(0)
        self.setLayout(self.panel_layout)
