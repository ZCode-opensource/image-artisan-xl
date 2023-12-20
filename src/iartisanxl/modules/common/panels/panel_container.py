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
        self.t2i_dialog = None
        self.ip_dialog = None

        self.init_ui()

    def init_ui(self):
        self.panel_layout = QVBoxLayout()
        self.panel_layout.setContentsMargins(0, 0, 0, 0)
        self.panel_layout.setSpacing(0)
        self.setLayout(self.panel_layout)

    def open_dialog(self, dialog_name, dialog_class, *args, **kwargs):
        dialog = getattr(self, f"{dialog_name}_dialog")
        if dialog is None:
            dialog = dialog_class(*args, **kwargs)
            setattr(self, f"{dialog_name}_dialog", dialog)
            dialog.closed.connect(lambda: self.on_dialog_closed(dialog_name))
            dialog.show()
        else:
            dialog.raise_()
            dialog.activateWindow()

    def on_dialog_closed(self, dialog_name):
        setattr(self, f"{dialog_name}_dialog", None)

    def close_all_dialogs(self):
        for attr in ["model_dialog", "lora_dialog", "controlnet_dialog", "t2i_dialog", "ip_dialog"]:
            dialog = getattr(self, attr)
            if dialog is not None:
                dialog.close()
