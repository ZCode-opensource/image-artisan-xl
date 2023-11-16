from PyQt6.QtWidgets import QVBoxLayout, QPushButton, QWidget

from iartisanxl.modules.common.panels.base_panel import BasePanel
from iartisanxl.modules.common.dialogs.controlnet_dialog import ControlNetDialog
from iartisanxl.modules.common.controlnet_added_item import ControlNetAddedItem


class ControlNetPanel(BasePanel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.controlnets = []
        self.init_ui()
        self.update_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        add_controlnet_button = QPushButton("Add ControlNet")
        add_controlnet_button.clicked.connect(self.open_controlnet_dialog)
        main_layout.addWidget(add_controlnet_button)

        added_controlnets_widget = QWidget()
        self.controlnets_layout = QVBoxLayout(added_controlnets_widget)
        main_layout.addWidget(added_controlnets_widget)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def open_controlnet_dialog(self):
        self.dialog_opened.emit(ControlNetDialog, "ControlNet")

    def update_ui(self):
        controlnets = self.image_generation_data.controlnets
        self.clear_controlnets()

        if len(controlnets) > 0:
            for controlnet in controlnets:
                controlnet_widget = ControlNetAddedItem(controlnet)
                controlnet_widget.remove_clicked.connect(
                    lambda lw=controlnet_widget: self.on_remove_controlnet(lw)
                )
                controlnet_widget.enabled.connect(
                    lambda lw=controlnet_widget: self.on_controlnet_enabled(lw)
                )
                self.controlnets_layout.addWidget(controlnet_widget)
                self.controlnets.append(controlnet_widget)

    def clear_controlnets(self):
        self.controlnets = []

        while self.controlnets_layout.count():
            item = self.controlnets_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def on_remove_controlnet(self, controlnet_widget: ControlNetAddedItem):
        index = self.controlnets_layout.indexOf(controlnet_widget)
        if index != -1:
            self.controlnets_layout.takeAt(index)
            controlnet_widget.deleteLater()
            self.image_generation_data.remove_controlnet(
                self.controlnets[index].controlnet
            )
            del self.controlnets[index]

    def on_controlnet_enabled(self, controlnet_widget: ControlNetAddedItem):
        self.image_generation_data.change_controlnet_enabled(
            controlnet_widget.controlnet, controlnet_widget.enabled_checkbox.isChecked()
        )
