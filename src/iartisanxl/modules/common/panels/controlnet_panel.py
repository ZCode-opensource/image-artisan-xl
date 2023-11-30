from PyQt6.QtWidgets import QVBoxLayout, QPushButton, QWidget

from iartisanxl.app.event_bus import EventBus
from iartisanxl.modules.common.panels.base_panel import BasePanel
from iartisanxl.modules.common.dialogs.controlnet_dialog import ControlNetDialog
from iartisanxl.modules.common.controlnet_added_item import ControlNetAddedItem
from iartisanxl.app.preferences import PreferencesObject
from iartisanxl.formats.image import ImageProcessor
from iartisanxl.generation.controlnet_data_object import ControlNetDataObject


class ControlNetPanel(BasePanel):
    def __init__(
        self,
        preferences: PreferencesObject,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.preferences = preferences

        self.event_bus = EventBus()
        self.event_bus.subscribe("controlnet", self.on_controlnet)
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
        self.dialog_opened.emit(self, ControlNetDialog, "ControlNet")

    def update_ui(self):
        if len(self.controlnet_list.controlnets) > 0:
            for controlnet in self.controlnet_list.controlnets:
                controlnet_widget = ControlNetAddedItem(controlnet)
                controlnet_widget.remove_clicked.connect(self.on_remove_clicked)
                controlnet_widget.edit_clicked.connect(self.on_edit_clicked)
                self.controlnets_layout.addWidget(controlnet_widget)

    def on_controlnet(self, data):
        if data["action"] == "add":
            controlnet_id = self.controlnet_list.add(data["controlnet"])
            data["controlnet"].controlnet_id = controlnet_id
            controlnet_widget = ControlNetAddedItem(data["controlnet"])
            controlnet_widget.remove_clicked.connect(self.on_remove_clicked)
            controlnet_widget.edit_clicked.connect(self.on_edit_clicked)
            self.controlnets_layout.addWidget(controlnet_widget)
        elif data["action"] == "update":
            controlnet = data["controlnet"]
            self.controlnet_list.update_with_controlnet_data_object(controlnet)
            for i in range(self.controlnets_layout.count()):
                widget = self.controlnets_layout.itemAt(i).widget()
                if widget.controlnet.controlnet_id == controlnet.controlnet_id:
                    widget.enabled_checkbox.setText(controlnet.controlnet_type)
                    image_processor = ImageProcessor()
                    image_processor.set_pillow_image(controlnet.source_image_thumb)
                    widget.source_thumb.setPixmap(image_processor.get_qpixmap())
                    image_processor.set_pillow_image(controlnet.annotator_image_thumb)
                    widget.annotator_thumb.setPixmap(image_processor.get_qpixmap())
                    widget.controlnet = data["controlnet"]
                    break

    def on_edit_clicked(self, controlnet: ControlNetDataObject):
        if self.current_dialog is None or not self.current_dialog.isVisible():
            self.dialog_opened.emit(self, ControlNetDialog, "ControlNet")

        self.current_dialog.controlnet = controlnet
        print(f"{controlnet=}")
        self.current_dialog.update_ui()

    def clear_controlnets(self):
        self.controlnets = []

        while self.controlnets_layout.count():
            item = self.controlnets_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def on_remove_clicked(self, controlnet_widget: ControlNetAddedItem):
        self.controlnet_list.remove(controlnet_widget.controlnet)
        self.controlnets_layout.removeWidget(controlnet_widget)
        controlnet_widget.deleteLater()

    def on_controlnet_enabled(self, controlnet_widget: ControlNetAddedItem):
        self.image_generation_data.change_controlnet_enabled(controlnet_widget.controlnet, controlnet_widget.enabled_checkbox.isChecked())

    def clean_up(self):
        self.event_bus.unsubscribe("controlnet", self.on_controlnet)
