from PyQt6.QtWidgets import QVBoxLayout, QPushButton, QWidget

from iartisanxl.app.event_bus import EventBus
from iartisanxl.modules.common.panels.base_panel import BasePanel
from iartisanxl.modules.common.dialogs.t2i_dialog import T2IDialog
from iartisanxl.modules.common.adapter_added_item import AdapterAddedItem
from iartisanxl.generation.t2i_adapter_data_object import T2IAdapterDataObject
from iartisanxl.app.preferences import PreferencesObject
from iartisanxl.formats.image import ImageProcessor


class T2IPanel(BasePanel):
    def __init__(self, preferences: PreferencesObject, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preferences = preferences

        self.event_bus = EventBus()
        self.event_bus.subscribe("t2i_adapters", self.on_t2i_adapters)
        self.t2i_adapters = []

        self.init_ui()
        self.update_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        add_adapter_button = QPushButton("Add T2I Adapter")
        add_adapter_button.clicked.connect(self.open_t2i_dialog)
        main_layout.addWidget(add_adapter_button)

        added_adapters_widget = QWidget()
        self.adapters_layout = QVBoxLayout(added_adapters_widget)
        main_layout.addWidget(added_adapters_widget)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def update_ui(self):
        if len(self.t2i_adapter_list.adapters) > 0:
            for adapter in self.t2i_adapter_list.adapters:
                adapter_widget = AdapterAddedItem(adapter)
                adapter_widget.remove_clicked.connect(self.on_remove_clicked)
                adapter_widget.edit_clicked.connect(self.on_edit_clicked)
                adapter_widget.enabled.connect(self.on_enabled)
                self.adapters_layout.addWidget(adapter_widget)

    def open_t2i_dialog(self):
        self.dialog_opened.emit(self, T2IDialog, "T2IDialog")

    def on_t2i_adapters(self, data):
        if data["action"] == "add":
            adataper_id = self.t2i_adapter_list.add(data["t2i_adapter"])
            data["t2i_adapter"].adapter_id = adataper_id
            adapter_widget = AdapterAddedItem(data["t2i_adapter"])
            adapter_widget.remove_clicked.connect(self.on_remove_clicked)
            adapter_widget.edit_clicked.connect(self.on_edit_clicked)
            adapter_widget.enabled.connect(self.on_enabled)
            self.adapters_layout.addWidget(adapter_widget)
        elif data["action"] == "update":
            adapter = data["t2i_adapter"]
            self.t2i_adapter_list.update_with_adapter_data_object(adapter)
            for i in range(self.adapters_layout.count()):
                widget = self.adapters_layout.itemAt(i).widget()
                if widget.adapter.adapter_id == adapter.adapter_id:
                    widget.enabled_checkbox.setText(adapter.adapter_type)
                    image_processor = ImageProcessor()
                    image_processor.set_pillow_image(adapter.source_image_thumb)
                    widget.source_thumb.setPixmap(image_processor.get_qpixmap())
                    image_processor.set_pillow_image(adapter.annotator_image_thumb)
                    widget.annotator_thumb.setPixmap(image_processor.get_qpixmap())
                    widget.adapter = data["t2i_adapter"]
                    break

    def on_remove_clicked(self, adapter_widget: AdapterAddedItem):
        self.t2i_adapter_list.remove(adapter_widget.adapter)
        self.adapters_layout.removeWidget(adapter_widget)
        adapter_widget.deleteLater()

    def on_edit_clicked(self, adapter: T2IAdapterDataObject):
        if self.current_dialog is None or not self.current_dialog.isVisible():
            self.dialog_opened.emit(self, T2IDialog, "T2IDialog")

        self.current_dialog.adapter = adapter
        self.current_dialog.update_ui()

    def clean_up(self):
        self.event_bus.unsubscribe("t2i_adapters", self.on_t2i_adapters)

    def on_enabled(self, adapter_id, enabled):
        self.t2i_adapter_list.update_adapter(adapter_id, {"enabled": enabled})
