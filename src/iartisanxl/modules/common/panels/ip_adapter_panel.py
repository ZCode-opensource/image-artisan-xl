from PyQt6.QtWidgets import QVBoxLayout, QPushButton, QWidget

from iartisanxl.app.event_bus import EventBus
from iartisanxl.modules.common.panels.base_panel import BasePanel
from iartisanxl.modules.common.dialogs.ip_adapter_dialog import IPAdapterDialog
from iartisanxl.modules.common.ip_adapter_added_item import IPAdapterAddedItem
from iartisanxl.formats.image import ImageProcessor


class IPAdapterPanel(BasePanel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_bus = EventBus()
        self.event_bus.subscribe("ip_adapters", self.on_ip_adapters)

        self.init_ui()
        self.update_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        add_adapter_button = QPushButton("Add IP Adapter")
        add_adapter_button.clicked.connect(self.open_ip_adapter_dialog)
        main_layout.addWidget(add_adapter_button)

        added_adapters_widget = QWidget()
        self.adapters_layout = QVBoxLayout(added_adapters_widget)
        main_layout.addWidget(added_adapters_widget)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def update_ui(self):
        if len(self.ip_adapter_list.adapters) > 0:
            for adapter in self.ip_adapter_list.adapters:
                adapter_widget = IPAdapterAddedItem(adapter)
                adapter_widget.remove_clicked.connect(self.on_remove_clicked)
                adapter_widget.edit_clicked.connect(self.on_edit_clicked)
                adapter_widget.enabled.connect(self.on_enabled)
                self.adapters_layout.addWidget(adapter_widget)

    def open_ip_adapter_dialog(self):
        self.parent().open_dialog(
            "ip",
            IPAdapterDialog,
            self.directories,
            self.preferences,
            "IP adapter",
            self.show_error,
            self.image_generation_data,
            self.image_viewer,
            self.prompt_window,
        )

    def on_ip_adapters(self, data):
        if data["action"] == "add":
            adataper_id = self.ip_adapter_list.add(data["ip_adapter"])
            data["ip_adapter"].adapter_id = adataper_id
            adapter_widget = IPAdapterAddedItem(data["ip_adapter"])
            adapter_widget.remove_clicked.connect(self.on_remove_clicked)
            adapter_widget.edit_clicked.connect(self.on_edit_clicked)
            adapter_widget.enabled.connect(self.on_enabled)
            self.adapters_layout.addWidget(adapter_widget)
        elif data["action"] == "update":
            adapter = data["ip_adapter"]
            self.t2i_adapter_list.update_with_adapter_data_object(adapter)
            for i in range(self.adapters_layout.count()):
                widget = self.adapters_layout.itemAt(i).widget()
                if widget.adapter.adapter_id == adapter.adapter_id:
                    widget.enabled_checkbox.setText(adapter.adapter_type)
                    image_processor = ImageProcessor()
                    image_processor.set_pillow_image(adapter.image_thumb)
                    widget.image_thumb.setPixmap(image_processor.get_qpixmap())
                    break

    def on_remove_clicked(self, adapter_widget: IPAdapterAddedItem):
        self.ip_adapter_list.remove(adapter_widget.adapter)
        self.adapters_layout.removeWidget(adapter_widget)
        adapter_widget.deleteLater()

        if self.parent().ip_dialog is not None:
            self.parent().ip_dialog.adapter = None
            self.parent().ip_dialog.reset_ui()

    def on_edit_clicked(self, adapter: IPAdapterAddedItem):
        if self.parent().ip_dialog is None:
            self.open_ip_adapter_dialog()

        self.parent().ip_dialog.adapter = adapter
        self.parent().ip_dialog.update_ui()

    def on_enabled(self, adapter_id, enabled):
        self.ip_adapter_list.update_adapter(adapter_id, {"enabled": enabled})

    def clean_up(self):
        self.event_bus.unsubscribe("ip_adapters", self.on_ip_adapters)
        super().clean_up()
