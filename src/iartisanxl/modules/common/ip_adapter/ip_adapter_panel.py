from PyQt6.QtWidgets import QVBoxLayout, QPushButton, QWidget

from iartisanxl.app.event_bus import EventBus
from iartisanxl.modules.common.panels.base_panel import BasePanel
from iartisanxl.modules.common.ip_adapter.ip_adapter_dialog import IPAdapterDialog
from iartisanxl.modules.common.ip_adapter.ip_adapter_added_item import IPAdapterAddedItem


class IPAdapterPanel(BasePanel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_bus = EventBus()

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
                adapter_widget.update_ui()
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

        if self.parent().ip_dialog is not None:
            self.parent().ip_dialog.make_new_adapter()

    def on_remove_clicked(self, adapter_widget: IPAdapterAddedItem):
        ip_adapter_id = adapter_widget.adapter.adapter_id
        self.ip_adapter_list.remove(adapter_widget.adapter)
        self.adapters_layout.removeWidget(adapter_widget)
        adapter_widget.deleteLater()

        if self.parent().ip_dialog is not None:
            if self.parent().ip_dialog.adapter.adapter_id == ip_adapter_id:
                self.parent().ip_dialog.make_new_adapter()

    def on_edit_clicked(self, adapter: IPAdapterAddedItem):
        if self.parent().ip_dialog is None:
            self.open_ip_adapter_dialog()

        self.parent().ip_dialog.adapter = adapter
        self.parent().ip_dialog.update_ui()

    def on_enabled(self, adapter_id, enabled):
        self.ip_adapter_list.update_adapter(adapter_id, {"enabled": enabled})
