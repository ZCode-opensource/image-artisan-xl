import torch

from PyQt6.QtWidgets import QVBoxLayout, QPushButton, QWidget

from iartisanxl.app.event_bus import EventBus
from iartisanxl.modules.common.panels.base_panel import BasePanel
from iartisanxl.modules.common.dialogs.t2i_dialog import T2IDialog
from iartisanxl.modules.common.adapter_added_item import AdapterAddedItem
from iartisanxl.generation.t2i_adapter_data_object import T2IAdapterDataObject
from iartisanxl.formats.image import ImageProcessor


class T2IPanel(BasePanel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_bus = EventBus()
        self.event_bus.subscribe("t2i_adapters", self.on_t2i_adapters)
        self.t2i_adapters = []
        self.dialog = None

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
        if self.parent().t2i_adapter_dialog is None:
            self.parent().t2i_adapter_dialog = T2IDialog(
                self.directories,
                "T2I adapter",
                self.show_error,
                self.image_generation_data,
                self.image_viewer,
                self.prompt_window,
            )
            self.parent().t2i_adapter_dialog.closed.connect(self.on_dialog_closed)
            self.parent().t2i_adapter_dialog.show()
        else:
            self.parent().t2i_adapter_dialog.raise_()
            self.parent().t2i_adapter_dialog.activate()

    def on_dialog_closed(self):
        self.parent().t2i_adapter_dialog.depth_estimator = None
        self.parent().t2i_adapter_dialog = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

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

        if self.dialog is not None:
            self.dialog.adapter = None
            self.dialog.reset_ui()

    def on_edit_clicked(self, adapter: T2IAdapterDataObject):
        if self.dialog is None:
            self.open_t2i_dialog()

        self.dialog.adapter = adapter
        self.dialog.update_ui()

    def clean_up(self):
        self.event_bus.unsubscribe("t2i_adapters", self.on_t2i_adapters)
        super().clean_up()

    def on_enabled(self, adapter_id, enabled):
        self.t2i_adapter_list.update_adapter(adapter_id, {"enabled": enabled})
