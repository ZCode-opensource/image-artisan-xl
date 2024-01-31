import os

from PyQt6.QtWidgets import QVBoxLayout, QPushButton, QWidget

from iartisanxl.modules.common.panels.base_panel import BasePanel
from iartisanxl.modules.common.t2i_adapter.t2i_dialog import T2IDialog
from iartisanxl.modules.common.t2i_adapter.adapter_added_item import AdapterAddedItem
from iartisanxl.modules.common.t2i_adapter.t2i_adapter_data_object import T2IAdapterDataObject


class T2IPanel(BasePanel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
                adapter_widget.update_ui()
                adapter_widget.remove_clicked.connect(self.on_remove_clicked)
                adapter_widget.edit_clicked.connect(self.on_edit_clicked)
                adapter_widget.enabled.connect(self.on_enabled)
                self.adapters_layout.addWidget(adapter_widget)

    def open_t2i_dialog(self):
        self.parent().open_dialog(
            "t2i",
            T2IDialog,
            self.directories,
            self.preferences,
            "T2I adapter",
            self.show_error,
            self.image_generation_data,
            self.image_viewer,
            self.prompt_window,
        )

        if self.parent().t2i_dialog is not None:
            self.parent().t2i_dialog.reset_ui()

    def on_remove_clicked(self, adapter_widget: AdapterAddedItem):
        if self.parent().t2i_dialog is not None:
            if adapter_widget.adapter.adapter_id == self.parent().t2i_dialog.adapter.adapter_id:
                self.parent().t2i_dialog.reset_ui()

        # delete images
        if adapter_widget.adapter.source_image.image_original:
            os.remove(adapter_widget.adapter.source_image.image_original)

        if adapter_widget.adapter.source_image.image_filename:
            os.remove(adapter_widget.adapter.source_image.image_filename)

        if adapter_widget.adapter.source_image.image_thumb:
            os.remove(adapter_widget.adapter.source_image.image_thumb)

        if adapter_widget.adapter.preprocessor_image.image_original:
            os.remove(adapter_widget.adapter.preprocessor_image.image_original)

        if adapter_widget.adapter.preprocessor_image.image_filename:
            os.remove(adapter_widget.adapter.preprocessor_image.image_filename)

        if adapter_widget.adapter.preprocessor_image.image_thumb:
            os.remove(adapter_widget.adapter.preprocessor_image.image_thumb)

        if adapter_widget.adapter.source_image.image_drawings:
            os.remove(adapter_widget.adapter.source_image.image_drawings)

        if adapter_widget.adapter.preprocessor_image.image_drawings:
            os.remove(adapter_widget.adapter.preprocessor_image.image_drawings)

        self.t2i_adapter_list.remove(adapter_widget.adapter)
        self.adapters_layout.removeWidget(adapter_widget)
        adapter_widget.deleteLater()

    def on_edit_clicked(self, adapter: T2IAdapterDataObject):
        if self.parent().t2i_dialog is None:
            self.open_t2i_dialog()

        self.parent().t2i_dialog.adapter = adapter
        self.parent().t2i_dialog.update_ui()

    def on_enabled(self, adapter_id, enabled):
        self.t2i_adapter_list.update_adapter(adapter_id, {"enabled": enabled})
