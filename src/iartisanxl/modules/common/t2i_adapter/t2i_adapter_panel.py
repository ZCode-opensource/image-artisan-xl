import torch
from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from iartisanxl.modules.common.panels.base_panel import BasePanel
from iartisanxl.modules.common.t2i_adapter.t2i_adapter_added_item import T2IAdapterAddedItem
from iartisanxl.modules.common.t2i_adapter.t2i_adapter_data_object import T2IAdapterDataObject
from iartisanxl.modules.common.t2i_adapter.t2i_adapter_dialog import T2IAdapterDialog
from iartisanxl.utilities.image.operations import remove_image_data_files


class T2IAdapterPanel(BasePanel):
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
                adapter_widget = T2IAdapterAddedItem(adapter)
                adapter_widget.update_ui()
                adapter_widget.remove_clicked.connect(self.on_remove_clicked)
                adapter_widget.edit_clicked.connect(self.on_edit_clicked)
                adapter_widget.enabled.connect(self.on_enabled)
                self.adapters_layout.addWidget(adapter_widget)

    def open_t2i_dialog(self):
        if self.parent().t2i_dialog is not None:
            self.parent().t2i_dialog.reset_ui()

        self.parent().open_dialog(
            "t2i",
            T2IAdapterDialog,
            self.directories,
            self.preferences,
            "T2I adapter",
            self.show_error,
            self.image_generation_data,
            self.image_viewer,
            self.prompt_window,
        )

    def on_dialog_closed(self):
        self.parent().t2i_dialog = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def on_remove_clicked(self, t2i_adapter_widget: T2IAdapterAddedItem):
        if self.parent().t2i_dialog is not None:
            if t2i_adapter_widget.t2i_adapter.adapter_id == self.parent().t2i_dialog.t2i_adapter.adapter_id:
                self.parent().t2i_dialog.reset_ui()

        # delete images
        remove_image_data_files(t2i_adapter_widget.t2i_adapter.source_image)
        remove_image_data_files(t2i_adapter_widget.t2i_adapter.preprocessor_image)

        self.t2i_adapter_list.remove(t2i_adapter_widget.t2i_adapter)
        self.adapters_layout.removeWidget(t2i_adapter_widget)
        t2i_adapter_widget.deleteLater()

    def on_edit_clicked(self, t2i_adapter: T2IAdapterDataObject):
        if self.parent().t2i_dialog is None:
            self.open_t2i_dialog()

        self.parent().t2i_dialog.t2i_adapter = t2i_adapter
        self.parent().t2i_dialog.update_ui()

    def on_enabled(self, adapter_id, enabled):
        self.t2i_adapter_list.update_adapter(adapter_id, {"enabled": enabled})
