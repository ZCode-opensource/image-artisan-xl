import os
from datetime import datetime

from PyQt6.QtCore import QSettings, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout

from iartisanxl.modules.common.dialogs.base_dialog import BaseDialog
from iartisanxl.modules.common.ip_adapter.ip_adapter_data_object import IPAdapterDataObject
from iartisanxl.modules.common.mask.mask_widget import MaskWidget


class MaskDialog(BaseDialog):
    mask_saved = pyqtSignal()

    def __init__(self, adapter: IPAdapterDataObject, *args):
        super().__init__(*args)

        self.adapter = adapter

        self.setWindowTitle("Mask Editor")
        self.setMinimumSize(500, 500)

        self.settings = QSettings("ZCode", "ImageArtisanXL")
        self.settings.beginGroup("ip_adapters_dialog")
        geometry = self.settings.value("mask_dialog_geometry")
        if geometry:
            self.restoreGeometry(geometry)
        self.settings.endGroup()

        self.editor_width = self.image_generation_data.image_width
        self.editor_height = self.image_generation_data.image_height

        self.init_ui()

    def init_ui(self):
        content_layout = QVBoxLayout()

        mask_layout = QVBoxLayout()
        self.mask_editor_widget = MaskWidget(
            "", "ip_adapter_mask", self.image_viewer, self.editor_width, self.editor_height
        )
        mask_layout.addWidget(self.mask_editor_widget)
        content_layout.addLayout(mask_layout)

        bottom_layout = QHBoxLayout()
        self.save_mask_button = QPushButton("Save mask")
        self.save_mask_button.clicked.connect(self.on_save_mask)
        bottom_layout.addWidget(self.save_mask_button)
        content_layout.addLayout(bottom_layout)

        self.main_layout.addLayout(content_layout)

    def closeEvent(self, event):
        self.settings.beginGroup("ip_adapters_dialog")
        self.settings.setValue("mask_dialog_geometry", self.saveGeometry())
        self.settings.endGroup()

        super().closeEvent(event)

    def on_save_mask(self):
        mask_layer = self.mask_editor_widget.image_editor.layer_manager.get_layer_by_id(
            self.mask_editor_widget.drawing_layer_id
        )
        mask_pixmap = mask_layer.pixmap_item.pixmap()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        mask_filename = f"ip_adapter_{self.adapter.adapter_id}_{timestamp}_original_mask.png"
        mask_path = os.path.join("tmp/", mask_filename)
        mask_pixmap.save(mask_path)
        self.adapter.mask_alpha_image = mask_path
        self.mask_saved.emit()
