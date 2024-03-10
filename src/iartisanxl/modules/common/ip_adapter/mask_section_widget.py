import os
from datetime import datetime

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from iartisanxl.modules.common.image.image_data_object import ImageDataObject
from iartisanxl.modules.common.image_viewer_simple import ImageViewerSimple
from iartisanxl.modules.common.ip_adapter.ip_adapter_data_object import IPAdapterDataObject
from iartisanxl.modules.common.mask.mask_image import MaskImage
from iartisanxl.modules.common.mask.mask_widget import MaskWidget


class MaskSectionWidget(QWidget):
    mask_saved = pyqtSignal(QPixmap)
    mask_canceled = pyqtSignal()

    def __init__(
        self, ip_adapter: IPAdapterDataObject, image_viewer: ImageViewerSimple, target_width: int, target_height: int
    ):
        super().__init__()

        self.ip_adapter = ip_adapter

        self.setWindowTitle("Mask Editor")
        self.setMinimumSize(500, 500)

        self.image_viewer = image_viewer
        self.target_width = target_width
        self.target_height = target_height

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        mask_layout = QVBoxLayout()
        self.image_widget = MaskWidget("", "ip_adapter_mask", self.image_viewer, self.target_width, self.target_height)
        mask_layout.addWidget(self.image_widget)
        main_layout.addLayout(mask_layout)

        bottom_layout = QHBoxLayout()
        self.save_mask_button = QPushButton("Save mask")
        self.save_mask_button.clicked.connect(self.on_save_mask)
        bottom_layout.addWidget(self.save_mask_button)
        cancel_mask_button = QPushButton("Cancel")
        cancel_mask_button.clicked.connect(self.on_cancel_mask)
        bottom_layout.addWidget(cancel_mask_button)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

    def on_save_mask(self):
        bg_layer = self.image_widget.image_editor.layer_manager.get_layer_by_id(self.image_widget.image_layer_id)

        mask_layer = self.image_widget.image_editor.layer_manager.get_layer_by_id(self.image_widget.drawing_layer_id)
        mask_pixmap = mask_layer.pixmap_item.pixmap()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        mask_filename = f"ip_adapter_{self.ip_adapter.adapter_id}_{timestamp}_mask.png"
        mask_path = os.path.join("tmp/", mask_filename)
        mask_pixmap.save(mask_path)

        thumb_pixmap = mask_pixmap.scaled(
            80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        thumb_filename = f"ip_adapter_{self.ip_adapter.adapter_id}_{timestamp}_mask_thumb.png"
        thumb_path = os.path.join("tmp/", thumb_filename)
        thumb_pixmap.save(thumb_path)

        if self.ip_adapter.mask_image is not None:
            os.remove(self.ip_adapter.mask_image.mask_image.image_filename)
            os.remove(self.ip_adapter.mask_image.mask_image.image_thumb)

        background_image = ImageDataObject(
            image_original=bg_layer.original_path,
            image_scale=bg_layer.pixmap_item.scale(),
            image_x_pos=bg_layer.pixmap_item.x(),
            image_y_pos=bg_layer.pixmap_item.y(),
            image_rotation=bg_layer.pixmap_item.rotation(),
        )

        mask_image = ImageDataObject(
            image_filename=mask_path,
            image_original=mask_path,
            image_thumb=thumb_path,
            image_scale=mask_layer.pixmap_item.scale(),
            image_x_pos=mask_layer.pixmap_item.x(),
            image_y_pos=mask_layer.pixmap_item.y(),
            image_rotation=mask_layer.pixmap_item.rotation(),
        )

        ip_mask_image = MaskImage(background_image=background_image, mask_image=mask_image)
        self.ip_adapter.mask_image = ip_mask_image

        self.mask_saved.emit(thumb_pixmap)

    def on_cancel_mask(self):
        self.mask_canceled.emit()

    def update_ui(self, ip_adapter: IPAdapterDataObject):
        self.image_widget.reset_controls()
        self.ip_adapter = ip_adapter

        if self.ip_adapter.mask_image is not None:
            self.image_widget.reload_mask(self.ip_adapter.mask_image)
            self.save_mask_button.setText("Update Mask")
