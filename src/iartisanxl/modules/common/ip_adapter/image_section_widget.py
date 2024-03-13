from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from iartisanxl.modules.common.image_viewer_simple import ImageViewerSimple
from iartisanxl.modules.common.ip_adapter.ip_adapter_data_object import IPAdapterDataObject
from iartisanxl.modules.common.ip_adapter.ip_adapter_image import IPAdapterImage
from iartisanxl.modules.common.ip_adapter.ip_adapter_image_items_view import IpAdapterImageItemsView
from iartisanxl.modules.common.ip_adapter.ip_adapter_image_widget import IPAdapterImageWidget
from iartisanxl.modules.common.ip_adapter.ip_mask_item import IpMaskItem
from iartisanxl.threads.image.transformed_images_saver_thread import TransformedImagesSaverThread


class ImageSectionWidget(QWidget):
    add_mask = pyqtSignal()
    add_ip_adapter = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        ip_adapter: IPAdapterDataObject,
        image_viewer: ImageViewerSimple,
        outputh_path: str,
        target_width: int,
        targe_height: int,
    ):
        super().__init__()

        self.ip_adapter = ip_adapter
        self.image_viewer = image_viewer
        self.outputh_path = outputh_path
        self.target_width = target_width
        self.targe_height = targe_height

        self.image_prepare_thread = None

        self.init_ui()

        self.image_widget.image_editor.brush_size = 15
        self.image_widget.add_empty_layer()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(1, 3, 0, 0)
        main_layout.setSpacing(0)

        middle_layout = QHBoxLayout()

        self.images_view = QWidget()
        all_images_layout = QVBoxLayout()
        all_images_layout.setContentsMargins(0, 0, 0, 0)
        all_images_layout.setSpacing(0)
        self.dataset_items_count_label = QLabel("0/0")
        all_images_layout.addWidget(self.dataset_items_count_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.image_items_view = IpAdapterImageItemsView(self.ip_adapter)
        self.image_items_view.item_selected.connect(self.on_item_selected)
        self.image_items_view.item_deleted.connect(self.on_item_deleted)
        self.image_items_view.finished_loading.connect(self.on_images_finished_loading)
        all_images_layout.addWidget(self.image_items_view)
        self.ip_mask_item = IpMaskItem()
        all_images_layout.addWidget(self.ip_mask_item)
        self.images_view.setLayout(all_images_layout)
        middle_layout.addWidget(self.images_view)

        image_layout = QVBoxLayout()
        image_layout.setContentsMargins(5, 0, 5, 0)
        self.image_widget = IPAdapterImageWidget(
            "image", self.image_viewer, self.outputh_path, self.target_width, self.targe_height
        )
        self.image_widget.image_added.connect(self.on_image_added)
        self.image_widget.new_image.connect(self.on_new_image)
        image_layout.addWidget(self.image_widget)

        middle_layout.addLayout(image_layout)

        middle_layout.setStretch(0, 2)
        middle_layout.setStretch(1, 3)

        main_layout.addLayout(middle_layout)

        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(5, 0, 5, 0)
        self.add_mask_button = QPushButton("Add Mask")
        self.add_mask_button.clicked.connect(self.on_add_mask_clicked)
        self.add_mask_button.setObjectName("blue_button")
        bottom_layout.addWidget(self.add_mask_button)
        self.add_button = QPushButton("Add IP-Adapter")
        self.add_button.setObjectName("green_button")
        self.add_button.clicked.connect(self.on_ip_adapter_added)
        bottom_layout.addWidget(self.add_button)
        main_layout.addLayout(bottom_layout)

        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 0)
        main_layout.setStretch(2, 0)
        main_layout.setSpacing(3)

        self.setLayout(main_layout)

    def on_image_added(self):
        layers = self.image_widget.image_editor.get_all_layers()

        if len(layers) == 0:
            self.error.emit("You must load an image or create a new one to be able to add it.")
            return

        self.image_prepare_thread = TransformedImagesSaverThread(
            layers,
            self.target_width,
            self.targe_height,
            prefix="ip_adapter_",
        )

        self.image_prepare_thread.merge_finished.connect(self.on_image_prepared)
        self.image_prepare_thread.finished.connect(self.on_image_prepare_thread_finished)
        self.image_prepare_thread.error.connect(self.on_error)
        self.image_prepare_thread.start()

    def on_image_prepared(self, image_path: str, thumbnail_path: str):
        ip_adapter_image = IPAdapterImage(
            image=image_path,
            thumb=thumbnail_path,
            weight=self.image_widget.image_weight_slider.value(),
            noise=self.image_widget.image_noise_slider.value(),
            noise_type=self.image_widget.noise_type_combo.currentData(),
            noise_type_index=self.image_widget.noise_type_combo.currentIndex(),
        )

        for layer in self.image_widget.image_editor.get_all_layers():
            layer_name = self.image_widget.layer_manager_widget.get_layer_name(layer.layer_id)
            ip_adapter_image.add_image(layer.original_path, layer.image_path, 1.0, 0, 0, 0.0, layer_name, layer.order)

        if self.image_items_view.current_item is None:
            self.ip_adapter.add_ip_adapter_image(ip_adapter_image)
            image_item = self.image_items_view.add_item_data_object(ip_adapter_image)
            self.image_items_view.on_item_selected(image_item)
        else:
            ip_adapter_image.ip_adapter_id = self.image_items_view.current_item.ip_adapter_image.ip_adapter_id
            self.ip_adapter.update_ip_adapter_image(ip_adapter_image)
            self.image_items_view.update_current_item(ip_adapter_image)
            self.on_item_selected(ip_adapter_image)

        self.image_widget.add_image_button.setText("Update image")
        self.image_widget.new_image_button.setEnabled(True)
        self.image_widget.delete_image_button.setEnabled(True)

    def on_image_prepare_thread_finished(self):
        self.image_prepare_thread.finished.disconnect(self.on_image_prepare_thread_finished)
        self.image_prepare_thread = None

    def on_new_image(self):
        if self.image_items_view.current_item is not None:
            self.image_items_view.current_item.set_selected(False)

        self.image_items_view.current_item = None

    def on_add_mask_clicked(self):
        self.add_mask.emit()

    def on_ip_adapter_added(self):
        self.add_ip_adapter.emit()

    def on_error(self, message: str):
        self.error.emit(message)

    def on_item_selected(self, ip_adapter_Image: IPAdapterImage):
        self.image_widget.image_editor.clear_all()
        self.image_widget.layer_manager_widget.list_widget.clear()

        self.image_widget.image_weight_slider.setValue(ip_adapter_Image.weight)
        self.image_widget.image_noise_slider.setValue(ip_adapter_Image.noise)
        self.image_widget.noise_type_combo.setCurrentIndex(ip_adapter_Image.noise_type_index)

        for image in sorted(ip_adapter_Image.images, key=lambda img: img.order):
            layer_id = self.image_widget.reload_image_layer(image.image_filename, image.image_original, image.order)
            self.image_widget.set_layer_parameters(
                layer_id, image.image_scale, image.image_x_pos, image.image_y_pos, image.image_rotation
            )
            self.image_widget.image_editor.selected_layer_id = layer_id
            self.image_widget.layer_manager_widget.add_layer(layer_id, image.layer_name)

        self.image_widget.add_image_button.setText("Update image")
        self.image_widget.add_image_button.setEnabled(False)
        self.image_widget.delete_image_button.setEnabled(True)
        self.image_widget.new_image_button.setEnabled(True)

        self.dataset_items_count_label.setText(
            f"{self.image_items_view.current_item_index + 1}/{self.image_items_view.item_count}"
        )

    def on_item_deleted(self, ip_adapter_image: IPAdapterImage, clear_view: bool):
        if clear_view:
            self.image_widget.on_new_image()

        self.ip_adapter.delete_ip_adapter_image(ip_adapter_image.ip_adapter_id)

    def on_images_finished_loading(self):
        self.image_widget.new_image_button.setEnabled(True)
        self.on_item_selected(self.image_items_view.current_item.ip_adapter_image)
        self.dataset_items_count_label.setText(
            f"{self.image_items_view.current_item_index + 1}/{self.image_items_view.item_count}"
        )

    def reset_ui(self):
        self.image_widget.clear_image()
        self.image_items_view.clear()
        self.add_button.setText("Add IP-Adapter")

    def update_ui(self, ip_adapter: IPAdapterDataObject):
        self.image_widget.clear_image()
        self.image_items_view.clear()

        self.ip_adapter = ip_adapter

        if ip_adapter.mask_image is not None:
            self.ip_mask_item.set_thumb_image(ip_adapter.mask_image.mask_image.image_thumb)

        self.image_items_view.ip_adapter_data = self.ip_adapter
        self.image_items_view.load_items()
        self.add_button.setText("Update IP-Adapter")
