from PyQt6.QtCore import QEvent, QSettings, Qt
from PyQt6.QtGui import QColor, QCursor, QGuiApplication
from PyQt6.QtWidgets import QApplication, QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget
from superqt import QDoubleSlider

from iartisanxl.app.event_bus import EventBus
from iartisanxl.buttons.brush_erase_button import BrushEraseButton
from iartisanxl.buttons.color_button import ColorButton
from iartisanxl.buttons.eyedropper_button import EyeDropperButton
from iartisanxl.modules.common.dialogs.base_dialog import BaseDialog
from iartisanxl.modules.common.ip_adapter.ip_adapter_data_object import IPAdapterDataObject
from iartisanxl.modules.common.ip_adapter.ip_adapter_image import IPAdapterImage
from iartisanxl.modules.common.ip_adapter.ip_adapter_image_items_view import IpAdapterImageItemsView
from iartisanxl.modules.common.ip_adapter.ip_adapter_image_widget import IPAdapterImageWidget
from iartisanxl.modules.common.mask.mask_dialog import MaskDialog
from iartisanxl.threads.image.transformed_images_saver_thread import TransformedImagesSaverThread
from iartisanxl.utilities.image.operations import remove_image_data_files


class IPAdapterDialog(BaseDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle("IP Adapters")
        self.setMinimumSize(500, 500)

        self.settings = QSettings("ZCode", "ImageArtisanXL")
        self.settings.beginGroup("ip_adapters_dialog")
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        self.settings.endGroup()

        self.event_bus = EventBus()

        self.target_width = 512
        self.targe_height = 512
        self.adapter = IPAdapterDataObject()
        self.adapter_scale = 1.0
        self.image_changed = False

        self.image_prepare_thread = None

        self.mask_dialog = None

        self.init_ui()

        self.image_widget.image_editor.brush_size = 15
        self.image_widget.add_empty_layer()

    def init_ui(self):
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(10, 0, 10, 0)
        top_layout.setSpacing(10)

        self.type_combo = QComboBox()
        self.type_combo.addItem("IP Adapter", "ip_adapter_vit_h")
        self.type_combo.addItem("IP Adapter Plus", "ip_adapter_plus")
        self.type_combo.addItem("IP Adapter Plus Face", "ip_adapter_plus_face")
        top_layout.addWidget(self.type_combo)

        adapter_scale_label = QLabel("Adapter scale:")
        top_layout.addWidget(adapter_scale_label)
        self.adapter_scale_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        self.adapter_scale_slider.setRange(0.0, 1.0)
        self.adapter_scale_slider.setValue(self.adapter_scale)
        self.adapter_scale_slider.valueChanged.connect(self.on_adapter_scale_changed)
        top_layout.addWidget(self.adapter_scale_slider)
        self.adapter_scale_value_label = QLabel(f"{self.adapter_scale}")
        top_layout.addWidget(self.adapter_scale_value_label)

        self.main_layout.addLayout(top_layout)

        brush_layout = QHBoxLayout()
        brush_layout.setContentsMargins(10, 0, 10, 0)
        brush_layout.setSpacing(10)

        brush_size_label = QLabel("Brush size:")
        brush_layout.addWidget(brush_size_label)
        brush_size_slider = QSlider(Qt.Orientation.Horizontal)
        brush_size_slider.setRange(3, 300)
        brush_size_slider.setValue(15)
        brush_layout.addWidget(brush_size_slider)

        brush_hardness_label = QLabel("Brush hardness:")
        brush_layout.addWidget(brush_hardness_label)
        brush_hardness_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        brush_hardness_slider.setRange(0.0, 0.99)
        brush_hardness_slider.setValue(0.5)
        brush_layout.addWidget(brush_hardness_slider)

        brush_erase_button = BrushEraseButton()
        brush_layout.addWidget(brush_erase_button)

        self.color_button = ColorButton("Color:")
        brush_layout.addWidget(self.color_button, 0)

        eyedropper_button = EyeDropperButton(25, 25)
        eyedropper_button.clicked.connect(self.on_eyedropper_clicked)
        brush_layout.addWidget(eyedropper_button, 0)

        self.main_layout.addLayout(brush_layout)

        middle_layout = QHBoxLayout()

        self.images_view = QWidget()
        all_images_layout = QVBoxLayout()
        all_images_layout.setContentsMargins(0, 0, 0, 0)
        all_images_layout.setSpacing(0)
        self.dataset_items_count_label = QLabel("0/0")
        all_images_layout.addWidget(self.dataset_items_count_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.image_items_view = IpAdapterImageItemsView(self.adapter)
        self.image_items_view.item_selected.connect(self.on_item_selected)
        self.image_items_view.item_deleted.connect(self.on_item_deleted)
        self.image_items_view.finished_loading.connect(self.on_images_finished_loading)
        all_images_layout.addWidget(self.image_items_view)
        self.images_view.setLayout(all_images_layout)
        middle_layout.addWidget(self.images_view)

        image_layout = QVBoxLayout()
        image_layout.setContentsMargins(5, 0, 5, 0)
        self.image_widget = IPAdapterImageWidget(
            "image", self.image_viewer, self.directories.outputs_images, self.target_width, self.targe_height
        )
        self.image_widget.image_added.connect(self.on_image_added)
        self.image_widget.new_image.connect(self.on_new_image)
        image_layout.addWidget(self.image_widget)

        middle_layout.addLayout(image_layout)

        middle_layout.setStretch(0, 2)
        middle_layout.setStretch(1, 3)

        self.main_layout.addLayout(middle_layout)

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
        self.main_layout.addLayout(bottom_layout)

        self.main_layout.setStretch(0, 0)
        self.main_layout.setStretch(1, 1)
        self.main_layout.setStretch(2, 0)
        self.main_layout.setSpacing(3)

        self.color_button.color_changed.connect(self.image_widget.image_editor.set_brush_color)
        brush_size_slider.valueChanged.connect(self.image_widget.image_editor.set_brush_size)
        brush_size_slider.sliderReleased.connect(self.image_widget.image_editor.hide_brush_preview)
        brush_hardness_slider.valueChanged.connect(self.image_widget.image_editor.set_brush_hardness)
        brush_hardness_slider.sliderReleased.connect(self.image_widget.image_editor.hide_brush_preview)

        brush_erase_button.brush_selected.connect(self.image_widget.set_erase_mode)

    def closeEvent(self, event):
        self.settings.beginGroup("ip_adapters_dialog")
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.endGroup()

        super().closeEvent(event)

    def on_adapter_scale_changed(self, value):
        self.adapter_scale = value
        self.adapter_scale_value_label.setText(f"{value:.2f}")

    def on_ip_adapter_added(self):
        if len(self.adapter.images) > 0:
            self.adapter.adapter_name = self.type_combo.currentText()
            self.adapter.adapter_type = self.type_combo.currentData()
            self.adapter.type_index = self.type_combo.currentIndex()
            self.adapter.ip_adapter_scale = self.adapter_scale

            if self.adapter.adapter_id is None:
                self.event_bus.publish("ip_adapters", {"action": "add", "ip_adapter": self.adapter})
                self.add_button.setText("Update IP-Adapter")
            else:
                self.event_bus.publish("ip_adapters", {"action": "update", "ip_adapter": self.adapter})
        else:
            self.show_error("Adapter must have at least one image.")

    def on_new_image(self):
        if self.image_items_view.current_item is not None:
            self.image_items_view.current_item.set_selected(False)

        self.image_items_view.current_item = None

    def on_image_added(self):
        layers = self.image_widget.image_editor.get_all_layers()

        if len(layers) == 0:
            self.show_error("You must load an image or create a new one to be able to add it.")
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

    def on_error(self, message: str):
        self.error = True
        self.show_error(message)

    def on_image_prepare_thread_finished(self):
        self.image_prepare_thread.finished.disconnect(self.on_image_prepare_thread_finished)
        self.image_prepare_thread = None

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
            ip_adapter_image.add_image(
                layer.original_path,
                layer.image_path,
                layer.pixmap_item.scale(),
                layer.pixmap_item.x(),
                layer.pixmap_item.y(),
                layer.pixmap_item.rotation(),
                layer_name,
                layer.order,
            )

        if self.image_items_view.current_item is None:
            self.adapter.add_ip_adapter_image(ip_adapter_image)
            image_item = self.image_items_view.add_item_data_object(ip_adapter_image)
            self.image_items_view.on_item_selected(image_item)
        else:
            ip_adapter_image.ip_adapter_id = self.image_items_view.current_item.ip_adapter_image.ip_adapter_id
            self.adapter.update_ip_adapter_image(ip_adapter_image)
            self.image_items_view.update_current_item(ip_adapter_image)
            self.on_item_selected(ip_adapter_image)

        self.image_widget.add_image_button.setText("Update image")
        self.image_widget.new_image_button.setEnabled(True)
        self.image_widget.delete_image_button.setEnabled(True)

    def on_item_selected(self, ip_adapter_Image: IPAdapterImage):
        self.image_widget.image_editor.clear_all()
        self.image_widget.reset_controls()
        self.image_widget.layer_manager_widget.list_widget.clear()

        self.image_widget.image_weight_slider.setValue(ip_adapter_Image.weight)
        self.image_widget.image_noise_slider.setValue(ip_adapter_Image.noise)
        self.image_widget.noise_type_combo.setCurrentIndex(ip_adapter_Image.noise_type_index)

        for image in sorted(ip_adapter_Image.images, key=lambda img: img.order):
            layer_id = self.image_widget.reload_image_layer(image.image_filename, image.image_original, image.order)
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
            self.image_widget.clear_image()

        # delete images
        for image_data in ip_adapter_image.images:
            remove_image_data_files(image_data)

        self.adapter.delete_ip_adapter_image(ip_adapter_image.ip_adapter_id)

    def reset_ui(self):
        self.image_widget.clear_image()

        self.adapter = IPAdapterDataObject()
        self.image_items_view.clear()
        self.add_button.setText("Add IP-Adapter")

    def update_ui(self):
        self.image_widget.clear_image()
        self.image_items_view.clear()

        self.image_items_view.ip_adapter_data = self.adapter
        self.image_items_view.load_items()

        self.type_combo.setCurrentIndex(self.adapter.type_index)
        self.on_adapter_scale_changed(self.adapter.ip_adapter_scale)
        self.adapter_scale_slider.setValue(self.adapter_scale)
        self.add_button.setText("Update IP-Adapter")

    def make_new_adapter(self):
        self.reset_ui()
        self.adapter = IPAdapterDataObject()
        self.image_items_view.ip_adapter_data = self.adapter
        self.on_adapter_scale_changed(1.0)
        self.adapter_scale_slider.setValue(1.0)

    def on_images_finished_loading(self):
        self.image_widget.new_image_button.setEnabled(True)
        self.on_item_selected(self.image_items_view.current_item.ip_adapter_image)
        self.dataset_items_count_label.setText(
            f"{self.image_items_view.current_item_index + 1}/{self.image_items_view.item_count}"
        )

    def on_add_mask_clicked(self):
        # if self.adapter.adapter_id is None:
        #     self.show_error("To add a mask, you first need to add the IP Adapter.")
        #     return

        if self.mask_dialog is None:
            self.mask_dialog = MaskDialog(
                self.adapter,
                self.directories,
                self.preferences,
                "IP Adapter - Mask Editor",
                self.show_error,
                self.image_generation_data,
                self.image_viewer,
                self.prompt_window,
            )
            self.mask_dialog.mask_saved.connect(self.on_mask_saved)
            self.mask_dialog.closed.connect(self.on_mask_dialog_closed)
            self.mask_dialog.show()
        else:
            self.mask_dialog.raise_()
            self.mask_dialog.activateWindow()

    def on_mask_saved(self):
        self.mask_dialog.close()
        self.add_mask_button.setText("Edit mask")

    def on_mask_dialog_closed(self):
        self.mask_dialog = None

    def on_eyedropper_clicked(self):
        QApplication.instance().setOverrideCursor(Qt.CursorShape.CrossCursor)
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event):
        if (
            QApplication.instance().overrideCursor() == Qt.CursorShape.CrossCursor
            and event.type() == QEvent.Type.MouseButtonPress
        ):
            QApplication.instance().restoreOverrideCursor()
            QApplication.instance().removeEventFilter(self)
            x, y = QCursor.pos().x(), QCursor.pos().y()
            pixmap = QGuiApplication.primaryScreen().grabWindow(0, x, y, 1, 1)
            color = QColor(pixmap.toImage().pixel(0, 0))
            rgb_color = (color.red(), color.green(), color.blue())
            self.color_button.set_color(rgb_color)
            return True
        return super().eventFilter(obj, event)
