import os
import shutil
import math
from datetime import datetime

from PIL import Image
from PyQt6.QtCore import QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPushButton, QWidget
from superqt import QDoubleSlider

from iartisanxl.modules.common.dialogs.base_dialog import BaseDialog
from iartisanxl.app.event_bus import EventBus
from iartisanxl.modules.common.ip_adapter.ip_adapter_image_widget import IPAdapterImageWidget
from iartisanxl.modules.common.ip_adapter.ip_adapter_data_object import IPAdapterDataObject
from iartisanxl.modules.common.ip_adapter.ip_adapter_image_items_view import IpAdapterImageItemsView
from iartisanxl.modules.common.image.image_adder_preview import ImageAdderPreview
from iartisanxl.modules.common.image.image_data_object import ImageDataObject


class ImageProcessThread(QThread):
    image_saved = pyqtSignal(ImageDataObject)

    def __init__(self, image_editor: ImageAdderPreview, image_data_object: ImageDataObject):
        super().__init__()

        self.image_editor = image_editor
        self.image_data_object = image_data_object

    def run(self):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        if self.image_data_object.id is None or self.image_data_object.replace_original:
            original_filename = f"ip_{timestamp}_original.png"
            original_path = os.path.join("tmp/", original_filename)

            if self.image_data_object.image_original is not None:
                shutil.copy2(self.image_data_object.image_original, original_path)
            else:
                pixmap = self.image_editor.pixmap_item.pixmap()
                pixmap.save(original_path)

            self.image_data_object.image_original = original_path
            self.image_data_object.replace_original = False

        pil_image = Image.open(self.image_data_object.image_original)
        width, height = pil_image.size

        dx = self.image_editor.pixmap_item.x()
        dy = self.image_editor.pixmap_item.y()
        angle = self.image_editor.pixmap_item.rotation()

        center = (width / 2, height / 2)
        pil_image = pil_image.rotate(-angle, Image.Resampling.BICUBIC, center=center, expand=True)

        new_width = round(self.image_editor.pixmap_item.sceneBoundingRect().width())
        new_height = round(self.image_editor.pixmap_item.sceneBoundingRect().height())
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

        left = math.floor(new_width / 2 - (width / 2 + dx))
        top = math.floor(new_height / 2 - (height / 2 + dy))
        right = round(self.image_editor.mapToScene(self.image_editor.viewport().rect()).boundingRect().width()) + left
        bottom = round(self.image_editor.mapToScene(self.image_editor.viewport().rect()).boundingRect().height()) + top
        pil_image = pil_image.crop((left, top, right, bottom))

        original_pixmap = self.image_editor.original_pixmap

        if self.image_data_object.id is None:
            original_filename = f"ip_{timestamp}_original.png"

            original_path = os.path.join("tmp/", original_filename)
            original_pixmap.save(original_path)
            self.image_data_object.image_original = original_path
        else:
            os.remove(self.image_data_object.image_filename)
            os.remove(self.image_data_object.image_thumb)

        filename = f"ip_{timestamp}.png"
        thumb_filename = f"ip_{timestamp}_thumb.png"

        image_path = os.path.join("tmp/", filename)
        thumb_path = os.path.join("tmp/", thumb_filename)

        pil_image.save(image_path)
        pil_image.thumbnail((80, 80))
        pil_image.save(thumb_path)

        self.image_data_object.image_filename = image_path
        self.image_data_object.image_thumb = thumb_path

        self.image_saved.emit(self.image_data_object)


class IPAdapterDialog(BaseDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle("IP Adapters")
        self.setMinimumSize(900, 700)

        self.settings = QSettings("ZCode", "ImageArtisanXL")
        self.settings.beginGroup("ip_adapters_dialog")
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        self.settings.endGroup()

        self.event_bus = EventBus()

        self.adapter = IPAdapterDataObject()
        self.adapter_scale = 1.0

        self.image_process_thread = None

        self.init_ui()

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
        all_images_layout.addWidget(self.image_items_view)
        self.images_view.setLayout(all_images_layout)
        middle_layout.addWidget(self.images_view)

        image_layout = QVBoxLayout()
        image_layout.setContentsMargins(5, 0, 5, 0)
        self.image_widget = IPAdapterImageWidget("image", self.image_viewer, self.directories.outputs_images)
        self.image_widget.image_added.connect(self.on_image_added)
        image_layout.addWidget(self.image_widget)

        middle_layout.addLayout(image_layout)

        middle_layout.setStretch(0, 2)
        middle_layout.setStretch(1, 3)

        self.main_layout.addLayout(middle_layout)

        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(5, 0, 5, 0)
        self.add_button = QPushButton("Add IP-Adapter")
        self.add_button.setObjectName("green_button")
        self.add_button.clicked.connect(self.on_ip_adapter_added)
        bottom_layout.addWidget(self.add_button)
        self.main_layout.addLayout(bottom_layout)

        self.main_layout.setStretch(0, 0)
        self.main_layout.setStretch(1, 1)
        self.main_layout.setStretch(2, 0)
        self.main_layout.setSpacing(3)

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

    def on_image_added(self):
        image_id = self.image_widget.image_id

        if image_id is None:
            image_data_object = ImageDataObject(
                image_scale=self.image_widget.image_scale_control.value,
                image_x_pos=self.image_widget.image_x_pos_control.value,
                image_y_pos=self.image_widget.image_y_pos_control.value,
                image_rotation=self.image_widget.image_rotation_control.value,
                weight=self.image_widget.image_weight_slider.value(),
                noise=self.image_widget.image_noise_slider.value(),
                image_original=self.image_widget.image_path,
            )
        else:
            image_data_object = self.adapter.get_image_data_object(image_id)

            if image_data_object is not None:
                image_data_object.image_scale = self.image_widget.image_scale_control.value
                image_data_object.image_x_pos = self.image_widget.image_x_pos_control.value
                image_data_object.image_y_pos = self.image_widget.image_y_pos_control.value
                image_data_object.image_rotation = self.image_widget.image_rotation_control.value
                image_data_object.weight = self.image_widget.image_weight_slider.value()
                image_data_object.noise = self.image_widget.image_noise_slider.value()

                if image_data_object.image_original != self.image_widget.image_path:
                    os.remove(image_data_object.image_original)
                    image_data_object.image_original = self.image_widget.image_path
                    image_data_object.replace_original = True
            else:
                self.show_error("Couldn't obtain the information to add the image.")
                return

        self.image_process_thread = ImageProcessThread(self.image_widget.image_editor, image_data_object)
        self.image_process_thread.image_saved.connect(self.on_image_saved)
        self.image_process_thread.finished.connect(self.on_image_process_finished)
        self.image_process_thread.start()

    def on_image_saved(self, image_data_object: ImageDataObject):
        if image_data_object.id is not None:
            self.image_items_view.update_current_item(image_data_object)
        else:
            self.adapter.add_image_data_object(image_data_object)
            image_item = self.image_items_view.add_item_data_object(image_data_object)
            self.image_items_view.set_current_item(image_item)

        self.image_widget.image_id = image_data_object.id
        self.image_widget.add_image_button.setText("Update image")
        self.image_widget.new_image_button.setEnabled(True)
        self.image_widget.delete_image_button.setEnabled(True)

    def on_image_process_finished(self):
        self.image_process_thread = None

    def on_item_selected(self, image_data: ImageDataObject):
        self.image_widget.show_image(image_data.image_original)
        self.image_widget.set_image_parameters(
            image_data.id,
            image_data.image_scale,
            image_data.image_x_pos,
            image_data.image_y_pos,
            image_data.image_rotation,
            image_data.weight,
            image_data.noise,
        )
        self.image_widget.add_image_button.setText("Update image")
        self.image_widget.add_image_button.setEnabled(False)
        self.image_widget.delete_image_button.setEnabled(True)
        self.image_widget.new_image_button.setEnabled(True)

    def on_item_deleted(self, image_data: ImageDataObject, clear_view: bool):
        if clear_view:
            self.image_widget.clear_image()

        self.adapter.delete_image(image_data.id)

    def reset_ui(self):
        self.adapter = IPAdapterDataObject()
        self.image_widget.clear_image()
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
