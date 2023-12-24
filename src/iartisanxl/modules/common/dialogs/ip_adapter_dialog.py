from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPushButton
from superqt import QDoubleSlider

from iartisanxl.modules.common.dialogs.base_dialog import BaseDialog
from iartisanxl.app.event_bus import EventBus
from iartisanxl.modules.common.dialogs.ip_adapter_image_widget import IPAdapterImageWidget
from iartisanxl.generation.ip_adapter_data_object import IPAdapterDataObject
from iartisanxl.formats.image import ImageProcessor


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

        self.adapter = None
        self.adapter_scale = 1.0

        self.init_ui()

    def init_ui(self):
        content_layout = QVBoxLayout()

        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(10, 0, 10, 0)
        control_layout.setSpacing(10)

        self.type_combo = QComboBox()
        self.type_combo.addItem("IP Adapter", "ip_adapter_vit_h")
        self.type_combo.addItem("IP¨Adapter Plus", "ip_adapter_plus")
        self.type_combo.addItem("IP¨Adapter Plus Face", "ip_adapter_plus_face")
        self.type_combo.addItem("IP Adapter ViT-bigG", "ip_adapter")
        self.type_combo.currentIndexChanged.connect(self.adapter_changed)
        control_layout.addWidget(self.type_combo)

        adapter_scale_label = QLabel("Adapter scale:")
        control_layout.addWidget(adapter_scale_label)
        self.adapter_scale_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        self.adapter_scale_slider.setRange(0.0, 1.0)
        self.adapter_scale_slider.setValue(self.adapter_scale)
        self.adapter_scale_slider.valueChanged.connect(self.on_adapter_scale_changed)
        control_layout.addWidget(self.adapter_scale_slider)
        self.adapter_scale_value_label = QLabel(f"{self.adapter_scale}")
        control_layout.addWidget(self.adapter_scale_value_label)

        content_layout.addLayout(control_layout)

        images_layout = QHBoxLayout()
        images_layout.setContentsMargins(2, 0, 4, 0)
        images_layout.setSpacing(2)

        image_layout = QVBoxLayout()
        self.image_widget = IPAdapterImageWidget("image", self.image_viewer, self.image_generation_data)
        image_layout.addWidget(self.image_widget)
        images_layout.addLayout(image_layout)

        content_layout.addLayout(images_layout)

        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.on_ip_adapter_added)
        content_layout.addWidget(self.add_button)

        content_layout.setStretch(0, 0)
        content_layout.setStretch(1, 1)
        content_layout.setStretch(2, 0)

        self.main_layout.addLayout(content_layout)

    def closeEvent(self, event):
        self.settings.beginGroup("ip_adapters_dialog")
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.endGroup()

        super().closeEvent(event)

    def adapter_changed(self):
        pass

    def on_adapter_scale_changed(self, value):
        self.adapter_scale = value
        self.adapter_scale_value_label.setText(f"{value:.2f}")

    def on_ip_adapter_added(self):
        if self.adapter is None:
            self.adapter = IPAdapterDataObject(
                adapter_type=self.type_combo.currentData(),
                enabled=True,
                ip_adapter_scale=self.adapter_scale,
                type_index=self.type_combo.currentIndex(),
            )
        else:
            self.adapter.adapter_type = self.type_combo.currentData()
            self.adapter.ip_adapter_scale = self.adapter_scale

        image = ImageProcessor()
        qimage = self.image_widget.image_editor.get_painted_image()
        image.set_qimage(qimage)
        self.adapter.image_thumb = image.get_pillow_thumbnail(target_height=80)
        self.adapter.image = image.get_pillow_image()

        if self.adapter.adapter_id is None:
            self.event_bus.publish("ip_adapters", {"action": "add", "ip_adapter": self.adapter})
            self.add_button.setText("Update")
        else:
            self.event_bus.publish("ip_adapters", {"action": "update", "ip_adapter": self.adapter})
