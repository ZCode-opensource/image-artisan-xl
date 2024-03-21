from PyQt6.QtCore import QEvent, QSettings, Qt
from PyQt6.QtGui import QColor, QCursor, QGuiApplication, QPixmap
from PyQt6.QtWidgets import QApplication, QComboBox, QHBoxLayout, QLabel, QSlider
from superqt import QDoubleSlider

from iartisanxl.app.event_bus import EventBus
from iartisanxl.buttons.brush_erase_button import BrushEraseButton
from iartisanxl.buttons.color_button import ColorButton
from iartisanxl.buttons.eyedropper_button import EyeDropperButton
from iartisanxl.modules.common.dialogs.base_dialog import BaseDialog
from iartisanxl.modules.common.ip_adapter.image_section_widget import ImageSectionWidget
from iartisanxl.modules.common.ip_adapter.ip_adapter_data_object import IPAdapterDataObject
from iartisanxl.modules.common.ip_adapter.mask_section_widget import MaskSectionWidget


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

        self.generation_width = self.image_generation_data.image_width
        self.generation_height = self.image_generation_data.image_height
        self.target_width = 512
        self.targe_height = 512
        self.adapter = IPAdapterDataObject()
        self.adapter_scale = 1.0
        self.image_changed = False

        self.mask_dialog = None

        self.init_ui()

    def init_ui(self):
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(10, 0, 10, 0)
        top_layout.setSpacing(10)

        self.type_combo = QComboBox()
        self.type_combo.addItem("IP Adapter", "ip_adapter_vit_h")
        self.type_combo.addItem("IP Adapter Plus", "ip_adapter_plus")
        self.type_combo.addItem("IP Adapter Plus Face", "ip_adapter_plus_face")
        self.type_combo.addItem("IP Adapter Composition", "ip_plus_composition_sdxl")
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
        self.brush_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_size_slider.setRange(3, 300)
        self.brush_size_slider.setValue(15)
        brush_layout.addWidget(self.brush_size_slider)

        brush_hardness_label = QLabel("Brush hardness:")
        brush_layout.addWidget(brush_hardness_label)
        self.brush_hardness_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        self.brush_hardness_slider.setRange(0.0, 0.99)
        self.brush_hardness_slider.setValue(0.5)
        brush_layout.addWidget(self.brush_hardness_slider)

        self.brush_erase_button = BrushEraseButton()
        brush_layout.addWidget(self.brush_erase_button)

        self.color_button = ColorButton("Color:")
        brush_layout.addWidget(self.color_button, 0)

        eyedropper_button = EyeDropperButton(25, 25)
        eyedropper_button.clicked.connect(self.on_eyedropper_clicked)
        brush_layout.addWidget(eyedropper_button, 0)

        self.main_layout.addLayout(brush_layout)

        self.image_section_widget = ImageSectionWidget(
            self.adapter, self.image_viewer, self.directories.outputs_images, self.target_width, self.targe_height
        )
        self.image_section_widget.error.connect(self.on_error)
        self.image_section_widget.add_mask.connect(self.on_add_mask_clicked)
        self.image_section_widget.add_ip_adapter.connect(self.on_ip_adapter_added)

        self.main_layout.addWidget(self.image_section_widget)

        self.mask_section_widget = MaskSectionWidget(
            self.adapter, self.image_viewer, self.generation_width, self.generation_height
        )
        self.mask_section_widget.mask_saved.connect(self.on_mask_saved)
        self.mask_section_widget.mask_canceled.connect(self.on_cancel_mask)
        self.mask_section_widget.setVisible(False)
        self.main_layout.addWidget(self.mask_section_widget)

        self.main_layout.setStretch(0, 0)
        self.main_layout.setStretch(1, 0)
        self.main_layout.setStretch(2, 1)
        self.main_layout.setStretch(3, 1)

        self.connect_image_editor()

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
                self.image_section_widget.add_button.setText("Update IP-Adapter")
            else:
                self.event_bus.publish("ip_adapters", {"action": "update", "ip_adapter": self.adapter})
        else:
            self.show_error("Adapter must have at least one image.")

    def on_error(self, message: str):
        self.error = True
        self.show_error(message)

    def reset_ui(self):
        self.image_section_widget.reset_ui()
        self.adapter = IPAdapterDataObject()

    def update_ui(self):
        self.image_section_widget.update_ui(self.adapter)
        self.mask_section_widget.update_ui(self.adapter)
        self.type_combo.setCurrentIndex(self.adapter.type_index)
        self.on_adapter_scale_changed(self.adapter.ip_adapter_scale)
        self.adapter_scale_slider.setValue(self.adapter_scale)

    def make_new_adapter(self):
        self.reset_ui()
        self.adapter = IPAdapterDataObject()
        self.image_section_widget.image_items_view.ip_adapter_data = self.adapter
        self.on_adapter_scale_changed(1.0)
        self.adapter_scale_slider.setValue(1.0)

    def connect_editor(self, section_widget, editor):
        self.color_button.color_changed.connect(editor.set_brush_color)
        self.brush_size_slider.valueChanged.connect(editor.set_brush_size)
        self.brush_size_slider.sliderReleased.connect(editor.hide_brush_preview)
        self.brush_hardness_slider.valueChanged.connect(editor.set_brush_hardness)
        self.brush_hardness_slider.sliderReleased.connect(editor.hide_brush_preview)
        self.brush_erase_button.brush_selected.connect(section_widget.image_widget.set_erase_mode)

    def connect_image_editor(self):
        self.connect_editor(self.image_section_widget, self.image_section_widget.image_widget.image_editor)

    def connect_mask_editor(self):
        self.connect_editor(self.mask_section_widget, self.mask_section_widget.image_widget.image_editor)

    def on_add_mask_clicked(self):
        self.connect_mask_editor()
        self.image_section_widget.hide()
        self.mask_section_widget.show()

    def on_mask_saved(self, thumb_pixmap: QPixmap):
        self.connect_image_editor()
        self.image_section_widget.ip_mask_item.set_pixmap(thumb_pixmap)
        self.image_section_widget.show()
        self.mask_section_widget.hide()
        self.image_section_widget.add_mask_button.setText("Edit mask")

    def on_cancel_mask(self):
        self.connect_image_editor()
        self.image_section_widget.show()
        self.mask_section_widget.hide()

        if self.adapter.mask_image is not None and self.adapter.mask_image.mask_image.image_filename:
            self.image_section_widget.add_mask_button.setText("Edit mask")

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
