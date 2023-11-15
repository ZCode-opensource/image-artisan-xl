from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout
from PyQt6.QtCore import QSettings, Qt
from superqt import QLabeledDoubleRangeSlider, QLabeledSlider, QLabeledDoubleSlider

from iartisanxl.modules.common.dialogs.base_dialog import BaseDialog
from iartisanxl.modules.common.dialogs.control_image_widget import ControlImageWidget


class ControlNetDialog(BaseDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle("ControlNet")
        self.setMinimumSize(1160, 800)

        self.settings = QSettings("ZCode", "ImageArtisanXL")
        self.settings.beginGroup("controlnet_dialog")
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        self.settings.endGroup()

        self.control_guidance_start = 0.0
        self.control_guidance_end = 1.0

        self.init_ui()

    def init_ui(self):
        content_layout = QVBoxLayout()

        top_layout = QHBoxLayout()

        slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 1)
        slider.setValue((self.control_guidance_start, self.control_guidance_end))
        slider.valueChanged.connect(self.on_guidance_changed)
        top_layout.addWidget(slider)

        brush_size_slider = QLabeledSlider()
        brush_size_slider.setRange(1, 100)
        brush_size_slider.setValue(20)
        top_layout.addWidget(brush_size_slider)

        brush_hardness_slider = QLabeledDoubleSlider()
        brush_hardness_slider.setRange(0.0, 0.99)
        brush_hardness_slider.setValue(0.5)
        top_layout.addWidget(brush_hardness_slider)

        content_layout.addLayout(top_layout)

        images_layout = QHBoxLayout()
        images_layout.setContentsMargins(2, 0, 4, 0)
        images_layout.setSpacing(2)
        source_widget = ControlImageWidget(
            "Source image", self.image_viewer, self.image_generation_data
        )
        images_layout.addWidget(source_widget)

        annotator_widget = ControlImageWidget(
            "Annotator", self.image_viewer, self.image_generation_data
        )
        images_layout.addWidget(annotator_widget)

        content_layout.addLayout(images_layout)

        content_layout.setStretch(0, 2)
        content_layout.setStretch(1, 6)

        self.main_layout.addLayout(content_layout)

        brush_size_slider.valueChanged.connect(
            source_widget.image_editor.set_brush_size
        )
        brush_hardness_slider.valueChanged.connect(
            source_widget.image_editor.set_brush_hardness
        )
        brush_size_slider.valueChanged.connect(
            annotator_widget.image_editor.set_brush_size
        )
        brush_hardness_slider.valueChanged.connect(
            annotator_widget.image_editor.set_brush_hardness
        )

    def closeEvent(self, event):
        self.settings.beginGroup("controlnet_dialog")
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.endGroup()
        super().closeEvent(event)

    def on_guidance_changed(self, values):
        print(f"{values=}")
