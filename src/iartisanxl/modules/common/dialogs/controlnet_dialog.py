import os
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from PyQt6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QComboBox,
)
from PyQt6.QtCore import QSettings, Qt, QBuffer, QIODevice
from PyQt6.QtGui import QImage, QPixmap
from superqt import QDoubleRangeSlider, QDoubleSlider

from iartisanxl.buttons.color_button import ColorButton
from iartisanxl.modules.common.dialogs.base_dialog import BaseDialog
from iartisanxl.modules.common.dialogs.control_image_widget import ControlImageWidget
from iartisanxl.generation.controlnet_data_object import ControlNetDataObject


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

        self.conditioning_scale = 0.50
        self.control_guidance_start = 0.0
        self.control_guidance_end = 1.0

        self.init_ui()

    def init_ui(self):
        content_layout = QVBoxLayout()

        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(10, 0, 10, 0)
        control_layout.setSpacing(10)

        self.controlnet_combo = QComboBox()
        self.controlnet_combo.addItem("Canny", "canny")
        self.controlnet_combo.addItem("Depth", "depth")
        self.controlnet_combo.addItem("Pose", "pose")
        control_layout.addWidget(self.controlnet_combo)

        conditioning_scale_label = QLabel("Conditioning scale:")
        control_layout.addWidget(conditioning_scale_label)
        conditioning_scale_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        conditioning_scale_slider.setRange(0.0, 2.0)
        conditioning_scale_slider.setValue(self.conditioning_scale)
        conditioning_scale_slider.valueChanged.connect(
            self.on_conditional_scale_changed
        )
        control_layout.addWidget(conditioning_scale_slider)
        self.conditioning_scale_value_label = QLabel(f"{self.conditioning_scale}")
        control_layout.addWidget(self.conditioning_scale_value_label)

        guidance_start_label = QLabel("Guidance Start:")
        control_layout.addWidget(guidance_start_label)
        self.guidance_start_value_label = QLabel(
            f"{int(self.control_guidance_start * 100)}%"
        )
        control_layout.addWidget(self.guidance_start_value_label)
        slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 1)
        slider.setValue((self.control_guidance_start, self.control_guidance_end))
        slider.valueChanged.connect(self.on_guidance_changed)
        control_layout.addWidget(slider)
        guidance_end_label = QLabel("End:")
        control_layout.addWidget(guidance_end_label)
        self.guidance_end_value_label = QLabel(
            f"{int(self.control_guidance_end * 100)}%"
        )
        control_layout.addWidget(self.guidance_end_value_label)

        annotate_button = QPushButton("Annotate")
        annotate_button.clicked.connect(self.on_annotate)
        control_layout.addWidget(annotate_button)
        add_button = QPushButton("Add")
        add_button.clicked.connect(self.on_controlnet_added)
        control_layout.addWidget(add_button)

        content_layout.addLayout(control_layout)

        brush_layout = QHBoxLayout()
        brush_layout.setContentsMargins(10, 0, 10, 0)
        brush_layout.setSpacing(10)

        brush_size_label = QLabel("Brush size:")
        brush_layout.addWidget(brush_size_label)
        brush_size_slider = QSlider(Qt.Orientation.Horizontal)
        brush_size_slider.setRange(1, 300)
        brush_size_slider.setValue(20)
        brush_layout.addWidget(brush_size_slider)

        brush_hardness_label = QLabel("Brush hardness:")
        brush_layout.addWidget(brush_hardness_label)
        brush_hardness_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        brush_hardness_slider.setRange(0.0, 0.99)
        brush_hardness_slider.setValue(0.5)
        brush_layout.addWidget(brush_hardness_slider)

        color_button = ColorButton("Brush color:")
        brush_layout.addWidget(color_button)

        content_layout.addLayout(brush_layout)

        images_layout = QHBoxLayout()
        images_layout.setContentsMargins(2, 0, 4, 0)
        images_layout.setSpacing(2)
        self.source_widget = ControlImageWidget(
            "Source image", self.image_viewer, self.image_generation_data
        )
        images_layout.addWidget(self.source_widget)

        self.annotator_widget = ControlImageWidget(
            "Annotator", self.image_viewer, self.image_generation_data
        )
        images_layout.addWidget(self.annotator_widget)

        content_layout.addLayout(images_layout)

        content_layout.setStretch(0, 0)
        content_layout.setStretch(1, 0)
        content_layout.setStretch(2, 1)

        self.main_layout.addLayout(content_layout)

        color_button.color_changed.connect(
            self.source_widget.image_editor.set_brush_color
        )
        brush_size_slider.valueChanged.connect(
            self.source_widget.image_editor.set_brush_size
        )
        brush_hardness_slider.valueChanged.connect(
            self.source_widget.image_editor.set_brush_hardness
        )

        color_button.color_changed.connect(
            self.annotator_widget.image_editor.set_brush_color
        )
        brush_size_slider.valueChanged.connect(
            self.annotator_widget.image_editor.set_brush_size
        )
        brush_hardness_slider.valueChanged.connect(
            self.annotator_widget.image_editor.set_brush_hardness
        )

    def closeEvent(self, event):
        self.settings.beginGroup("controlnet_dialog")
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.endGroup()
        super().closeEvent(event)

    def on_annotate(self):
        annotator_index = self.controlnet_combo.currentIndex()

        if self.source_widget.image_editor.original_pixmap is not None:
            source_image = self.source_widget.image_editor.get_painted_image()

            if annotator_index == 0:
                ptr = source_image.bits()
                ptr.setsize(source_image.sizeInBytes())
                image = np.array(ptr).reshape(  # pylint: disable=too-many-function-args
                    source_image.height(), source_image.width(), 4
                )

                low_threshold = 50
                high_threshold = 200

                image = cv2.Canny(  # pylint: disable=no-member
                    image, low_threshold, high_threshold
                )

                image = np.stack([image] * 3, axis=-1)
                qimage = QImage(
                    image.data,
                    image.shape[1],
                    image.shape[0],
                    image.strides[0],
                    QImage.Format.Format_RGB888,
                )
                pixmap = QPixmap.fromImage(qimage)

                self.annotator_widget.image_editor.set_pixmap(pixmap)

    def on_controlnet_added(self):
        source_image = self.qimage_to_pil(
            self.source_widget.image_editor.get_painted_image()
        )

        annotator_image = self.qimage_to_pil(
            self.annotator_widget.image_editor.get_painted_image()
        )

        controlnet = ControlNetDataObject(
            name="canny",
            model_path=os.path.join(
                self.directories.models_controlnets, "controlnet-canny-sdxl-1.0-small"
            ),
            enabled=True,
            source_image=source_image,
            annotator_image=annotator_image,
            guess_mode=False,
            conditioning_scale=round(self.conditioning_scale, 2),
            guidance_start=self.control_guidance_start,
            guidance_end=self.control_guidance_end,
        )
        self.image_generation_data.add_controlnet(controlnet)
        self.generation_updated.emit()

    def qimage_to_pil(self, image: QImage):
        qimage = image
        buffer = QBuffer()
        buffer.open(QIODevice.ReadWrite)
        qimage.save(buffer, "PNG")
        strio = BytesIO()
        strio.write(buffer.data())
        buffer.close()
        strio.seek(0)
        pil_image = Image.open(strio)

        return pil_image

    def on_conditional_scale_changed(self, value):
        self.conditioning_scale = value
        self.conditioning_scale_value_label.setText(f"{value:.2f}")

    def on_guidance_changed(self, values):
        self.control_guidance_start = round(values[0], 2)
        self.control_guidance_end = round(values[1], 2)
        self.guidance_start_value_label.setText(
            f"{int(self.control_guidance_start * 100)}%"
        )
        self.guidance_end_value_label.setText(
            f"{int(self.control_guidance_end * 100)}%"
        )
