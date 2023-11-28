import cv2
import numpy as np
import torch

from PyQt6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QComboBox,
    QWidget,
)
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QImage, QPixmap
from superqt import QDoubleRangeSlider, QDoubleSlider
from transformers import DPTImageProcessor, DPTForDepthEstimation

from iartisanxl.app.event_bus import EventBus
from iartisanxl.buttons.color_button import ColorButton
from iartisanxl.modules.common.dialogs.base_dialog import BaseDialog
from iartisanxl.modules.common.dialogs.control_image_widget import ControlImageWidget
from iartisanxl.generation.controlnet_data_object import ControlNetDataObject
from iartisanxl.formats.image import ImageProcessor


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

        self.event_bus = EventBus()

        self.controlnet = None
        self.conditioning_scale = 0.50
        self.control_guidance_start = 0.0
        self.control_guidance_end = 1.0

        self.depth_estimator = None
        self.image_processor = None

        self.canny_low = 100
        self.canny_high = 300

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
        self.controlnet_combo.currentIndexChanged.connect(self.on_annotator_changed)
        control_layout.addWidget(self.controlnet_combo)

        conditioning_scale_label = QLabel("Conditioning scale:")
        control_layout.addWidget(conditioning_scale_label)
        self.conditioning_scale_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        self.conditioning_scale_slider.setRange(0.0, 2.0)
        self.conditioning_scale_slider.setValue(self.conditioning_scale)
        self.conditioning_scale_slider.valueChanged.connect(self.on_conditional_scale_changed)
        control_layout.addWidget(self.conditioning_scale_slider)
        self.conditioning_scale_value_label = QLabel(f"{self.conditioning_scale}")
        control_layout.addWidget(self.conditioning_scale_value_label)

        guidance_start_label = QLabel("Guidance Start:")
        control_layout.addWidget(guidance_start_label)
        self.guidance_start_value_label = QLabel(f"{int(self.control_guidance_start * 100)}%")
        control_layout.addWidget(self.guidance_start_value_label)
        self.guidance_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.guidance_slider.setRange(0, 1)
        self.guidance_slider.setValue((self.control_guidance_start, self.control_guidance_end))
        self.guidance_slider.valueChanged.connect(self.on_guidance_changed)
        control_layout.addWidget(self.guidance_slider)
        guidance_end_label = QLabel("End:")
        control_layout.addWidget(guidance_end_label)
        self.guidance_end_value_label = QLabel(f"{int(self.control_guidance_end * 100)}%")
        control_layout.addWidget(self.guidance_end_value_label)
        content_layout.addLayout(control_layout)

        self.canny_widget = QWidget()
        canny_layout = QHBoxLayout(self.canny_widget)
        canny_label = QLabel("Canny tresholds:")
        canny_layout.addWidget(canny_label)
        self.canny_low_label = QLabel(f"{self.canny_low}")
        canny_layout.addWidget(self.canny_low_label)
        self.canny_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.canny_slider.setRange(0, 600)
        self.canny_slider.setValue((self.canny_low, self.canny_high))
        self.canny_slider.valueChanged.connect(self.on_canny_threshold_changed)
        canny_layout.addWidget(self.canny_slider)
        self.canny_high_label = QLabel(f"{self.canny_high}")
        canny_layout.addWidget(self.canny_high_label)
        content_layout.addWidget(self.canny_widget)

        brush_layout = QHBoxLayout()
        brush_layout.setContentsMargins(10, 0, 10, 0)
        brush_layout.setSpacing(10)

        brush_size_label = QLabel("Brush size:")
        brush_layout.addWidget(brush_size_label)
        brush_size_slider = QSlider(Qt.Orientation.Horizontal)
        brush_size_slider.setRange(3, 300)
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

        source_layout = QVBoxLayout()
        self.source_widget = ControlImageWidget("Source image", self.image_viewer, self.image_generation_data)
        source_layout.addWidget(self.source_widget)
        annotate_button = QPushButton("Annotate")
        annotate_button.clicked.connect(self.on_annotate)
        source_layout.addWidget(annotate_button)
        images_layout.addLayout(source_layout)

        annotator_layout = QVBoxLayout()
        self.annotator_widget = ControlImageWidget("Annotator", self.image_viewer, self.image_generation_data)
        annotator_layout.addWidget(self.annotator_widget)
        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.on_controlnet_added)
        annotator_layout.addWidget(self.add_button)
        images_layout.addLayout(annotator_layout)

        content_layout.addLayout(images_layout)

        content_layout.setStretch(0, 0)
        content_layout.setStretch(1, 0)
        content_layout.setStretch(2, 0)
        content_layout.setStretch(3, 1)

        self.main_layout.addLayout(content_layout)

        color_button.color_changed.connect(self.source_widget.image_editor.set_brush_color)
        brush_size_slider.valueChanged.connect(self.source_widget.image_editor.set_brush_size)
        brush_hardness_slider.valueChanged.connect(self.source_widget.image_editor.set_brush_hardness)

        color_button.color_changed.connect(self.annotator_widget.image_editor.set_brush_color)
        brush_size_slider.valueChanged.connect(self.annotator_widget.image_editor.set_brush_size)
        brush_hardness_slider.valueChanged.connect(self.annotator_widget.image_editor.set_brush_hardness)

    def closeEvent(self, event):
        self.settings.beginGroup("controlnet_dialog")
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.endGroup()
        super().closeEvent(event)

    def on_annotate(self):
        annotator_index = self.controlnet_combo.currentIndex()

        if self.source_widget.image_editor.original_pixmap is not None:
            source_image = self.source_widget.image_editor.get_painted_image()
            width, height = source_image.width(), source_image.height()
            ptr = source_image.bits()
            ptr.setsize(height * width * 4)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
            numpy_image = arr[..., :3]

            if annotator_index == 0:
                low_threshold = self.canny_low
                high_threshold = self.canny_high

                canny_image = cv2.Canny(numpy_image, low_threshold, high_threshold)  # pylint: disable=no-member

                canny_image = np.stack([canny_image] * 3, axis=-1)
                qimage = QImage(
                    canny_image.data,
                    canny_image.shape[1],
                    canny_image.shape[0],
                    canny_image.strides[0],
                    QImage.Format.Format_RGB888,
                )
                pixmap = QPixmap.fromImage(qimage)

                self.annotator_widget.image_editor.set_pixmap(pixmap)
            elif annotator_index == 1:
                if self.depth_estimator is None:
                    self.depth_estimator = DPTForDepthEstimation.from_pretrained("./models/annotators/dpt-hybrid-midas").to("cuda")

                if self.image_processor is None:
                    self.image_processor = DPTImageProcessor.from_pretrained("./models/annotators/dpt-hybrid-midas")

                depthmap = self.get_depth_map(numpy_image, width, height)

                qimage = QImage(
                    depthmap.tobytes(),
                    depthmap.shape[1],
                    depthmap.shape[0],
                    QImage.Format.Format_RGB888,
                )
                pixmap = QPixmap.fromImage(qimage)
                self.annotator_widget.image_editor.set_pixmap(pixmap)
            else:
                pass

    def on_controlnet_added(self):
        if self.controlnet is None:
            self.controlnet = ControlNetDataObject(
                controlnet_type=self.controlnet_combo.currentText(),
                enabled=True,
                guess_mode=False,
                conditioning_scale=round(self.conditioning_scale, 2),
                guidance_start=self.control_guidance_start,
                guidance_end=self.control_guidance_end,
                type_index=self.controlnet_combo.currentIndex(),
                canny_low=self.canny_low,
                canny_high=self.canny_high,
            )
        else:
            self.controlnet.conditioning_scale = round(self.conditioning_scale, 2)
            self.controlnet.guidance_start = self.control_guidance_start
            self.controlnet.guidance_end = self.control_guidance_end

        source_image = ImageProcessor()
        source_qimage = self.source_widget.image_editor.get_painted_image()
        source_image.set_qimage(source_qimage)
        self.controlnet.source_image_thumb = source_image.get_pillow_thumbnail(target_height=80)
        self.controlnet.source_image = source_image.get_pillow_image()

        annotator_image = ImageProcessor()
        annotator_qimage = self.annotator_widget.image_editor.get_painted_image()
        annotator_image.set_qimage(annotator_qimage)
        self.controlnet.annotator_image_thumb = annotator_image.get_pillow_thumbnail(target_height=80)
        self.controlnet.annotator_image = annotator_image.get_pillow_image()

        if self.controlnet.controlnet_id is None:
            self.event_bus.publish("controlnet", {"action": "add", "controlnet": self.controlnet})
            self.add_button.setText("Update")
        else:
            self.event_bus.publish("controlnet", {"action": "update", "controlnet": self.controlnet})

    def on_conditional_scale_changed(self, value):
        self.conditioning_scale = value
        self.conditioning_scale_value_label.setText(f"{value:.2f}")

    def on_guidance_changed(self, values):
        self.control_guidance_start = round(values[0], 2)
        self.control_guidance_end = round(values[1], 2)
        self.guidance_start_value_label.setText(f"{int(self.control_guidance_start * 100)}%")
        self.guidance_end_value_label.setText(f"{int(self.control_guidance_end * 100)}%")

    def on_canny_threshold_changed(self, values):
        self.canny_low = int(values[0])
        self.canny_high = int(values[1])
        self.canny_low_label.setText(f"{self.canny_low}")
        self.canny_high_label.setText(f"{self.canny_high}")
        self.on_annotate()

    def on_annotator_changed(self):
        self.canny_widget.setVisible(self.controlnet_combo.currentIndex() == 0)
        self.annotator_widget.image_editor.clear()

    def update_ui(self):
        self.conditioning_scale_slider.setValue(self.controlnet.conditioning_scale)
        self.conditioning_scale_value_label.setText(f"{self.controlnet.conditioning_scale:.2f}")
        self.guidance_slider.setValue((self.controlnet.guidance_start, self.controlnet.guidance_end))
        self.canny_slider.setValue((self.controlnet.canny_low, self.controlnet.canny_high))
        self.controlnet_combo.setCurrentIndex(self.controlnet.type_index)
        self.on_annotator_changed()

        image_processor = ImageProcessor()
        image_processor.set_pillow_image(self.controlnet.source_image)
        self.source_widget.image_editor.set_pixmap(image_processor.get_qpixmap())
        image_processor.set_pillow_image(self.controlnet.annotator_image)
        self.annotator_widget.image_editor.set_pixmap(image_processor.get_qpixmap())
        del image_processor

    def get_depth_map(self, image, image_width, image_height):
        image = self.image_processor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = self.depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(image_height, image_width),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        depth_map = (depth_map * 255.0).clip(0, 255).to(torch.uint8)
        image = torch.cat([depth_map] * 3, dim=1)

        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]

        return image
