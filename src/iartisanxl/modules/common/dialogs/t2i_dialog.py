import cv2
import numpy as np


from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QComboBox, QWidget
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QImage, QPixmap
from superqt import QDoubleRangeSlider, QDoubleSlider
from transformers import DPTImageProcessor, DPTForDepthEstimation

from iartisanxl.app.event_bus import EventBus
from iartisanxl.buttons.color_button import ColorButton
from iartisanxl.modules.common.dialogs.base_dialog import BaseDialog
from iartisanxl.modules.common.dialogs.control_image_widget import ControlImageWidget
from iartisanxl.annotators.openpose import OpenposeDetector
from iartisanxl.annotators.lineart import LineartDetector
from iartisanxl.annotators.canny import CannyDetector
from iartisanxl.annotators.utils import HWC3, resize_image, get_depth_map, numpy_image_to_pixmap
from iartisanxl.generation.t2i_adapter_data_object import T2IAdapterDataObject
from iartisanxl.formats.image import ImageProcessor


class T2IDialog(BaseDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle("T2I Adapters")
        self.setMinimumSize(800, 800)

        self.settings = QSettings("ZCode", "ImageArtisanXL")
        self.settings.beginGroup("t2i_adapters_dialog")
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        self.settings.endGroup()

        self.event_bus = EventBus()

        self.adapter = None
        self.conditioning_scale = 0.50
        self.conditioning_factor = 1.0

        self.depth_estimator = None
        self.image_processor = None
        self.lineart_detector = None

        self.canny_low = 100
        self.canny_high = 300

        self.init_ui()

    def init_ui(self):
        content_layout = QVBoxLayout()

        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(10, 0, 10, 0)
        control_layout.setSpacing(10)

        self.annotator_combo = QComboBox()
        self.annotator_combo.addItem("Canny", "canny")
        self.annotator_combo.addItem("Depth Midas", "depth")
        self.annotator_combo.addItem("Depth Zoe", "depth_zoe")
        self.annotator_combo.addItem("Pose", "pose")
        self.annotator_combo.addItem("Line Art", "lineart")
        self.annotator_combo.addItem("Sketch", "sketch")
        self.annotator_combo.currentIndexChanged.connect(self.on_annotator_changed)
        control_layout.addWidget(self.annotator_combo)

        conditioning_scale_label = QLabel("Conditioning scale:")
        control_layout.addWidget(conditioning_scale_label)
        self.conditioning_scale_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        self.conditioning_scale_slider.setRange(0.0, 2.0)
        self.conditioning_scale_slider.setValue(self.conditioning_scale)
        self.conditioning_scale_slider.valueChanged.connect(self.on_conditional_scale_changed)
        control_layout.addWidget(self.conditioning_scale_slider)
        self.conditioning_scale_value_label = QLabel(f"{self.conditioning_scale}")
        control_layout.addWidget(self.conditioning_scale_value_label)

        conditioning_factor_label = QLabel("Conditioning Factor:")
        control_layout.addWidget(conditioning_factor_label)
        self.conditioning_factor_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        self.conditioning_factor_slider.setRange(0.0, 1.0)
        self.conditioning_factor_slider.setValue(self.conditioning_factor)
        self.conditioning_factor_slider.valueChanged.connect(self.on_conditioning_factor_changed)
        control_layout.addWidget(self.conditioning_factor_slider)
        self.conditioning_factor_value_label = QLabel(f"{int(self.conditioning_factor * 100)}%")
        control_layout.addWidget(self.conditioning_factor_value_label)
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
        self.add_button.clicked.connect(self.on_t2i_adapter_added)

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
        self.settings.beginGroup("t2i_adapters_dialog")
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.endGroup()
        super().closeEvent(event)

    def on_annotate(self):
        annotator_index = self.annotator_combo.currentIndex()

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

                canny_detector = CannyDetector()
                detected_map = canny_detector(
                    numpy_image,
                    low_threshold=low_threshold,
                    high_threshold=high_threshold,
                    detect_resolution=int(self.image_generation_data.image_height / 2),
                    image_resolution=self.image_generation_data.image_height,
                )

                pixmap = numpy_image_to_pixmap(detected_map)
                self.annotator_widget.image_editor.set_pixmap(pixmap)

                self.annotator_widget.image_editor.set_pixmap(pixmap)
            elif annotator_index == 1:
                if self.depth_estimator is None:
                    self.depth_estimator = DPTForDepthEstimation.from_pretrained("./models/annotators/dpt-hybrid-midas").to("cuda")

                if self.image_processor is None:
                    self.image_processor = DPTImageProcessor.from_pretrained("./models/annotators/dpt-hybrid-midas")

                depthmap = get_depth_map(numpy_image, width, height, self.image_processor, self.depth_estimator)

                qimage = QImage(
                    depthmap.tobytes(),
                    depthmap.shape[1],
                    depthmap.shape[0],
                    QImage.Format.Format_RGB888,
                )
                pixmap = QPixmap.fromImage(qimage)
                self.annotator_widget.image_editor.set_pixmap(pixmap)
            elif annotator_index == 3:
                model_openpose = OpenposeDetector()
                input_image = HWC3(numpy_image)
                detected_map, _ = model_openpose(resize_image(input_image, self.image_generation_data.image_height), True)
                detected_map = HWC3(detected_map)
                img = resize_image(input_image, self.image_generation_data.image_height)
                H, W, _ = img.shape
                # pylint: disable=no-member
                detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                pixmap = numpy_image_to_pixmap(detected_map)
                self.annotator_widget.image_editor.set_pixmap(pixmap)
            elif annotator_index == 4:
                if self.lineart_detector is None:
                    self.lineart_detector = LineartDetector.from_pretrained("./models/annotators/lineart").to("cuda")
                detected_map = self.lineart_detector(
                    numpy_image,
                    detect_resolution=int(self.image_generation_data.image_height / 3),
                    image_resolution=self.image_generation_data.image_height,
                    output_type="np",
                )
                pixmap = numpy_image_to_pixmap(detected_map)
                self.annotator_widget.image_editor.set_pixmap(pixmap)
            else:
                pass

    def on_t2i_adapter_added(self):
        if self.adapter is None:
            self.adapter = T2IAdapterDataObject(
                adapter_type=self.annotator_combo.currentText(),
                enabled=True,
                conditioning_scale=round(self.conditioning_scale, 2),
                conditioning_factor=self.conditioning_factor,
                type_index=self.annotator_combo.currentIndex(),
                canny_low=self.canny_low,
                canny_high=self.canny_high,
            )
        else:
            self.adapter.adapter_type = self.annotator_combo.currentText()
            self.adapter.conditioning_scale = round(self.conditioning_scale, 2)
            self.adapter.conditioning_factor = self.conditioning_factor

        source_image = ImageProcessor()
        source_qimage = self.source_widget.image_editor.get_painted_image()
        source_image.set_qimage(source_qimage)
        self.adapter.source_image_thumb = source_image.get_pillow_thumbnail(target_height=80)
        self.adapter.source_image = source_image.get_pillow_image()

        annotator_image = ImageProcessor()
        annotator_qimage = self.annotator_widget.image_editor.get_painted_image()
        annotator_image.set_qimage(annotator_qimage)
        self.adapter.annotator_image_thumb = annotator_image.get_pillow_thumbnail(target_height=80)
        self.adapter.annotator_image = annotator_image.get_pillow_image()

        if self.adapter.adapter_id is None:
            self.event_bus.publish("t2i_adapters", {"action": "add", "t2i_adapter": self.adapter})
            self.add_button.setText("Update")
        else:
            self.event_bus.publish("t2i_adapters", {"action": "update", "t2i_adapter": self.adapter})

    def on_conditional_scale_changed(self, value):
        self.conditioning_scale = value
        self.conditioning_scale_value_label.setText(f"{value:.2f}")

    def on_conditioning_factor_changed(self, value):
        self.conditioning_factor = value
        self.conditioning_factor_value_label.setText(f"{int(value * 100)}%")

    def on_canny_threshold_changed(self, values):
        self.canny_low = int(values[0])
        self.canny_high = int(values[1])
        self.canny_low_label.setText(f"{self.canny_low}")
        self.canny_high_label.setText(f"{self.canny_high}")
        self.on_annotate()

    def on_annotator_changed(self):
        self.canny_widget.setVisible(self.annotator_combo.currentIndex() == 0)
        self.annotator_widget.image_editor.clear()

    def update_ui(self):
        self.conditioning_scale_slider.setValue(self.adapter.conditioning_scale)
        self.conditioning_scale_value_label.setText(f"{self.adapter.conditioning_scale:.2f}")
        self.conditioning_factor_slider.setValue(self.adapter.conditioning_factor)
        self.canny_slider.setValue((self.adapter.canny_low, self.adapter.canny_high))
        self.annotator_combo.setCurrentIndex(self.adapter.type_index)
        self.on_annotator_changed()

        image_processor = ImageProcessor()
        image_processor.set_pillow_image(self.adapter.source_image)
        self.source_widget.image_editor.set_pixmap(image_processor.get_qpixmap())
        image_processor.set_pillow_image(self.adapter.annotator_image)
        self.annotator_widget.image_editor.set_pixmap(image_processor.get_qpixmap())
        del image_processor

        if self.adapter.adapter_id is not None:
            self.add_button.setText("Update")
