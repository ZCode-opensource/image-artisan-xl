import os
import shutil
from datetime import datetime

import torch
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QComboBox, QWidget
from PyQt6.QtCore import QSettings, Qt
from superqt import QDoubleRangeSlider, QDoubleSlider

from iartisanxl.app.event_bus import EventBus
from iartisanxl.buttons.color_button import ColorButton
from iartisanxl.modules.common.dialogs.base_dialog import BaseDialog
from iartisanxl.modules.common.dialogs.control_image_widget import ControlImageWidget
from iartisanxl.modules.common.controlnet.controlnet_data_object import ControlNetDataObject
from iartisanxl.modules.common.image.image_processor import ImageProcessor
from iartisanxl.modules.common.image.image_data_object import ImageDataObject
from iartisanxl.threads.annotator_thread import AnnotatorThread


class ControlNetDialog(BaseDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle("ControlNet")
        self.setMinimumSize(500, 500)

        self.settings = QSettings("ZCode", "ImageArtisanXL")
        self.settings.beginGroup("controlnet_dialog")
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        self.settings.endGroup()

        self.event_bus = EventBus()

        self.controlnet = ControlNetDataObject()
        self.conditioning_scale = 0.50
        self.control_guidance_start = 0.0
        self.control_guidance_end = 1.0
        self.annotator_resolution = 0.5
        self.canny_low = 100
        self.canny_high = 300

        self.source_changed = True
        self.annotating = False
        self.annotator_thread = None
        self.source_data = ImageDataObject()
        self.annotated_data = ImageDataObject()

        self.init_ui()

    def init_ui(self):
        content_layout = QVBoxLayout()

        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(10, 0, 10, 0)
        control_layout.setSpacing(10)

        self.annotator_combo = QComboBox()
        self.annotator_combo.addItem("Canny", "controlnet_canny_model")
        self.annotator_combo.addItem("Depth Midas", "controlnet_depth_model")
        self.annotator_combo.addItem("Pose", "controlnet_pose_model")
        self.annotator_combo.addItem("Inpaint", "controlnet_inpaint_model")
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

        second_control_layout = QHBoxLayout()
        second_control_layout.setSpacing(10)
        second_control_layout.setContentsMargins(10, 2, 10, 2)
        self.canny_widget = QWidget()
        canny_layout = QHBoxLayout(self.canny_widget)
        canny_layout.setSpacing(10)
        canny_layout.setContentsMargins(0, 0, 0, 0)
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
        second_control_layout.addWidget(self.canny_widget)

        self.depth_widget = QWidget()
        depth_layout = QHBoxLayout(self.depth_widget)
        depth_layout.setSpacing(10)
        depth_layout.setContentsMargins(0, 0, 0, 0)
        self.depth_type_combo = QComboBox()
        self.depth_type_combo.addItem("Depth Hybrid Midas", "dpt-hybrid-midas")
        self.depth_type_combo.addItem("Depth BEiT Base 384", "dpt-beit-base-384")
        self.depth_type_combo.addItem("Depth BEiT Large 512", "dpt-beit-large-512")
        depth_layout.addWidget(self.depth_type_combo)
        self.depth_widget.setVisible(False)
        second_control_layout.addWidget(self.depth_widget)

        annotator_resolution_label = QLabel("Annotator resolution:")
        second_control_layout.addWidget(annotator_resolution_label)
        self.annotator_resolution_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        self.annotator_resolution_slider.setRange(0.05, 1.0)
        self.annotator_resolution_slider.setValue(self.annotator_resolution)
        self.annotator_resolution_slider.valueChanged.connect(self.on_annotator_resolution_changed)
        second_control_layout.addWidget(self.annotator_resolution_slider)
        self.annotator_resolution_value_label = QLabel(f"{int(self.annotator_resolution * 100)}%")
        second_control_layout.addWidget(self.annotator_resolution_value_label)

        content_layout.addLayout(second_control_layout)

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

        color_button = ColorButton("Color:")
        brush_layout.addWidget(color_button)

        content_layout.addLayout(brush_layout)

        images_layout = QHBoxLayout()
        images_layout.setContentsMargins(2, 0, 4, 0)
        images_layout.setSpacing(2)

        source_layout = QVBoxLayout()
        self.source_widget = ControlImageWidget("Source image", self.image_viewer, self.image_generation_data)
        self.source_widget.image_loaded.connect(lambda: self.on_image_loaded(0))
        self.source_widget.image_changed.connect(self.on_source_changed)
        source_layout.addWidget(self.source_widget)
        annotate_button = QPushButton("Annotate")
        annotate_button.clicked.connect(self.on_annotate)
        source_layout.addWidget(annotate_button)
        images_layout.addLayout(source_layout)

        annotator_layout = QVBoxLayout()
        self.annotator_widget = ControlImageWidget("Annotator", self.image_viewer, self.image_generation_data)
        self.annotator_widget.image_loaded.connect(lambda: self.on_image_loaded(1))
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

        self.annotator_thread = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        super().closeEvent(event)

    def on_source_changed(self):
        self.source_changed = True

    def on_annotate(self):
        if not self.annotating:
            if self.source_data.image_original is None:
                self.show_error("You must load an image or create a new one to be able to use annotators.")

            self.annotating = True

            if self.annotated_data.image_filename is not None:
                os.remove(self.annotated_data.image_filename)

            self.source_data.image_scale = self.source_widget.image_scale_control.value
            self.source_data.image_x_pos = self.source_widget.image_x_pos_control.value
            self.source_data.image_y_pos = self.source_widget.image_y_pos_control.value
            self.source_data.image_rotation = self.source_widget.image_rotation_control.value

            resolution = (
                int(self.image_generation_data.image_width * self.annotator_resolution),
                int(self.image_generation_data.image_height * self.annotator_resolution),
            )

            drawings_pixmap = self.source_widget.image_editor.get_layer(1)

            self.annotator_thread = AnnotatorThread(
                self.annotator_combo.currentIndex(),
                self.source_data,
                drawings_pixmap,
                self.source_changed,
                self.image_generation_data.image_width,
                self.image_generation_data.image_height,
                resolution,
                self.canny_low,
                self.canny_high,
                self.depth_type_combo.currentData(),
            )
            self.annotator_thread.finished.connect(self.on_annotated)
            self.annotator_thread.start()

    def on_annotated(self):
        self.controlnet.source_image = self.source_data.image_filename
        self.controlnet.source_image_thumb = self.source_data.image_thumb

        pixmap = self.annotator_thread.annotator_pixmap
        self.annotator_widget.image_editor.set_pixmap(pixmap)
        self.source_changed = False
        self.annotating = False

    def on_image_loaded(self, image_type: int):
        # types: 0 - source, 1 - annotator
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        if image_type == 0:
            if self.source_data.image_original is not None:
                os.remove(self.source_data.image_original)
                os.remove(self.source_data.image_thumb)

            original_filename = f"cn_{timestamp}_original.png"
            original_path = os.path.join("tmp/", original_filename)

            if self.source_widget.image_path is not None:
                shutil.copy2(self.source_widget.image_path, original_path)
            else:
                # If there is no path in the widget, in means its a blank, so we need to create the image
                pixmap = self.image_viewer.pixmap_item.pixmap()
                original_filename = f"cn_{timestamp}_original.png"
                original_path = os.path.join("tmp/", original_filename)
                pixmap.save(original_path)

            self.source_data.image_original = original_path
        else:
            # the annotator doesn't have a original, its just the annotated image
            if self.annotated_data.image_filename is not None:
                os.remove(self.annotated_data.image_filename)

            annotated_filename = f"cn_{timestamp}_annotated.png"
            annotated_path = os.path.join("tmp/", annotated_filename)

            if self.annotator_widget.image_path is not None:
                shutil.copy2(self.source_widget.image_path, annotated_path)
            else:
                # If there is no path in the widget, in means its a blank, so we need to create the image
                pass

            self.annotated_data.image_filename = annotated_path

    def on_controlnet_added(self):
        self.controlnet.adapter_name = self.annotator_combo.currentText()
        self.controlnet.adapter_type = self.annotator_combo.currentData()
        self.controlnet.type_index = self.annotator_combo.currentIndex()
        self.controlnet.conditioning_scale = round(self.conditioning_scale, 2)
        self.controlnet.guidance_start = self.control_guidance_start
        self.controlnet.guidance_end = self.control_guidance_end
        self.controlnet.canny_low = self.canny_low
        self.controlnet.canny_high = self.canny_high

        resolution = (
            int(self.image_generation_data.image_width * self.annotator_resolution),
            int(self.image_generation_data.image_height * self.annotator_resolution),
        )

        drawings_pixmap = self.source_widget.image_editor.get_layer(1)
        annotator_drawings_pixmap = self.annotator_widget.image_editor.get_layer(1)

        self.annotator_thread = AnnotatorThread(
            self.annotator_combo.currentIndex(),
            self.source_data,
            drawings_pixmap,
            self.source_changed,
            self.image_generation_data.image_width,
            self.image_generation_data.image_height,
            resolution,
            self.canny_low,
            self.canny_high,
            self.depth_type_combo.currentData(),
            save_annotator=True,
            annotator_drawings=annotator_drawings_pixmap,
        )
        self.annotator_thread.finished.connect(self.on_controlnet_image_saved)
        self.annotator_thread.start()

    def on_controlnet_image_saved(self):
        if self.controlnet.annotator_image is not None:
            os.remove(self.controlnet.annotator_image)
            os.remove(self.controlnet.annotator_image_thumb)
        self.controlnet.annotator_image = self.annotator_thread.annotator_path
        self.controlnet.annotator_image_thumb = self.annotator_thread.annotator_thumb_path

        if self.controlnet.adapter_id is None:
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
        if self.annotator_combo.currentIndex() == 0:
            self.canny_widget.setVisible(True)
            self.depth_widget.setVisible(False)
        elif self.annotator_combo.currentIndex() == 1:
            self.canny_widget.setVisible(False)
            self.depth_widget.setVisible(True)
        else:
            self.canny_widget.setVisible(False)
            self.depth_widget.setVisible(False)

    def on_annotator_resolution_changed(self, value):
        self.annotator_resolution = value
        self.annotator_resolution_value_label.setText(f"{int(value * 100)}%")

    def update_ui(self):
        self.conditioning_scale_slider.setValue(self.controlnet.conditioning_scale)
        self.conditioning_scale_value_label.setText(f"{self.controlnet.conditioning_scale:.2f}")
        self.guidance_slider.setValue((self.controlnet.guidance_start, self.controlnet.guidance_end))
        self.canny_slider.setValue((self.controlnet.canny_low, self.controlnet.canny_high))
        self.annotator_combo.setCurrentIndex(self.controlnet.type_index)
        self.on_annotator_changed()

        image_processor = ImageProcessor()
        image_processor.set_pillow_image(self.controlnet.source_image)
        self.source_widget.image_editor.set_pixmap(image_processor.get_qpixmap())
        image_processor.set_pillow_image(self.controlnet.annotator_image)
        self.annotator_widget.image_editor.set_pixmap(image_processor.get_qpixmap())
        del image_processor

        if self.controlnet.adapter_id is not None:
            self.add_button.setText("Update")

    def reset_ui(self):
        self.conditioning_scale_slider.setValue(0.50)
        self.conditioning_scale_value_label.setText(f"{0.50:.2f}")
        self.guidance_slider.setValue((0, 1))
        self.canny_slider.setValue((100, 300))
        self.annotator_combo.setCurrentIndex(0)
        self.on_annotator_changed()

        self.source_widget.clear_image()
        self.annotator_widget.clear_image()

        self.add_button.setText("Add")
