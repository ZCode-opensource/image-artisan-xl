import os
import shutil
from datetime import datetime

import torch
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QComboBox, QWidget
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QPixmap
from superqt import QDoubleRangeSlider, QDoubleSlider

from iartisanxl.app.event_bus import EventBus
from iartisanxl.buttons.color_button import ColorButton
from iartisanxl.modules.common.dialogs.base_dialog import BaseDialog
from iartisanxl.modules.common.dialogs.control_image_widget import ControlImageWidget
from iartisanxl.modules.common.controlnet.controlnet_data_object import ControlNetDataObject
from iartisanxl.threads.preprocessor_thread import PreprocessorThread


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
        self.updating = False
        self.source_changed = False
        self.prepocessor_changed = True
        self.preprocessing = False
        self.preprocess = True
        self.preprocessor_thread = None

        self.init_ui()

    def init_ui(self):
        content_layout = QVBoxLayout()

        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(10, 0, 10, 0)
        control_layout.setSpacing(10)

        self.prepocessor_combo = QComboBox()
        self.prepocessor_combo.addItem("Canny", "controlnet_canny_model")
        self.prepocessor_combo.addItem("Depth", "controlnet_depth_model")
        self.prepocessor_combo.addItem("Pose", "controlnet_pose_model")
        self.prepocessor_combo.addItem("Inpaint", "controlnet_inpaint_model")
        self.prepocessor_combo.currentIndexChanged.connect(self.on_preprocessor_changed)
        control_layout.addWidget(self.prepocessor_combo)

        conditioning_scale_label = QLabel("Conditioning scale:")
        control_layout.addWidget(conditioning_scale_label)
        self.conditioning_scale_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        self.conditioning_scale_slider.setRange(0.0, 2.0)
        self.conditioning_scale_slider.setValue(self.controlnet.conditioning_scale)
        self.conditioning_scale_slider.valueChanged.connect(self.on_conditional_scale_changed)
        control_layout.addWidget(self.conditioning_scale_slider)
        self.conditioning_scale_value_label = QLabel(f"{self.controlnet.conditioning_scale}")
        control_layout.addWidget(self.conditioning_scale_value_label)

        guidance_start_label = QLabel("Guidance Start:")
        control_layout.addWidget(guidance_start_label)
        self.guidance_start_value_label = QLabel(f"{int(self.controlnet.guidance_start * 100)}%")
        control_layout.addWidget(self.guidance_start_value_label)
        self.guidance_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.guidance_slider.setRange(0, 1)
        self.guidance_slider.setValue((self.controlnet.guidance_start, self.controlnet.guidance_end))
        self.guidance_slider.valueChanged.connect(self.on_guidance_changed)
        control_layout.addWidget(self.guidance_slider)
        guidance_end_label = QLabel("End:")
        control_layout.addWidget(guidance_end_label)
        self.guidance_end_value_label = QLabel(f"{int(self.controlnet.guidance_end * 100)}%")
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
        self.canny_low_label = QLabel(f"{self.controlnet.canny_low}")
        canny_layout.addWidget(self.canny_low_label)
        self.canny_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.canny_slider.setRange(0, 600)
        self.canny_slider.setValue((self.controlnet.canny_low, self.controlnet.canny_high))
        self.canny_slider.valueChanged.connect(self.on_canny_threshold_changed)
        canny_layout.addWidget(self.canny_slider)
        self.canny_high_label = QLabel(f"{self.controlnet.canny_high}")
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
        self.depth_type_combo.currentIndexChanged.connect(self.on_preprocessor_type_changed)
        depth_layout.addWidget(self.depth_type_combo)
        self.depth_widget.setVisible(False)
        second_control_layout.addWidget(self.depth_widget)

        preprocessor_resolution_label = QLabel("Preprocessor resolution:")
        second_control_layout.addWidget(preprocessor_resolution_label)
        self.preprocessor_resolution_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        self.preprocessor_resolution_slider.setRange(0.05, 1.0)
        self.preprocessor_resolution_slider.setValue(self.controlnet.preprocessor_resolution)
        self.preprocessor_resolution_slider.valueChanged.connect(self.on_preprocessor_resolution_changed)
        second_control_layout.addWidget(self.preprocessor_resolution_slider)
        self.preprocessor_resolution_value_label = QLabel(f"{int(self.controlnet.preprocessor_resolution * 100)}%")
        second_control_layout.addWidget(self.preprocessor_resolution_value_label)

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
        preprocess_button = QPushButton("Preprocess")
        preprocess_button.setObjectName("blue_button")
        preprocess_button.clicked.connect(self.on_preprocess)
        source_layout.addWidget(preprocess_button)
        images_layout.addLayout(source_layout)

        preprocessor_layout = QVBoxLayout()
        self.preprocessor_widget = ControlImageWidget("Preprocessor", self.image_viewer, self.image_generation_data)
        self.preprocessor_widget.image_loaded.connect(lambda: self.on_image_loaded(1))
        self.preprocessor_widget.image_changed.connect(self.on_preprocessor_image_changed)
        preprocessor_layout.addWidget(self.preprocessor_widget)

        self.add_button = QPushButton("Add")
        self.add_button.setObjectName("green_button")
        self.add_button.clicked.connect(self.on_controlnet_added)

        preprocessor_layout.addWidget(self.add_button)
        images_layout.addLayout(preprocessor_layout)

        content_layout.addLayout(images_layout)

        content_layout.setStretch(0, 0)
        content_layout.setStretch(1, 0)
        content_layout.setStretch(2, 0)
        content_layout.setStretch(3, 1)

        self.main_layout.addLayout(content_layout)

        color_button.color_changed.connect(self.source_widget.image_editor.set_brush_color)
        brush_size_slider.valueChanged.connect(self.source_widget.image_editor.set_brush_size)
        brush_hardness_slider.valueChanged.connect(self.source_widget.image_editor.set_brush_hardness)

        color_button.color_changed.connect(self.preprocessor_widget.image_editor.set_brush_color)
        brush_size_slider.valueChanged.connect(self.preprocessor_widget.image_editor.set_brush_size)
        brush_hardness_slider.valueChanged.connect(self.preprocessor_widget.image_editor.set_brush_hardness)

    def closeEvent(self, event):
        self.settings.beginGroup("controlnet_dialog")
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.endGroup()

        self.preprocessor_thread = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        super().closeEvent(event)

    def on_source_changed(self):
        self.source_changed = True
        self.preprocess = True

    def on_preprocessor_image_changed(self):
        self.prepocessor_changed = True

    def on_preprocess(self):
        if not self.preprocessing:
            if self.controlnet.source_image.image_original is None:
                self.show_error("You must load an image or create a new one to be able to use preprocessors.")
                return

            if not self.preprocess:
                return

            self.preprocessing = True

            self.controlnet.source_image.image_scale = self.source_widget.image_scale_control.value
            self.controlnet.source_image.image_x_pos = self.source_widget.image_x_pos_control.value
            self.controlnet.source_image.image_y_pos = self.source_widget.image_y_pos_control.value
            self.controlnet.source_image.image_rotation = self.source_widget.image_rotation_control.value

            self.controlnet.adapter_name = self.prepocessor_combo.currentText()
            self.controlnet.adapter_type = self.prepocessor_combo.currentData()
            self.controlnet.type_index = self.prepocessor_combo.currentIndex()
            self.controlnet.depth_type = self.depth_type_combo.currentData()
            self.controlnet.depth_type_index = self.depth_type_combo.currentIndex()
            self.controlnet.generation_width = self.image_generation_data.image_width
            self.controlnet.generation_height = self.image_generation_data.image_height

            drawings_pixmap = self.source_widget.image_editor.get_layer(1)

            self.preprocessor_thread = PreprocessorThread(self.controlnet, drawings_pixmap, self.source_changed, self.preprocess)
            self.preprocessor_thread.finished.connect(self.on_preprocessed)
            self.preprocessor_thread.start()

    def on_preprocessed(self):
        pixmap = self.preprocessor_thread.preprocessor_pixmap
        self.preprocessor_thread = None
        self.preprocessor_widget.image_editor.set_pixmap(pixmap)
        self.source_changed = False
        self.preprocess = False
        self.prepocessor_changed = True
        self.preprocessing = False

    def on_image_loaded(self, image_type: int):
        # types: 0 - source, 1 - prepocessor
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        if image_type == 0:
            if self.controlnet.source_image.image_original is not None:
                os.remove(self.controlnet.source_image.image_original)

            original_filename = f"cn_{timestamp}_original.png"
            original_path = os.path.join("tmp/", original_filename)

            if self.source_widget.image_path is not None:
                shutil.copy2(self.source_widget.image_path, original_path)
            else:
                # If there is no path in the widget, in means its a blank or the current image, so we need to create the image
                pixmap = self.image_viewer.pixmap_item.pixmap()
                original_filename = f"cn_{timestamp}_original.png"
                original_path = os.path.join("tmp/", original_filename)
                pixmap.save(original_path)

            self.controlnet.source_image.image_original = original_path
            self.source_changed = True
            self.preprocess = True
        else:
            # the preprocessor doesn't have a original, its just the preprocessed image
            if self.controlnet.preprocessor_image.image_filename is not None:
                os.remove(self.controlnet.preprocessor_image.image_filename)

            if self.controlnet.preprocessor_image.image_thumb is not None:
                os.remove(self.controlnet.preprocessor_image.image_thumb)
                self.controlnet.preprocessor_image.image_thumb = None

            preprocessed_filename = f"cn_{timestamp}_preprocessed.png"
            self.controlnet.preprocessor_image.image_filename = os.path.join("tmp/", preprocessed_filename)

            if self.preprocessor_widget.image_path is not None:
                shutil.copy2(self.preprocessor_widget.image_path, self.controlnet.preprocessor_image.image_filename)
            else:
                # If there is no path in the widget, in means its a blank, so we need to create the image
                pass

            self.prepocessor_changed = True

    def on_controlnet_added(self):
        if self.updating:
            return

        if not self.prepocessor_changed:
            return

        self.updating = True

        drawings_pixmap = self.source_widget.image_editor.get_layer(1)
        prepocessor_drawings_pixmap = self.preprocessor_widget.image_editor.get_layer(1)

        self.preprocessor_thread = PreprocessorThread(
            self.controlnet, drawings_pixmap, self.source_changed, True, save_preprocessor=True, preprocessor_drawings=prepocessor_drawings_pixmap
        )
        self.preprocessor_thread.finished.connect(self.on_controlnet_image_saved)
        self.preprocessor_thread.start()

    def on_controlnet_image_saved(self):
        self.preprocessor_thread = None

        if self.controlnet.adapter_id is None:
            self.event_bus.publish("controlnet", {"action": "add", "controlnet": self.controlnet})
            self.add_button.setText("Update")
        else:
            self.event_bus.publish("controlnet", {"action": "update", "controlnet": self.controlnet})

        self.prepocessor_changed = False
        self.updating = False

    def on_conditional_scale_changed(self, value):
        self.controlnet.conditioning_scale = value
        self.conditioning_scale_value_label.setText(f"{value:.2f}")

    def on_guidance_changed(self, values):
        self.controlnet.guidance_start = round(values[0], 2)
        self.controlnet.guidance_end = round(values[1], 2)
        self.guidance_start_value_label.setText(f"{int(self.controlnet.guidance_start * 100)}%")
        self.guidance_end_value_label.setText(f"{int(self.controlnet.guidance_end * 100)}%")

    def on_canny_threshold_changed(self, values):
        self.controlnet.canny_low = int(values[0])
        self.controlnet.canny_high = int(values[1])
        self.canny_low_label.setText(f"{self.controlnet.canny_low}")
        self.canny_high_label.setText(f"{self.controlnet.canny_high}")

        if self.source_widget.image_editor.pixmap_item is not None:
            self.preprocess = True
            self.on_preprocess()

    def on_preprocessor_changed(self):
        if self.prepocessor_combo.currentIndex() == 0:
            self.canny_widget.setVisible(True)
            self.depth_widget.setVisible(False)
        elif self.prepocessor_combo.currentIndex() == 1:
            self.canny_widget.setVisible(False)
            self.depth_widget.setVisible(True)
        else:
            self.canny_widget.setVisible(False)
            self.depth_widget.setVisible(False)

        self.preprocess = True

    def on_preprocessor_type_changed(self):
        self.preprocess = True

    def on_preprocessor_resolution_changed(self, value):
        self.controlnet.preprocessor_resolution = value
        self.preprocessor_resolution_value_label.setText(f"{int(value * 100)}%")
        self.preprocess = True

    def update_ui(self):
        self.preprocess = False
        self.source_changed = False
        self.prepocessor_changed = False

        self.conditioning_scale_slider.setValue(self.controlnet.conditioning_scale)
        self.conditioning_scale_value_label.setText(f"{self.controlnet.conditioning_scale:.2f}")
        self.guidance_slider.setValue((self.controlnet.guidance_start, self.controlnet.guidance_end))

        self.annotator_combo.setCurrentIndex(self.controlnet.type_index)
        self.canny_slider.setValue((self.controlnet.canny_low, self.controlnet.canny_high))
        self.depth_type_combo.setCurrentIndex(self.controlnet.depth_type_index)
        self.on_preprocessor_changed()

        if self.controlnet.source_image:
            source_pixmap = QPixmap(self.controlnet.source_image.image_original)
            self.source_widget.image_editor.set_pixmap(source_pixmap)
            self.source_widget.set_image_parameters(
                self.controlnet.source_image.image_scale,
                self.controlnet.source_image.image_x_pos,
                self.controlnet.source_image.image_y_pos,
                self.controlnet.source_image.image_rotation,
            )

        preprocessor_pixmap = QPixmap(self.controlnet.preprocessor_image.image_filename)
        self.preprocessor_widget.image_editor.set_pixmap(preprocessor_pixmap)

        if self.controlnet.adapter_id is not None:
            self.add_button.setText("Update")

    def reset_ui(self):
        self.source_widget.clear_image()
        self.preprocessor_widget.clear_image()

        self.controlnet = ControlNetDataObject()
        self.source_changed = False
        self.prepocessor_changed = True
        self.preprocess = True

        self.conditioning_scale_slider.setValue(self.controlnet.conditioning_scale)
        self.conditioning_scale_value_label.setText(f"{self.controlnet.conditioning_scale:.2f}")
        self.guidance_slider.setValue((self.controlnet.guidance_start, self.controlnet.guidance_end))
        self.canny_slider.setValue((self.controlnet.canny_low, self.controlnet.canny_high))
        self.prepocessor_combo.setCurrentIndex(self.controlnet.type_index)
        self.on_preprocessor_changed()

        self.add_button.setText("Add")
