import os

import torch
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget
from superqt import QDoubleRangeSlider, QDoubleSlider

from iartisanxl.app.event_bus import EventBus
from iartisanxl.buttons.brush_erase_button import BrushEraseButton
from iartisanxl.buttons.color_button import ColorButton
from iartisanxl.modules.common.controlnet.controlnet_data import ControlNetData
from iartisanxl.modules.common.dialogs.base_dialog import BaseDialog
from iartisanxl.modules.common.image.image_widget import ImageWidget
from iartisanxl.preprocessors.canny.canny_edges_detector import CannyEdgesDetector
from iartisanxl.threads.controlnet.controlnet_preprocess_thread import ControlnetPreprocessThread
from iartisanxl.threads.image.transformed_images_saver_thread import TransformedImagesSaverThread
from iartisanxl.utilities.image.converters import (
    convert_numpy_argb_to_bgr,
    convert_numpy_to_pixmap,
    convert_pixmap_to_numpy,
)


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

        self.controlnet = ControlNetData()
        self.controlnet.generation_width = self.image_generation_data.image_width
        self.controlnet.generation_height = self.image_generation_data.image_height
        self.error = False

        self.canny_detector = None

        self.image_prepare_thread = None
        self.preprocessor_thread = None
        self.processing = False

        self.init_ui()

        self.source_widget.add_empty_layer()
        self.source_widget.set_enabled(True)
        self.preprocessor_widget.add_empty_layer()
        self.preprocessor_widget.set_enabled(True)

        self.source_changed = False
        self.preprocessor_changed = False

    def init_ui(self):
        content_layout = QVBoxLayout()

        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(10, 0, 10, 0)
        control_layout.setSpacing(10)

        self.preprocessor_combo = QComboBox()
        self.preprocessor_combo.addItem("Canny", "controlnet_canny_model")
        self.preprocessor_combo.addItem("Depth", "controlnet_depth_model")
        self.preprocessor_combo.addItem("Inpaint", "controlnet_inpaint_model")
        self.preprocessor_combo.currentIndexChanged.connect(self.on_preprocessor_changed)
        control_layout.addWidget(self.preprocessor_combo)

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
        self.canny_widget.setVisible(False)
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

        brush_erase_button = BrushEraseButton()
        brush_layout.addWidget(brush_erase_button)

        color_button = ColorButton("Color:")
        brush_layout.addWidget(color_button)

        content_layout.addLayout(brush_layout)

        images_layout = QHBoxLayout()
        images_layout.setContentsMargins(2, 0, 4, 0)
        images_layout.setSpacing(2)

        source_layout = QVBoxLayout()
        self.source_widget = ImageWidget(
            "Source image",
            "cn_source",
            self.image_viewer,
            self.controlnet.generation_width,
            self.controlnet.generation_height,
            show_layer_manager=True,
        )
        self.source_widget.set_enabled(False)
        self.source_widget.image_loaded.connect(lambda: self.on_image_loaded(0))
        self.source_widget.image_changed.connect(self.on_source_image_changed)
        self.source_widget.widget_updated.connect(self.on_source_image_changed)
        source_layout.addWidget(self.source_widget)
        self.preprocess_button = QPushButton("Preprocess")
        self.preprocess_button.setObjectName("blue_button")
        self.preprocess_button.setDisabled(True)
        self.preprocess_button.clicked.connect(self.on_prepare_source)
        source_layout.addWidget(self.preprocess_button)
        images_layout.addLayout(source_layout)

        preprocessor_layout = QVBoxLayout()
        self.preprocessor_widget = ImageWidget(
            "Preprocessor",
            "cn_preprocessor",
            self.image_viewer,
            self.controlnet.generation_width,
            self.controlnet.generation_height,
            show_layer_manager=True,
            layer_manager_to_right=True,
        )
        self.preprocessor_widget.set_enabled(False)
        self.preprocessor_widget.image_loaded.connect(lambda: self.on_image_loaded(1))
        self.preprocessor_widget.image_changed.connect(self.on_preprocessor_image_changed)
        self.preprocessor_widget.widget_updated.connect(self.on_preprocessor_image_changed)
        preprocessor_layout.addWidget(self.preprocessor_widget)

        self.add_button = QPushButton("Add")
        self.add_button.setObjectName("green_button")
        self.add_button.setDisabled(True)
        self.add_button.clicked.connect(self.on_prepare_preprocessor)

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
        brush_size_slider.sliderReleased.connect(self.source_widget.image_editor.hide_brush_preview)
        brush_hardness_slider.valueChanged.connect(self.source_widget.image_editor.set_brush_hardness)
        brush_hardness_slider.sliderReleased.connect(self.source_widget.image_editor.hide_brush_preview)

        color_button.color_changed.connect(self.preprocessor_widget.image_editor.set_brush_color)
        brush_size_slider.valueChanged.connect(self.preprocessor_widget.image_editor.set_brush_size)
        brush_size_slider.sliderReleased.connect(self.preprocessor_widget.image_editor.hide_brush_preview)
        brush_hardness_slider.valueChanged.connect(self.preprocessor_widget.image_editor.set_brush_hardness)
        brush_hardness_slider.sliderReleased.connect(self.preprocessor_widget.image_editor.hide_brush_preview)

        self.canny_slider.valueChanged.connect(self.on_canny_threshold_changed)
        self.canny_slider.sliderReleased.connect(self.on_canny_slider_released)

        brush_erase_button.brush_selected.connect(self.source_widget.set_erase_mode)
        brush_erase_button.brush_selected.connect(self.preprocessor_widget.set_erase_mode)

    def closeEvent(self, event):
        self.settings.beginGroup("controlnet_dialog")
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.endGroup()

        self.preprocessor_thread = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        super().closeEvent(event)

    def on_source_image_changed(self):
        self.source_changed = True
        self.preprocess_button.setEnabled(True)

    def on_preprocessor_image_changed(self):
        self.preprocessor_changed = True
        self.add_button.setEnabled(True)

    def on_prepare_source(self):
        if not self.processing:
            layers = self.source_widget.image_editor.get_all_layers()

            if len(layers) == 0:
                self.show_error("You must load an image or create a new one to be able to use preprocessors.")
                return

            self.set_preprocessing()

            if self.source_changed:
                self.image_prepare_thread = TransformedImagesSaverThread(
                    layers,
                    self.controlnet.generation_width,
                    self.controlnet.generation_height,
                    prefix="cn_source_",
                )
                self.image_prepare_thread.merge_finished.connect(self.on_source_prepared)
                self.image_prepare_thread.finished.connect(self.on_image_prepare_thread_finished)
                self.image_prepare_thread.error.connect(self.on_error)
                self.image_prepare_thread.start()
            else:
                self.preprocess()

    def set_preprocessing(self):
        self.processing = True

        self.preprocess_button.setText("Preprocessing...")
        self.preprocess_button.setEnabled(False)
        self.add_button.setEnabled(False)

    def on_error(self, message: str):
        self.error = True
        self.show_error(message)

        self.preprocess_button.setText("Preprocess")
        self.preprocess_button.setDisabled(False)

    def on_image_loaded(self, image_type: int):
        # types: 0 - source, 1 - preprocessor
        if image_type == 0:
            if self.controlnet.source_image is not None and os.path.isfile(self.controlnet.source_image):
                os.remove(self.controlnet.source_image)
                self.controlnet.source_image = None

            if self.controlnet.source_thumb is not None and os.path.isfile(self.controlnet.source_thumb):
                os.remove(self.controlnet.source_thumb)
                self.controlnet.source_thumb = None

            self.source_changed = True
            self.source_widget.set_enabled(True)
        else:
            self.preprocessor_changed = True
            self.preprocessor_widget.set_enabled(True)

    def on_source_prepared(self, image_path: str, thumbnail_path: str):
        if self.controlnet.source_image is not None and os.path.isfile(self.controlnet.source_image):
            os.remove(self.controlnet.source_image)

        if self.controlnet.source_thumb is not None and os.path.isfile(self.controlnet.source_thumb):
            os.remove(self.controlnet.source_thumb)

        self.controlnet.source_image = image_path
        self.controlnet.source_thumb = thumbnail_path

        self.preprocess()

    def on_image_prepare_thread_finished(self):
        self.image_prepare_thread.finished.disconnect(self.on_image_prepare_thread_finished)
        self.image_prepare_thread = None

    def preprocess(self):
        if self.controlnet.source_image is None:
            self.show_error("Couldn't find a source image to preprocess.")
            self.preprocess_button.setText("Preprocess")
            self.preprocess_button.setDisabled(False)
            return

        self.controlnet.adapter_name = self.preprocessor_combo.currentText()
        self.controlnet.adapter_type = self.preprocessor_combo.currentData()
        self.controlnet.type_index = self.preprocessor_combo.currentIndex()
        self.controlnet.depth_type = self.depth_type_combo.currentData()
        self.controlnet.depth_type_index = self.depth_type_combo.currentIndex()

        layer = self.preprocessor_widget.image_editor.get_selected_layer()

        self.preprocessor_thread = ControlnetPreprocessThread(self.controlnet, layer, prefix="cn_preprocessor_")
        self.preprocessor_thread.preprocessor_finished.connect(self.on_preprocessed)
        self.preprocessor_thread.start()

    def on_preprocessed(self, preprocesor_pixmap: QPixmap, image_path: str):
        self.preprocessor_widget.image_editor.set_pixmap(
            preprocesor_pixmap, self.preprocessor_widget.image_editor.selected_layer_id, image_path
        )

        if self.preprocessor_combo.currentData() == "controlnet_canny_model":
            self.canny_widget.setVisible(True)

        self.preprocessor_widget.set_enabled(True)

        self.processing = False
        self.preprocess_button.setText("Preprocess")
        self.source_changed = False

    def preprocess_canny(self):
        source_pixmap = self.source_widget.image_editor.get_scene_as_pixmap()
        numpy_image = convert_pixmap_to_numpy(source_pixmap)
        numpy_image = convert_numpy_argb_to_bgr(numpy_image)

        preprocessor_resolution = (
            int(self.controlnet.generation_width * self.preprocessor_resolution_slider.value()),
            int(self.controlnet.generation_height * self.preprocessor_resolution_slider.value()),
        )

        canny_values = self.canny_slider.value()
        canny_low = int(canny_values[0])
        canny_high = int(canny_values[1])

        self.canny_detector = CannyEdgesDetector()
        preprocessor_image = self.canny_detector.get_canny_edges(
            numpy_image, canny_low, canny_high, resolution=preprocessor_resolution
        )
        preprocesor_pixmap = convert_numpy_to_pixmap(preprocessor_image)

        self.preprocessor_widget.image_editor.set_pixmap(
            preprocesor_pixmap, self.preprocessor_widget.image_editor.selected_layer_id, delete_prev_image=False
        )

    def on_canny_slider_released(self):
        self.set_preprocessing()
        self.preprocess()

    def on_prepare_preprocessor(self):
        if self.processing:
            return

        if not self.preprocessor_changed:
            return

        self.set_adding_controlnet()

        layers = self.preprocessor_widget.image_editor.get_all_layers()

        if len(layers) == 0:
            self.show_error(
                "You'll need to either preprocess an image or load an image in the preprocessor to add the controlnet."
            )
            return

        self.image_prepare_thread = TransformedImagesSaverThread(
            layers,
            self.controlnet.generation_width,
            self.controlnet.generation_height,
            prefix="cn_preprocessor_",
        )
        self.image_prepare_thread.merge_finished.connect(self.on_preprocessor_prepared)
        self.image_prepare_thread.finished.connect(self.on_image_prepare_thread_finished)
        self.image_prepare_thread.error.connect(self.on_error)
        self.image_prepare_thread.start()

    def set_adding_controlnet(self):
        self.processing = True

        self.add_button.setText("Adding controlnet...")
        self.preprocess_button.setEnabled(False)
        self.add_button.setEnabled(False)

    def on_preprocessor_prepared(self, image_path: str, thumbnail_path: str):
        if self.controlnet.preprocessor_image is not None and os.path.isfile(self.controlnet.preprocessor_image):
            os.remove(self.controlnet.preprocessor_image)

        if self.controlnet.preprocessor_thumb is not None and os.path.isfile(self.controlnet.preprocessor_thumb):
            os.remove(self.controlnet.preprocessor_thumb)

        self.controlnet.preprocessor_image = image_path
        self.controlnet.preprocessor_thumb = thumbnail_path

        # save source layer data
        self.controlnet.source_images.images = []
        for layer in self.source_widget.image_editor.get_all_layers():
            layer_name = self.source_widget.layer_manager_widget.get_layer_name(layer.layer_id)
            self.controlnet.source_images.add_image(
                layer.original_path,
                layer.image_path,
                layer.pixmap_item.scale(),
                layer.pixmap_item.x(),
                layer.pixmap_item.y(),
                layer.pixmap_item.rotation(),
                layer_name,
                layer.order,
            )

        # save preprocessor layer data
        self.controlnet.preprocessor_images.images = []
        for layer in self.preprocessor_widget.image_editor.get_all_layers():
            layer_name = self.preprocessor_widget.layer_manager_widget.get_layer_name(layer.layer_id)
            self.controlnet.preprocessor_images.add_image(
                layer.original_path,
                layer.image_path,
                layer.pixmap_item.scale(),
                layer.pixmap_item.x(),
                layer.pixmap_item.y(),
                layer.pixmap_item.rotation(),
                layer_name,
                layer.order,
            )

        self.processing = False
        self.source_changed = False
        self.preprocessor_changed = False
        self.add_button.setText("Update")

        if self.controlnet.adapter_id is None:
            self.event_bus.publish("controlnet", {"action": "add", "controlnet": self.controlnet})
            self.add_button.setText("Update")
        else:
            self.event_bus.publish("controlnet", {"action": "update", "controlnet": self.controlnet})

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

        self.preprocess_canny()

    def on_preprocessor_changed(self):
        if self.preprocessor_combo.currentIndex() == 0:
            if not self.source_changed:
                self.canny_widget.setVisible(True)
            self.depth_widget.setVisible(False)
        elif self.preprocessor_combo.currentIndex() == 1:
            self.canny_widget.setVisible(False)
            self.depth_widget.setVisible(True)
        else:
            self.canny_widget.setVisible(False)
            self.depth_widget.setVisible(False)

        self.preprocessor_changed = True
        self.preprocess_button.setEnabled(True)

    def on_preprocessor_type_changed(self):
        self.preprocessor_changed = True
        self.preprocess_button.setEnabled(True)

    def on_preprocessor_resolution_changed(self, value):
        self.controlnet.preprocessor_resolution = value
        self.preprocessor_resolution_value_label.setText(f"{int(value * 100)}%")
        self.preprocessor_changed = True
        self.preprocess_button.setEnabled(True)

        if self.preprocessor_combo.currentData() == "controlnet_canny_model" and self.canny_widget.isVisible():
            self.preprocess_canny()

    def update_ui(self):
        self.conditioning_scale_slider.setValue(self.controlnet.conditioning_scale)
        self.conditioning_scale_value_label.setText(f"{self.controlnet.conditioning_scale:.2f}")
        self.guidance_slider.setValue((self.controlnet.guidance_start, self.controlnet.guidance_end))

        self.preprocessor_combo.setCurrentIndex(self.controlnet.type_index)

        self.canny_slider.valueChanged.disconnect(self.on_canny_threshold_changed)
        self.canny_slider.setValue((self.controlnet.canny_low, self.controlnet.canny_high))
        self.canny_slider.valueChanged.connect(self.on_canny_threshold_changed)

        self.depth_type_combo.setCurrentIndex(self.controlnet.depth_type_index)
        self.on_preprocessor_changed()

        # restore source layers
        self.source_widget.image_editor.clear_all()
        self.source_widget.layer_manager_widget.list_widget.clear()
        for image in sorted(self.controlnet.source_images.images, key=lambda img: img.order):
            layer_id = self.source_widget.reload_image_layer(image.image_filename, image.image_original, image.order)
            self.source_widget.set_layer_parameters(
                layer_id, image.image_scale, image.image_x_pos, image.image_y_pos, image.image_rotation
            )
            self.source_widget.layer_manager_widget.add_layer(layer_id, image.layer_name)

        # restore preprocessor layers
        self.preprocessor_widget.image_editor.clear_all()
        self.preprocessor_widget.layer_manager_widget.list_widget.clear()
        for image in sorted(self.controlnet.preprocessor_images.images, key=lambda img: img.order):
            layer_id = self.preprocessor_widget.reload_image_layer(
                image.image_filename, image.image_original, image.order
            )
            self.preprocessor_widget.set_layer_parameters(
                layer_id, image.image_scale, image.image_x_pos, image.image_y_pos, image.image_rotation
            )
            self.preprocessor_widget.layer_manager_widget.add_layer(layer_id, image.layer_name)

        if self.controlnet.adapter_id is not None:
            self.add_button.setText("Update")

        self.source_changed = False
        self.preprocessor_changed = False
        self.source_widget.set_enabled(True)
        self.preprocessor_widget.set_enabled(True)

    def reset_ui(self):
        self.source_widget.clear_image()
        self.preprocessor_widget.clear_image()

        self.controlnet = ControlNetData()
        self.controlnet.generation_width = self.image_generation_data.image_width
        self.controlnet.generation_height = self.image_generation_data.image_height

        self.conditioning_scale_slider.setValue(self.controlnet.conditioning_scale)
        self.conditioning_scale_value_label.setText(f"{self.controlnet.conditioning_scale:.2f}")
        self.guidance_slider.setValue((self.controlnet.guidance_start, self.controlnet.guidance_end))
        self.canny_slider.setValue((self.controlnet.canny_low, self.controlnet.canny_high))
        self.preprocessor_combo.setCurrentIndex(self.controlnet.type_index)
        self.on_preprocessor_changed()

        self.source_changed = True
        self.preprocessor_changed = True

        self.add_button.setText("Add")
