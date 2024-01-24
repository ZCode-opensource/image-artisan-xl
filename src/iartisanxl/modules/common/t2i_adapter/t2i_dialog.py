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
from iartisanxl.modules.common.t2i_adapter.t2i_adapter_data_object import T2IAdapterDataObject
from iartisanxl.threads.t2i_annotator_thread import T2IAnnotatorThread


class T2IDialog(BaseDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle("T2I Adapters")
        self.setMinimumSize(500, 500)

        self.settings = QSettings("ZCode", "ImageArtisanXL")
        self.settings.beginGroup("t2i_adapters_dialog")
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        self.settings.endGroup()

        self.event_bus = EventBus()

        self.adapter = T2IAdapterDataObject()
        self.updating = False
        self.source_changed = False
        self.annotator_changed = True
        self.annotating = False
        self.annotate = True
        self.annotator_thread = None

        self.init_ui()

    def init_ui(self):
        content_layout = QVBoxLayout()

        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(10, 0, 10, 0)
        control_layout.setSpacing(10)

        self.annotator_combo = QComboBox()
        self.annotator_combo.addItem("Canny", "t2i_adapter_canny_model")
        self.annotator_combo.addItem("Depth", "t2i_adapter_depth_model")
        self.annotator_combo.addItem("Pose", "t2i_adapter_pose_model")
        self.annotator_combo.addItem("Line Art", "t2i_adapter_lineart_model")
        self.annotator_combo.addItem("Sketch", "t2i_adapter_sketch_model")
        self.annotator_combo.currentIndexChanged.connect(self.on_annotator_changed)
        control_layout.addWidget(self.annotator_combo)

        conditioning_scale_label = QLabel("Conditioning scale:")
        control_layout.addWidget(conditioning_scale_label)
        self.conditioning_scale_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        self.conditioning_scale_slider.setRange(0.0, 2.0)
        self.conditioning_scale_slider.setValue(self.adapter.conditioning_scale)
        self.conditioning_scale_slider.valueChanged.connect(self.on_conditional_scale_changed)
        control_layout.addWidget(self.conditioning_scale_slider)
        self.conditioning_scale_value_label = QLabel(f"{self.adapter.conditioning_scale}")
        control_layout.addWidget(self.conditioning_scale_value_label)

        conditioning_factor_label = QLabel("Conditioning Factor:")
        control_layout.addWidget(conditioning_factor_label)
        self.conditioning_factor_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        self.conditioning_factor_slider.setRange(0.0, 1.0)
        self.conditioning_factor_slider.setValue(self.adapter.conditioning_factor)
        self.conditioning_factor_slider.valueChanged.connect(self.on_conditioning_factor_changed)
        control_layout.addWidget(self.conditioning_factor_slider)
        self.conditioning_factor_value_label = QLabel(f"{int(self.adapter.conditioning_factor * 100)}%")
        control_layout.addWidget(self.conditioning_factor_value_label)
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
        self.canny_low_label = QLabel(f"{self.adapter.canny_low}")
        canny_layout.addWidget(self.canny_low_label)
        self.canny_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.canny_slider.setRange(0, 600)
        self.canny_slider.setValue((self.adapter.canny_low, self.adapter.canny_high))
        self.canny_slider.valueChanged.connect(self.on_canny_threshold_changed)
        canny_layout.addWidget(self.canny_slider)
        self.canny_high_label = QLabel(f"{self.adapter.canny_high}")
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
        self.depth_type_combo.currentIndexChanged.connect(self.on_annotator_type_changed)
        depth_layout.addWidget(self.depth_type_combo)
        self.depth_widget.setVisible(False)
        second_control_layout.addWidget(self.depth_widget)

        self.lineart_widget = QWidget()
        lineart_layout = QHBoxLayout(self.lineart_widget)
        lineart_layout.setSpacing(10)
        lineart_layout.setContentsMargins(0, 0, 0, 0)
        self.lineart_type_combo = QComboBox()
        self.lineart_type_combo.addItem("Anime", "anime_style")
        self.lineart_type_combo.addItem("Open Sketch", "opensketch_style")
        self.lineart_type_combo.addItem("Countour", "contour_style")
        self.lineart_type_combo.currentIndexChanged.connect(self.on_annotator_type_changed)
        lineart_layout.addWidget(self.lineart_type_combo)
        self.lineart_widget.setVisible(False)
        second_control_layout.addWidget(self.lineart_widget)

        self.sketch_widget = QWidget()
        sketch_layout = QHBoxLayout(self.sketch_widget)
        sketch_layout.setSpacing(10)
        sketch_layout.setContentsMargins(0, 0, 0, 0)
        self.sketch_type_combo = QComboBox()
        self.sketch_type_combo.addItem("Pidinet Table 5", "table5")
        self.sketch_type_combo.addItem("Pidinet Table 7", "table7")
        self.sketch_type_combo.currentIndexChanged.connect(self.on_annotator_type_changed)
        sketch_layout.addWidget(self.sketch_type_combo)
        self.sketch_widget.setVisible(False)
        second_control_layout.addWidget(self.sketch_widget)

        annotator_resolution_label = QLabel("Annotator resolution:")
        second_control_layout.addWidget(annotator_resolution_label)
        self.annotator_resolution_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        self.annotator_resolution_slider.setRange(0.05, 1.0)
        self.annotator_resolution_slider.setValue(self.adapter.annotator_resolution)
        self.annotator_resolution_slider.valueChanged.connect(self.on_annotator_resolution_changed)
        second_control_layout.addWidget(self.annotator_resolution_slider)
        self.annotator_resolution_value_label = QLabel(f"{int(self.adapter.annotator_resolution * 100)}%")
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

        color_button = ColorButton("Brush color:")
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
        annotate_button.setObjectName("blue_button")
        annotate_button.clicked.connect(self.on_annotate)
        source_layout.addWidget(annotate_button)
        images_layout.addLayout(source_layout)

        annotator_layout = QVBoxLayout()
        self.annotator_widget = ControlImageWidget("Annotator", self.image_viewer, self.image_generation_data)
        self.annotator_widget.image_loaded.connect(lambda: self.on_image_loaded(1))
        self.annotator_widget.image_changed.connect(self.on_annotator_image_changed)
        annotator_layout.addWidget(self.annotator_widget)

        self.add_button = QPushButton("Add")
        self.add_button.setObjectName("green_button")
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

        self.annotator_thread = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        super().closeEvent(event)

    def on_source_changed(self):
        self.source_changed = True
        self.annotate = True

    def on_annotator_image_changed(self):
        self.annotator_changed = True

    def on_annotate(self):
        if not self.annotating:
            if self.adapter.source_image.image_original is None:
                self.show_error("You must load an image or create a new one to be able to use annotators.")
                return

            if not self.annotate:
                return

            self.annotating = True

            self.adapter.source_image.image_scale = self.source_widget.image_scale_control.value
            self.adapter.source_image.image_x_pos = self.source_widget.image_x_pos_control.value
            self.adapter.source_image.image_y_pos = self.source_widget.image_y_pos_control.value
            self.adapter.source_image.image_rotation = self.source_widget.image_rotation_control.value

            self.adapter.adapter_name = self.annotator_combo.currentText()
            self.adapter.adapter_type = self.annotator_combo.currentData()
            self.adapter.type_index = self.annotator_combo.currentIndex()
            self.adapter.depth_type = self.depth_type_combo.currentData()
            self.adapter.depth_type_index = self.depth_type_combo.currentIndex()
            self.adapter.lineart_type = self.lineart_type_combo.currentData()
            self.adapter.lineart_type_index = self.lineart_type_combo.currentIndex()
            self.adapter.sketch_type = self.sketch_type_combo.currentData()
            self.adapter.sketch_type_index = self.sketch_type_combo.currentIndex()
            self.adapter.generation_width = self.image_generation_data.image_width
            self.adapter.generation_height = self.image_generation_data.image_height

            drawings_pixmap = self.source_widget.image_editor.get_layer(1)

            self.annotator_thread = T2IAnnotatorThread(self.adapter, drawings_pixmap, self.source_changed, self.annotate)
            self.annotator_thread.finished.connect(self.on_annotated)
            self.annotator_thread.start()

    def on_annotated(self):
        pixmap = self.annotator_thread.annotator_pixmap
        self.annotator_thread = None
        self.annotator_widget.image_editor.set_pixmap(pixmap)
        self.source_changed = False
        self.annotate = False
        self.annotator_changed = True
        self.annotating = False

    def on_image_loaded(self, image_type: int):
        # types: 0 - source, 1 - annotator
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        if image_type == 0:
            if self.adapter.source_image.image_original is not None:
                os.remove(self.adapter.source_image.image_original)

            original_filename = f"t2i_{timestamp}_original.png"
            original_path = os.path.join("tmp/", original_filename)

            if self.source_widget.image_path is not None:
                shutil.copy2(self.source_widget.image_path, original_path)
            else:
                # If there is no path in the widget, in means its a blank or the current image, so we need to create the image
                pixmap = self.image_viewer.pixmap_item.pixmap()
                original_filename = f"t2i_{timestamp}_original.png"
                original_path = os.path.join("tmp/", original_filename)
                pixmap.save(original_path)

            self.adapter.source_image.image_original = original_path
            self.source_changed = True
            self.annotate = True
        else:
            # the annotator doesn't have a original, its just the annotated image
            if self.adapter.annotator_image.image_filename is not None:
                os.remove(self.adapter.annotator_image.image_filename)

            if self.adapter.annotator_image.image_thumb is not None:
                os.remove(self.adapter.annotator_image.image_thumb)
                self.adapter.annotator_image.image_thumb = None

            annotated_filename = f"t2i_{timestamp}_annotated.png"
            self.adapter.annotator_image.image_filename = os.path.join("tmp/", annotated_filename)

            if self.annotator_widget.image_path is not None:
                shutil.copy2(self.annotator_widget.image_path, self.adapter.annotator_image.image_filename)
            else:
                # If there is no path in the widget, in means its a blank, so we need to create the image
                pass

            self.annotator_changed = True

    def on_t2i_adapter_added(self):
        if self.updating:
            return

        if not self.annotator_changed:
            return

        self.updating = True

        drawings_pixmap = self.source_widget.image_editor.get_layer(1)
        annotator_drawings_pixmap = self.annotator_widget.image_editor.get_layer(1)

        self.annotator_thread = T2IAnnotatorThread(
            self.adapter, drawings_pixmap, self.source_changed, True, save_annotator=True, annotator_drawings=annotator_drawings_pixmap
        )
        self.annotator_thread.finished.connect(self.on_adapter_image_saved)
        self.annotator_thread.start()

    def on_adapter_image_saved(self):
        self.annotator_thread = None

        if self.adapter.adapter_id is None:
            self.event_bus.publish("t2i_adapters", {"action": "add", "t2i_adapter": self.adapter})
            self.add_button.setText("Update")
        else:
            self.event_bus.publish("t2i_adapters", {"action": "update", "t2i_adapter": self.adapter})

        self.annotator_changed = False
        self.updating = False

    def on_conditional_scale_changed(self, value):
        self.adapter.conditioning_scale = value
        self.conditioning_scale_value_label.setText(f"{value:.2f}")

    def on_conditioning_factor_changed(self, value):
        self.adapter.conditioning_factor = value
        self.conditioning_factor_value_label.setText(f"{int(value * 100)}%")

    def on_canny_threshold_changed(self, values):
        self.adapter.canny_low = int(values[0])
        self.adapter.canny_high = int(values[1])
        self.canny_low_label.setText(f"{self.adapter.canny_low}")
        self.canny_high_label.setText(f"{self.adapter.canny_high}")

        if self.source_widget.image_editor.pixmap_item is not None:
            self.annotate = True
            self.on_annotate()

    def on_annotator_changed(self):
        if self.annotator_combo.currentIndex() == 0:
            self.canny_widget.setVisible(True)
            self.depth_widget.setVisible(False)
            self.lineart_widget.setVisible(False)
            self.sketch_widget.setVisible(False)
        elif self.annotator_combo.currentIndex() == 1:
            self.canny_widget.setVisible(False)
            self.depth_widget.setVisible(True)
            self.lineart_widget.setVisible(False)
            self.sketch_widget.setVisible(False)
        elif self.annotator_combo.currentIndex() == 3:
            self.canny_widget.setVisible(False)
            self.depth_widget.setVisible(False)
            self.lineart_widget.setVisible(True)
            self.sketch_widget.setVisible(False)
        elif self.annotator_combo.currentIndex() == 4:
            self.canny_widget.setVisible(False)
            self.depth_widget.setVisible(False)
            self.lineart_widget.setVisible(False)
            self.sketch_widget.setVisible(True)
        else:
            self.canny_widget.setVisible(False)
            self.depth_widget.setVisible(False)
            self.lineart_widget.setVisible(False)
            self.sketch_widget.setVisible(False)

        self.annotate = True

    def on_annotator_type_changed(self):
        self.annotate = True

    def on_annotator_resolution_changed(self, value):
        self.adapter.annotator_resolution = value
        self.annotator_resolution_value_label.setText(f"{int(value * 100)}%")
        self.annotate = True

    def update_ui(self):
        self.annotate = False
        self.source_changed = False
        self.annotator_changed = False

        self.conditioning_scale_slider.setValue(self.adapter.conditioning_scale)
        self.conditioning_scale_value_label.setText(f"{self.adapter.conditioning_scale:.2f}")
        self.conditioning_factor_slider.setValue(self.adapter.conditioning_factor)

        self.annotator_combo.setCurrentIndex(self.adapter.type_index)
        self.canny_slider.setValue((self.adapter.canny_low, self.adapter.canny_high))
        self.depth_type_combo.setCurrentIndex(self.adapter.depth_type_index)
        self.lineart_type_combo.setCurrentIndex(self.adapter.lineart_type_index)
        self.sketch_type_combo.setCurrentIndex(self.adapter.sketch_type_index)
        self.on_annotator_changed()

        if self.adapter.source_image:
            source_pixmap = QPixmap(self.adapter.source_image.image_original)
            self.source_widget.image_editor.set_pixmap(source_pixmap)
            self.source_widget.set_image_parameters(
                self.adapter.source_image.image_scale,
                self.adapter.source_image.image_x_pos,
                self.adapter.source_image.image_y_pos,
                self.adapter.source_image.image_rotation,
            )

        annotator_pixmap = QPixmap(self.adapter.annotator_image.image_filename)
        self.annotator_widget.image_editor.set_pixmap(annotator_pixmap)

        if self.adapter.adapter_id is not None:
            self.add_button.setText("Update")

    def reset_ui(self):
        self.source_widget.clear_image()
        self.annotator_widget.clear_image()

        self.adapter = T2IAdapterDataObject()
        self.source_changed = False
        self.annotator_changed = True
        self.annotate = True

        self.conditioning_scale_slider.setValue(self.adapter.conditioning_scale)
        self.conditioning_scale_value_label.setText(f"{self.adapter.conditioning_scale:.2f}")
        self.conditioning_factor_slider.setValue(self.adapter.conditioning_factor)
        self.canny_slider.setValue((self.adapter.canny_low, self.adapter.canny_high))
        self.annotator_combo.setCurrentIndex(self.adapter.type_index)
        self.on_annotator_changed()

        self.add_button.setText("Add")
