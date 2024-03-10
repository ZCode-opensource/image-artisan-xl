import os
from datetime import datetime

from PIL import Image
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImageReader, QPixmap
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QPushButton, QSizePolicy, QSpacerItem, QVBoxLayout, QWidget

from iartisanxl.layouts.aspect_ratio_layout import AspectRatioLayout
from iartisanxl.modules.common.image.image_editor import ImageEditor
from iartisanxl.modules.common.image_control import ImageControl
from iartisanxl.modules.common.image_viewer_simple import ImageViewerSimple
from iartisanxl.modules.common.mask.mask_image import MaskImage


class MaskWidget(QWidget):
    image_loaded = pyqtSignal()
    image_changed = pyqtSignal()

    def __init__(self, text: str, prefix: str, image_viewer: ImageViewerSimple, editor_width: int, editor_height: int):
        super().__init__()

        self.setObjectName("image_widget")
        self.text = text
        self.image_viewer = image_viewer
        self.image_path = None

        self.setAcceptDrops(True)

        self.prefix = prefix
        self.editor_width = editor_width
        self.editor_height = editor_height
        self.aspect_ratio = float(self.editor_width) / float(self.editor_height)

        self.init_ui()

        self.image_layer_id = self.image_editor.add_empty_layer()
        self.drawing_layer_id = self.image_editor.add_empty_layer()
        self.image_editor.set_layer_locked(self.drawing_layer_id, False)
        self.image_editor.selected_layer_id = self.drawing_layer_id

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        top_layout = QHBoxLayout()
        fit_image_button = QPushButton("Fit background")
        top_layout.addWidget(fit_image_button)
        current_image_button = QPushButton("Generated image as background")
        current_image_button.setToolTip("Copy the current generated image an set it as an image.")
        top_layout.addWidget(current_image_button)
        load_image_button = QPushButton("Load background image")
        load_image_button.setToolTip("Load an image from your computer.")
        load_image_button.clicked.connect(self.on_load_image)
        top_layout.addWidget(load_image_button)
        main_layout.addLayout(top_layout)

        image_widget = QWidget()
        self.image_editor = ImageEditor(self.editor_width, self.editor_height, self.aspect_ratio)
        self.image_editor.set_enable_copy(False)
        self.image_editor.set_enable_save(False)
        self.image_editor.image_changed.connect(self.on_image_changed)
        self.image_editor.image_scaled.connect(self.update_image_scale)
        self.image_editor.image_moved.connect(self.update_image_position)
        self.image_editor.image_rotated.connect(self.update_image_angle)
        self.image_editor.image_pasted.connect(self.set_image)
        editor_layout = AspectRatioLayout(image_widget, self.aspect_ratio)
        editor_layout.addWidget(self.image_editor)
        image_widget.setLayout(editor_layout)
        main_layout.addWidget(image_widget)

        image_bottom_layout = QHBoxLayout()
        reset_image_button = QPushButton("Reset background image")
        image_bottom_layout.addWidget(reset_image_button)
        clear_mask_button = QPushButton("Clear mask")
        clear_mask_button.clicked.connect(self.on_clear_mask)
        image_bottom_layout.addWidget(clear_mask_button)
        main_layout.addLayout(image_bottom_layout)

        image_controls_layout = QHBoxLayout()
        self.image_scale_control = ImageControl("Scale: ", 1.0, 3)
        self.image_scale_control.value_changed.connect(self.on_image_scale)
        image_controls_layout.addWidget(self.image_scale_control)
        self.image_x_pos_control = ImageControl("X Pos: ", 0, 0)
        self.image_x_pos_control.value_changed.connect(self.on_image_x)
        image_controls_layout.addWidget(self.image_x_pos_control)
        self.image_y_pos_control = ImageControl("Y Pos: ", 0, 0)
        self.image_y_pos_control.value_changed.connect(self.on_image_y)
        image_controls_layout.addWidget(self.image_y_pos_control)
        self.image_rotation_control = ImageControl("Rotation: ", 0, 0)
        self.image_rotation_control.value_changed.connect(self.on_image_rotate)
        image_controls_layout.addWidget(self.image_rotation_control)
        main_layout.addLayout(image_controls_layout)

        main_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding))

        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 1)
        main_layout.setStretch(2, 0)
        main_layout.setStretch(3, 0)

        self.setLayout(main_layout)

        fit_image_button.clicked.connect(self.on_fit_image)
        current_image_button.clicked.connect(self.set_current_image)
        reset_image_button.clicked.connect(self.on_reset_image)

    def update_image_scale(self, scale: float):
        self.image_scale_control.set_value(scale)

    def update_image_position(self, x, y):
        self.image_x_pos_control.set_value(x)
        self.image_y_pos_control.set_value(y)

    def update_image_angle(self, angle):
        self.image_rotation_control.set_value(angle)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.image_editor.drop_lightbox.show()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.image_editor.drop_lightbox.hide()
        event.accept()

    def dropEvent(self, event):
        self.image_editor.drop_lightbox.hide()

        for url in event.mimeData().urls():
            path = url.toLocalFile()

            reader = QImageReader(path)

            if reader.canRead():
                self.set_image(path)

    def on_load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        selected_path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg)", options=options)
        if selected_path:
            self.set_image(selected_path)

    def set_image(self, path: str):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{self.prefix}_{timestamp}_original.png"
        temp_image_path = os.path.join("tmp/", filename)
        pil_image = Image.open(path)
        pil_image.save(temp_image_path, format="PNG")
        self.image_editor.selected_layer_id = self.image_layer_id

        # if there's an image before and not saved, delete it
        layer = self.image_editor.layer_manager.get_layer_by_id(self.image_layer_id)
        if layer.image_path is None and layer.original_path is not None:
            os.remove(layer.original_path)

        self.image_editor.set_image(temp_image_path)
        self.image_editor.selected_layer_id = self.drawing_layer_id
        self.image_loaded.emit()

    def reload_mask(self, mask_image: MaskImage):
        if mask_image.background_image is not None:
            self.image_editor.selected_layer_id = self.image_layer_id
            self.image_editor.set_image(mask_image.background_image.image_original, delete_prev_image=False)

            self.set_image_parameters(
                self.image_layer_id,
                mask_image.background_image.image_scale,
                mask_image.background_image.image_x_pos,
                mask_image.background_image.image_y_pos,
                mask_image.background_image.image_rotation,
            )

        if mask_image.mask_image is not None:
            self.image_editor.selected_layer_id = self.drawing_layer_id
            self.image_editor.set_image(mask_image.mask_image.image_filename, delete_prev_image=False)
            self.image_editor.set_layer_locked(self.drawing_layer_id, False)

    def set_current_image(self):
        if self.image_viewer.pixmap_item is not None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{self.prefix}_{timestamp}_original.png"
            filepath = os.path.join("tmp/", filename)

            pixmap = self.image_viewer.pixmap_item.pixmap()
            pixmap.save(filepath)

            self.set_image(filepath)

    def set_image_parameters(self, layer_id: int, scale: float, x: int, y: int, angle: float):
        self.image_editor.selected_layer_id = layer_id

        self.image_scale_control.set_value(scale)
        self.image_editor.set_image_scale(scale)
        self.image_x_pos_control.set_value(x)
        self.image_editor.set_image_x(x)
        self.image_y_pos_control.set_value(y)
        self.image_editor.set_image_y(y)
        self.image_rotation_control.set_value(angle)
        self.image_editor.rotate_image(angle)

    def reset_controls(self):
        self.image_scale_control.reset()
        self.image_x_pos_control.reset()
        self.image_y_pos_control.reset()
        self.image_rotation_control.reset()

    def clear_image(self):
        self.image_scale_control.reset()
        self.image_x_pos_control.reset()
        self.image_y_pos_control.reset()
        self.image_rotation_control.reset()
        self.image_editor.clear_all()

    def on_image_changed(self):
        self.image_changed.emit()

    def on_fit_image(self):
        self.image_editor.selected_layer_id = self.image_layer_id
        self.image_editor.fit_image()
        self.image_editor.selected_layer_id = self.drawing_layer_id

    def on_image_scale(self, scale_factor: float):
        self.image_editor.selected_layer_id = self.image_layer_id
        self.image_editor.set_image_scale(scale_factor)
        self.image_editor.selected_layer_id = self.drawing_layer_id

    def on_image_x(self, x_position: float):
        self.image_editor.selected_layer_id = self.image_layer_id
        self.image_editor.set_image_x(x_position)
        self.image_editor.selected_layer_id = self.drawing_layer_id

    def on_image_y(self, y_position: float):
        self.image_editor.selected_layer_id = self.image_layer_id
        self.image_editor.set_image_y(y_position)
        self.image_editor.selected_layer_id = self.drawing_layer_id

    def on_image_rotate(self, angle: float):
        self.image_editor.selected_layer_id = self.image_layer_id
        self.image_editor.rotate_image(angle)
        self.image_editor.selected_layer_id = self.drawing_layer_id

    def on_reset_image(self):
        self.image_editor.selected_layer_id = self.image_layer_id
        self.image_editor.clear_and_restore()
        self.image_editor.selected_layer_id = self.drawing_layer_id
        self.image_changed.emit()

    def on_reset_drawings(self):
        self.image_editor.clear_and_restore()

    def set_erase_mode(self, value: bool):
        self.image_editor.erasing = value

    def on_clear_mask(self):
        pixmap = QPixmap(self.editor_width, self.editor_height)
        pixmap.fill(Qt.GlobalColor.transparent)

        self.image_editor.selected_layer_id = self.drawing_layer_id
        self.image_editor.set_pixmap(pixmap, self.drawing_layer_id)
        self.image_editor.set_layer_locked(self.drawing_layer_id, False)
