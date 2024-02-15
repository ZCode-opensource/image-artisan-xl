import os
from datetime import datetime

from PIL import Image
from PyQt6.QtCore import QMimeData, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QImageReader, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)
from superqt import QDoubleSlider

from iartisanxl.buttons.color_button import ColorButton
from iartisanxl.layouts.aspect_ratio_layout import AspectRatioLayout
from iartisanxl.modules.common.image.image_data_object import ImageDataObject
from iartisanxl.modules.common.image.image_editor import ImageEditor
from iartisanxl.modules.common.image_control import ImageControl
from iartisanxl.modules.common.image_viewer_simple import ImageViewerSimple
from iartisanxl.threads.save_merged_image_thread import SaveMergedImageThread


class MaskWidget(QWidget):
    image_loaded = pyqtSignal()
    image_changed = pyqtSignal()

    def __init__(
        self,
        text: str,
        prefix: str,
        image_viewer: ImageViewerSimple,
        editor_width: int,
        editor_height: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.setObjectName("image_widget")
        self.text = text
        self.image_viewer = image_viewer
        self.image_path = None

        self.setAcceptDrops(True)

        self.prefix = prefix
        self.editor_width = editor_width
        self.editor_height = editor_height
        self.aspect_ratio = float(self.editor_width) / float(self.editor_height)

        self.image_copy_thread = None

        self.image_layer_id = None
        self.drawing_layer_id = None

        self.init_ui()
        self.create_drawing_pixmap()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        top_layout = QHBoxLayout()
        fit_image_button = QPushButton("Fit")
        top_layout.addWidget(fit_image_button)
        current_image_button = QPushButton("Current")
        current_image_button.setToolTip("Copy the current generated image an set it as an image.")
        top_layout.addWidget(current_image_button)
        load_image_button = QPushButton("Load")
        load_image_button.setToolTip("Load an image from your computer.")
        load_image_button.clicked.connect(self.on_load_image)
        top_layout.addWidget(load_image_button)
        main_layout.addLayout(top_layout)

        middle_layout = QHBoxLayout()
        middle_left_layout = QVBoxLayout()

        brush_layout = QGridLayout()
        brush_layout.setSpacing(5)
        brush_layout.setContentsMargins(5, 0, 0, 0)
        brush_size_label = QLabel("Brush size:")
        brush_layout.addWidget(brush_size_label, 0, 0)
        brush_size_slider = QSlider(Qt.Orientation.Horizontal)
        brush_size_slider.setFixedWidth(150)
        brush_size_slider.setRange(3, 300)
        brush_size_slider.setValue(20)
        brush_layout.addWidget(brush_size_slider, 0, 1)

        brush_hardness_label = QLabel("Brush hardness:")
        brush_layout.addWidget(brush_hardness_label, 1, 0)
        brush_hardness_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        brush_hardness_slider.setRange(0.0, 0.99)
        brush_hardness_slider.setValue(0.5)
        brush_layout.addWidget(brush_hardness_slider, 1, 1)
        middle_left_layout.addLayout(brush_layout)

        color_layout = QHBoxLayout()
        color_button = ColorButton("Color:")
        color_layout.addWidget(color_button, alignment=Qt.AlignmentFlag.AlignCenter)
        middle_left_layout.addLayout(color_layout)

        erase_layout = QHBoxLayout()
        self.erase_button = QPushButton("Erase")
        self.erase_button.clicked.connect(self.on_erase_clicked)
        erase_layout.addWidget(self.erase_button)
        middle_left_layout.addLayout(erase_layout)

        middle_left_layout.addStretch()

        middle_layout.addLayout(middle_left_layout, 0)

        image_widget = QWidget()
        self.image_editor = ImageEditor(self.editor_width, self.editor_height, self.aspect_ratio)
        self.image_editor.image_changed.connect(self.on_image_changed)
        self.image_editor.image_scaled.connect(self.update_image_scale)
        self.image_editor.image_moved.connect(self.update_image_position)
        self.image_editor.image_rotated.connect(self.update_image_angle)
        self.image_editor.image_pasted.connect(self.set_image)
        self.image_editor.image_copy.connect(self.on_image_copy)
        self.image_editor.image_save.connect(self.on_image_save)
        editor_layout = AspectRatioLayout(image_widget, self.aspect_ratio)
        editor_layout.addWidget(self.image_editor)
        image_widget.setLayout(editor_layout)
        middle_layout.addWidget(image_widget, 1)
        main_layout.addLayout(middle_layout)

        image_bottom_layout = QHBoxLayout()
        reset_image_button = QPushButton("Reset")
        reset_image_button.setToolTip("Reset all modifications of the image including zoom, position and drawing.")
        image_bottom_layout.addWidget(reset_image_button)
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
        # self.delete_drawings_button.clicked.connect(self.on_reset_drawings)
        # self.reset_view_button.clicked.connect(self.image_editor.reset_view)

        color_button.color_changed.connect(self.image_editor.set_brush_color)
        brush_size_slider.valueChanged.connect(self.image_editor.set_brush_size)
        brush_size_slider.sliderReleased.connect(self.image_editor.hide_brush_preview)
        brush_hardness_slider.valueChanged.connect(self.image_editor.set_brush_hardness)
        brush_hardness_slider.sliderReleased.connect(self.image_editor.hide_brush_preview)

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

        if pil_image.size[0] < self.editor_width or pil_image.size[1] < self.editor_height:
            new_image = Image.new("RGBA", (self.editor_width, self.editor_height))
            new_image.paste(pil_image, (0, 0))
            new_image.save(temp_image_path, format="PNG")
        else:
            pil_image.save(temp_image_path, format="PNG")

        self.reset_controls()
        self.set_image_on_layer(temp_image_path, self.image_layer_id)

    def set_image_on_layer(self, path: str, image_layer_id: int = None):
        image_layer_id = self.image_editor.set_image(path, self.image_layer_id)

        if self.image_layer_id is None:
            self.image_editor.set_layer_order(image_layer_id, 0)
            self.image_editor.layer_manager.edit_layer(self.drawing_layer_id, parent_id=image_layer_id)

        self.image_layer_id = image_layer_id
        self.image_loaded.emit()

    def reload_image_layer(self, image_path: str):
        if self.image_layer_id is None:
            image_layer_id = self.image_editor.set_image(image_path, delete_prev_image=False)
            self.image_editor.set_layer_order(image_layer_id, 0)
            self.image_editor.layer_manager.edit_layer(self.drawing_layer_id, parent_id=image_layer_id)
            self.image_layer_id = image_layer_id
        else:
            self.image_editor.set_image(image_path, layer_id=self.image_layer_id, delete_prev_image=False)

    def set_current_image(self):
        if self.image_viewer.pixmap_item is not None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{self.prefix}_{timestamp}_original.png"
            filepath = os.path.join("tmp/", filename)

            pixmap = self.image_viewer.pixmap_item.pixmap()
            pixmap.save(filepath)

            self.reset_controls()
            self.set_image_on_layer(filepath, self.image_layer_id)

    def create_drawing_pixmap(self):
        pixmap = QPixmap(self.editor_width, self.editor_height)
        pixmap.fill(Qt.GlobalColor.transparent)

        self.drawing_layer_id = self.image_editor.set_pixmap(pixmap, self.drawing_layer_id)
        self.image_editor.selected_layer_id = self.drawing_layer_id

    def set_image_parameters(self, image_layer_id, scale, x, y, angle):
        previous_layer_id = self.image_editor.selected_layer_id
        self.image_editor.selected_layer_id = image_layer_id

        self.image_scale_control.set_value(scale)
        self.image_editor.set_image_scale(scale)
        self.image_x_pos_control.set_value(x)
        self.image_editor.set_image_x(x)
        self.image_y_pos_control.set_value(y)
        self.image_editor.set_image_y(y)
        self.image_rotation_control.set_value(angle)
        self.image_editor.rotate_image(angle)

        self.image_editor.selected_layer_id = previous_layer_id

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

    def on_image_copy(self):
        layer = self.image_editor.layer_manager.get_layer_by_id(self.image_layer_id)

        if layer.image_path is not None:
            self.prepare_copy_thread()
            self.image_copy_thread.image_done.connect(self.on_copy_image_done)
            self.image_copy_thread.start()

    def prepare_copy_thread(self, save_path: str = None):
        image_data = ImageDataObject()
        layer = self.image_editor.layer_manager.get_layer_by_id(self.image_layer_id)
        image_data.image_original = layer.image_path
        image_data.image_scale = self.image_scale_control.value
        image_data.image_x_pos = self.image_x_pos_control.value
        image_data.image_y_pos = self.image_y_pos_control.value
        image_data.image_rotation = self.image_rotation_control.value

        drawings_layer = self.image_editor.layer_manager.get_layer_by_id(self.drawing_layer_id)
        drawings_pixmap = drawings_layer.pixmap_item.pixmap()

        self.image_copy_thread = SaveMergedImageThread(
            self.editor_width, self.editor_height, image_data, drawings_pixmap, save_path=save_path
        )
        self.image_copy_thread.finished.connect(self.on_copy_thread_finished)

    def on_copy_thread_finished(self):
        self.image_copy_thread = None

    def on_copy_image_done(self, image_path):
        clipboard = QGuiApplication.clipboard()
        mime_data = QMimeData()
        mime_data.setUrls([QUrl.fromLocalFile(image_path)])
        clipboard.setMimeData(mime_data)

    def on_image_save(self, image_path):
        self.prepare_copy_thread(save_path=image_path)
        self.image_copy_thread.start()

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

    def on_erase_clicked(self):
        pass
