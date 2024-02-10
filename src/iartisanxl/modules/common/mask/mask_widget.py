import os
from datetime import datetime

from PIL import Image
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QSpacerItem, QSizePolicy, QFileDialog, QLabel, QSlider, QGridLayout
from PyQt6.QtCore import pyqtSignal, QUrl, QMimeData, Qt
from PyQt6.QtGui import QImageReader, QGuiApplication, QPixmap
from superqt import QDoubleSlider

from iartisanxl.buttons.color_button import ColorButton
from iartisanxl.modules.common.image.image_editor import ImageEditor
from iartisanxl.modules.common.image_viewer_simple import ImageViewerSimple
from iartisanxl.modules.common.image_control import ImageControl
from iartisanxl.layouts.aspect_ratio_layout import AspectRatioLayout
from iartisanxl.threads.save_merged_image_thread import SaveMergedImageThread
from iartisanxl.modules.common.image.image_data_object import ImageDataObject


class MaskWidget(QWidget):
    image_loaded = pyqtSignal()
    image_changed = pyqtSignal()

    def __init__(self, text: str, prefix: str, image_viewer: ImageViewerSimple, editor_width: int, editor_height: int, *args, **kwargs):
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

        self.init_ui()
        self.create_transparent_background_image()

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
        self.image_editor.image_pasted.connect(self.create_original_image)
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
        undo_button = QPushButton("Undo")
        undo_button.setToolTip("Undo the last drawing.")
        image_bottom_layout.addWidget(undo_button)
        redo_button = QPushButton("Redo")
        redo_button.setToolTip("Redo last drawing that was reverted.")
        image_bottom_layout.addWidget(redo_button)
        main_layout.addLayout(image_bottom_layout)

        image_controls_layout = QHBoxLayout()
        self.image_scale_control = ImageControl("Scale: ", 1.0, 3)
        self.image_scale_control.value_changed.connect(self.image_editor.set_image_scale)
        image_controls_layout.addWidget(self.image_scale_control)
        self.image_x_pos_control = ImageControl("X Pos: ", 0, 0)
        self.image_x_pos_control.value_changed.connect(self.image_editor.set_image_x)
        image_controls_layout.addWidget(self.image_x_pos_control)
        self.image_y_pos_control = ImageControl("Y Pos: ", 0, 0)
        self.image_y_pos_control.value_changed.connect(self.image_editor.set_image_y)
        image_controls_layout.addWidget(self.image_y_pos_control)
        self.image_rotation_control = ImageControl("Rotation: ", 0, 0)
        self.image_rotation_control.value_changed.connect(self.image_editor.rotate_image)
        image_controls_layout.addWidget(self.image_rotation_control)
        main_layout.addLayout(image_controls_layout)

        main_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding))

        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 1)
        main_layout.setStretch(2, 0)
        main_layout.setStretch(3, 0)

        self.setLayout(main_layout)

        fit_image_button.clicked.connect(self.image_editor.fit_image)
        current_image_button.clicked.connect(self.set_current_image)
        reset_image_button.clicked.connect(self.on_reset_image)
        undo_button.clicked.connect(self.image_editor.undo)
        redo_button.clicked.connect(self.image_editor.redo)

        color_button.color_changed.connect(self.image_editor.set_brush_color)
        brush_size_slider.valueChanged.connect(self.image_editor.set_brush_size)
        brush_hardness_slider.valueChanged.connect(self.image_editor.set_brush_hardness)

    def update_image_scale(self, scale: float):
        self.image_scale_control.set_value(scale)

    def update_image_position(self, x, y):
        self.image_x_pos_control.set_value(x)
        self.image_y_pos_control.set_value(y)

    def on_reset_image(self):
        self.image_scale_control.reset()
        self.image_x_pos_control.reset()
        self.image_y_pos_control.reset()
        self.image_rotation_control.reset()
        self.image_editor.clear_and_restore()
        self.image_changed.emit()

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
                self.create_original_image(path)

    def on_load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        selected_path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg)", options=options)
        if selected_path:
            self.create_original_image(selected_path)

    def set_current_image(self):
        if self.image_viewer.pixmap_item is not None:
            self.clear_image()

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{self.prefix}_{timestamp}_original.png"
            filepath = os.path.join("tmp/", filename)

            pixmap = self.image_viewer.pixmap_item.pixmap()
            pixmap.save(filepath)
            self.image_editor.image_path = filepath

            self.set_editor_image_by_path(filepath)

    def create_original_image(self, path):
        self.clear_image()

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{self.prefix}_{timestamp}_original.png"
        image_path = os.path.join("tmp/", filename)
        pil_image = Image.open(path)
        pil_image.save(image_path, format="PNG")

        self.set_editor_image_by_path(image_path)

    def set_editor_image_by_path(self, image_path):
        self.image_editor.set_image(image_path)
        self.image_path = image_path
        self.image_loaded.emit()

    def set_image_parameters(self, scale, x, y, angle):
        self.image_scale_control.set_value(scale)
        self.image_editor.set_image_scale(scale)
        self.image_x_pos_control.set_value(x)
        self.image_editor.set_image_x(x)
        self.image_y_pos_control.set_value(y)
        self.image_editor.set_image_y(y)
        self.image_rotation_control.set_value(angle)
        self.image_editor.rotate_image(angle)

    def clear_image(self):
        if self.image_path is not None and os.path.isfile(self.image_path):
            os.remove(self.image_path)
            self.image_path = None

        self.image_editor.image_path = None

        self.image_scale_control.reset()
        self.image_x_pos_control.reset()
        self.image_y_pos_control.reset()
        self.image_rotation_control.reset()
        self.image_editor.clear()

    def on_image_changed(self):
        self.image_changed.emit()

    def on_image_copy(self):
        self.prepare_copy_thread()
        self.image_copy_thread.image_done.connect(self.on_copy_image_done)
        self.image_copy_thread.start()

    def prepare_copy_thread(self, save_path: str = None):
        image_data = ImageDataObject()
        image_data.image_original = self.image_path
        image_data.image_scale = self.image_scale_control.value
        image_data.image_x_pos = self.image_x_pos_control.value
        image_data.image_y_pos = self.image_y_pos_control.value
        image_data.image_rotation = self.image_rotation_control.value

        drawing_image = self.image_editor.get_layer(1)

        self.image_copy_thread = SaveMergedImageThread(self.editor_width, self.editor_height, image_data, drawing_image, save_path=save_path)
        self.image_copy_thread.finished.connect(self.on_copy_thread_finished)

    def on_copy_image_done(self, image_path):
        clipboard = QGuiApplication.clipboard()
        mime_data = QMimeData()
        mime_data.setUrls([QUrl.fromLocalFile(image_path)])
        clipboard.setMimeData(mime_data)

    def on_image_save(self, image_path):
        self.prepare_copy_thread(save_path=image_path)
        self.image_copy_thread.start()

    def on_copy_thread_finished(self):
        self.image_copy_thread = None

    def create_transparent_background_image(self):
        self.clear_image()

        transparent_pixmap = QPixmap(self.editor_width, self.editor_height)
        transparent_pixmap.fill(Qt.GlobalColor.transparent)
        self.image_editor.set_pixmap(transparent_pixmap)

    def on_erase_clicked(self):
        pass
