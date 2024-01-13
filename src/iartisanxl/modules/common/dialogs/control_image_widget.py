from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QWidget, QPushButton, QSpacerItem, QSizePolicy, QFileDialog
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImageReader, QPixmap

from iartisanxl.modules.common.image.image_editor import ImageEditor
from iartisanxl.modules.common.image_viewer_simple import ImageViewerSimple
from iartisanxl.modules.common.image_control import ImageControl
from iartisanxl.generation.image_generation_data import ImageGenerationData
from iartisanxl.layouts.aspect_ratio_layout import AspectRatioLayout


class ControlImageWidget(QWidget):
    image_loaded = pyqtSignal()
    image_changed = pyqtSignal()

    def __init__(self, text: str, image_viewer: ImageViewerSimple, image_generation_data: ImageGenerationData, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setObjectName("control_image_widget")
        self.text = text
        self.image_viewer = image_viewer
        self.image_generation_data = image_generation_data
        self.image_path = None

        self.setAcceptDrops(True)

        self.editor_width = self.image_generation_data.image_width
        self.editor_height = self.image_generation_data.image_height
        self.aspect_ratio = float(self.editor_width) / float(self.editor_height)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        top_layout = QHBoxLayout()
        source_text_label = QLabel(self.text)
        top_layout.addWidget(source_text_label, alignment=Qt.AlignmentFlag.AlignCenter)
        fit_image_button = QPushButton("Fit")
        top_layout.addWidget(fit_image_button)
        blank_image_button = QPushButton("Blank")
        blank_image_button.setToolTip("Create a new empty image with the selected color as background.")
        top_layout.addWidget(blank_image_button)
        current_image_button = QPushButton("Current")
        current_image_button.setToolTip("Copy the current generated image an set it as an image.")
        top_layout.addWidget(current_image_button)
        load_image_button = QPushButton("Load")
        load_image_button.setToolTip("Load an image from your computer.")
        load_image_button.clicked.connect(self.on_load_image)
        top_layout.addWidget(load_image_button)
        main_layout.addLayout(top_layout)

        image_widget = QWidget()
        self.image_editor = ImageEditor(self.editor_width, self.editor_height, self.aspect_ratio)
        self.image_editor.image_changed.connect(self.on_image_changed)
        self.image_editor.image_scaled.connect(self.update_image_scale)
        self.image_editor.image_moved.connect(self.update_image_position)
        editor_layout = AspectRatioLayout(image_widget, self.aspect_ratio)
        editor_layout.addWidget(self.image_editor)
        image_widget.setLayout(editor_layout)
        main_layout.addWidget(image_widget)

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
        blank_image_button.clicked.connect(self.set_color_image)
        current_image_button.clicked.connect(self.set_current_image)
        reset_image_button.clicked.connect(self.on_reset_image)
        undo_button.clicked.connect(self.image_editor.undo)
        redo_button.clicked.connect(self.image_editor.redo)

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
                self.image_path = path
                pixmap = QPixmap(self.image_path)
                self.image_editor.set_pixmap(pixmap)
                self.image_loaded.emit()

    def on_load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        image_path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg)", options=options)
        if image_path:
            self.image_path = image_path
            self.clear_image()
            pixmap = QPixmap(self.image_path)
            self.image_editor.set_pixmap(pixmap)
            self.image_loaded.emit()

    def set_current_image(self):
        if self.image_viewer.pixmap_item is not None:
            self.clear_image()
            pixmap = self.image_viewer.pixmap_item.pixmap()
            self.image_editor.set_pixmap(pixmap)
            self.image_loaded.emit()

    def set_color_image(self):
        width = self.image_generation_data.image_width
        height = self.image_generation_data.image_height
        self.image_editor.set_color_pixmap(width, height)
        self.image_loaded.emit()

    def clear_image(self):
        self.image_scale_control.reset()
        self.image_x_pos_control.reset()
        self.image_y_pos_control.reset()
        self.image_rotation_control.reset()
        self.image_editor.clear()

    def on_image_changed(self):
        self.image_changed.emit()
