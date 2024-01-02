from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog
from PyQt6.QtGui import QImageReader, QPixmap
from PyQt6.QtCore import pyqtSignal, QTimer

from iartisanxl.modules.common.image.image_adder_preview import ImageAdderPreview
from iartisanxl.modules.common.image_viewer_simple import ImageViewerSimple
from iartisanxl.modules.common.image_control import ImageControl
from iartisanxl.layouts.aspect_ratio_layout import AspectRatioLayout


class IPAdapterImageWidget(QWidget):
    image_added = pyqtSignal()
    image_dropped = pyqtSignal()

    def __init__(self, text: str, image_viewer: ImageViewerSimple, save_directory: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setObjectName("ip_adapter_image_widget")
        self.text = text
        self.image_viewer = image_viewer
        self.image_path = ""
        self.save_directory = save_directory
        self.image_id = None

        self.setAcceptDrops(True)

        self.editor_width = 224
        self.editor_height = 224
        self.aspect_ratio = float(self.editor_width) / float(self.editor_height)

        self.init_ui()

        self.enable_add_image_button()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        top_layout = QHBoxLayout()
        load_image_button = QPushButton("Load")
        load_image_button.clicked.connect(self.on_load_image)
        top_layout.addWidget(load_image_button)
        current_image_button = QPushButton("Current")
        top_layout.addWidget(current_image_button)
        fit_image_button = QPushButton("Fit")
        top_layout.addWidget(fit_image_button)
        reset_image_button = QPushButton("Reset")
        top_layout.addWidget(reset_image_button)

        main_layout.addLayout(top_layout)

        image_widget = QWidget()
        self.image_editor = ImageAdderPreview(self.editor_width, self.editor_height, self.aspect_ratio, self.save_directory)
        self.image_editor.image_moved.connect(self.update_image_position)
        self.image_editor.image_scaled.connect(self.update_image_scale)
        editor_layout = AspectRatioLayout(image_widget, self.aspect_ratio)
        editor_layout.addWidget(self.image_editor)
        main_layout.addWidget(image_widget)

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

        images_actions_layout = QHBoxLayout()
        self.add_image_button = QPushButton("Add image")
        self.add_image_button.clicked.connect(self.on_add_image)
        images_actions_layout.addWidget(self.add_image_button)

        main_layout.addLayout(images_actions_layout)

        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 1)
        main_layout.setStretch(2, 0)
        main_layout.setStretch(3, 0)

        self.setLayout(main_layout)

        reset_image_button.clicked.connect(self.on_reset_image)
        current_image_button.clicked.connect(self.set_current_image)
        fit_image_button.clicked.connect(self.image_editor.fit_image)

    def on_reset_image(self):
        self.image_scale_control.reset()
        self.image_x_pos_control.reset()
        self.image_y_pos_control.reset()
        self.image_rotation_control.reset()

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
                self.clear_image()
                self.image_id = None
                self.image_path = path
                pixmap = QPixmap(self.image_path)
                self.image_editor.set_pixmap(pixmap)
                self.image_dropped.emit()

    def on_load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg)", options=options)
        if fileName:
            self.clear_image()
            self.image_id = None
            self.show_image(fileName)

    def show_image(self, path):
        pixmap = QPixmap(path)
        self.image_editor.set_pixmap(pixmap)

    def reset_values(self):
        self.image_scale_control.reset()
        self.image_x_pos_control.reset()
        self.image_y_pos_control.reset()
        self.image_rotation_control.reset()

    def update_image_position(self, x, y):
        self.image_x_pos_control.set_value(x)
        self.image_y_pos_control.set_value(y)

    def update_image_scale(self, scale):
        self.image_scale_control.set_value(scale)

    def set_image_parameters(self, image_id, scale, x, y, angle):
        self.image_id = image_id
        self.add_image_button.setText("Update image")

        self.image_scale_control.set_value(scale)
        self.image_editor.set_image_scale(scale)
        self.image_x_pos_control.set_value(x)
        self.image_editor.set_image_x(x)
        self.image_y_pos_control.set_value(y)
        self.image_editor.set_image_y(y)
        self.image_rotation_control.set_value(angle)
        self.image_editor.rotate_image(angle)

    def set_current_image(self):
        if self.image_viewer.pixmap_item is not None:
            pixmap = self.image_viewer.pixmap_item.pixmap()
            self.image_editor.set_pixmap(pixmap)

    def clear_image(self):
        self.add_image_button.setText("Add image")
        self.image_scale_control.reset()
        self.image_x_pos_control.reset()
        self.image_y_pos_control.reset()
        self.image_rotation_control.reset()
        self.image_editor.clear()

    def on_add_image(self):
        if self.image_editor.pixmap_item is not None:
            self.disable_add_image_button()
            QTimer.singleShot(10, self.image_added.emit)

    def enable_add_image_button(self):
        self.add_image_button.setStyleSheet(
            """
            QPushButton {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #234a92, stop: 1 #12314e);
            }
            QPushButton:hover {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #2c67b3, stop: 1 #173864);
            }
            QPushButton:pressed {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #357ad4, stop: 1 #1f5088);
            }             
            """
        )
        self.add_image_button.setEnabled(True)

    def disable_add_image_button(self):
        self.add_image_button.setStyleSheet(
            """
            QPushButton {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #333333, stop: 1 #1d1d1d);
            }
            """
        )

        self.add_image_button.setEnabled(False)
