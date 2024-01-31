import os
from datetime import datetime

from PIL import Image
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog, QLabel
from PyQt6.QtGui import QImageReader, QPixmap, QGuiApplication
from PyQt6.QtCore import pyqtSignal, QTimer, Qt, QUrl, QMimeData
from superqt import QLabeledDoubleSlider

from iartisanxl.modules.common.image.image_adder_preview import ImageAdderPreview
from iartisanxl.modules.common.image_viewer_simple import ImageViewerSimple
from iartisanxl.modules.common.image_control import ImageControl
from iartisanxl.layouts.aspect_ratio_layout import AspectRatioLayout
from iartisanxl.threads.save_merged_image_thread import SaveMergedImageThread
from iartisanxl.modules.common.image.image_data_object import ImageDataObject


class IPAdapterImageWidget(QWidget):
    image_added = pyqtSignal()

    def __init__(self, text: str, image_viewer: ImageViewerSimple, save_directory: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setObjectName("ip_adapter_image_widget")
        self.text = text
        self.image_viewer = image_viewer
        self.image_path = None
        self.save_directory = save_directory
        self.image_id = None

        self.setAcceptDrops(True)

        self.editor_width = 224
        self.editor_height = 224
        self.aspect_ratio = float(self.editor_width) / float(self.editor_height)

        self.image_copy_thread = None

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
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
        self.image_editor.image_updated.connect(self.on_image_updated)
        self.image_editor.image_moved.connect(self.update_image_position)
        self.image_editor.image_scaled.connect(self.update_image_scale)
        self.image_editor.image_pasted.connect(self.create_original_image)
        self.image_editor.image_copy.connect(self.on_image_copy)
        self.image_editor.image_save.connect(self.on_image_save)
        editor_layout = AspectRatioLayout(image_widget, self.aspect_ratio)
        editor_layout.addWidget(self.image_editor)
        main_layout.addWidget(image_widget)

        image_controls_layout = QHBoxLayout()
        image_controls_layout.setContentsMargins(0, 0, 0, 0)
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

        image_actions_layout = QHBoxLayout()
        image_actions_layout.setContentsMargins(0, 0, 0, 0)
        image_actions_layout.setSpacing(10)
        image_weight_label = QLabel("Image weight:")
        image_actions_layout.addWidget(image_weight_label)
        self.image_weight_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.image_weight_slider.setRange(0.0, 1.0)
        self.image_weight_slider.setValue(1.0)
        self.image_weight_slider.valueChanged.connect(self.on_image_updated)
        image_actions_layout.addWidget(self.image_weight_slider)

        image_noise_label = QLabel("Noise:")
        image_actions_layout.addWidget(image_noise_label)
        self.image_noise_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.image_noise_slider.setRange(0.0, 1.0)
        self.image_noise_slider.setValue(0.0)
        self.image_noise_slider.valueChanged.connect(self.on_image_updated)
        image_actions_layout.addWidget(self.image_noise_slider)

        image_actions_layout.setStretch(0, 0)
        image_actions_layout.setStretch(1, 1)
        image_actions_layout.setStretch(2, 0)
        image_actions_layout.setStretch(3, 1)
        main_layout.addLayout(image_actions_layout)

        image_bottom_actions_layout = QHBoxLayout()
        image_bottom_actions_layout.setContentsMargins(0, 5, 0, 5)
        self.delete_image_button = QPushButton("Delete image")
        self.delete_image_button.setObjectName("red_button")
        self.delete_image_button.setDisabled(True)
        image_bottom_actions_layout.addWidget(self.delete_image_button)
        self.new_image_button = QPushButton("New image")
        self.new_image_button.setObjectName("yellow_button")
        self.new_image_button.clicked.connect(self.on_new_image)
        self.new_image_button.setDisabled(True)
        image_bottom_actions_layout.addWidget(self.new_image_button)
        self.add_image_button = QPushButton("Add image")
        self.add_image_button.setObjectName("blue_button")
        self.add_image_button.clicked.connect(self.on_add_image)
        self.add_image_button.setDisabled(True)
        image_bottom_actions_layout.addWidget(self.add_image_button)
        main_layout.addLayout(image_bottom_actions_layout)

        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 1)
        main_layout.setStretch(2, 0)
        main_layout.setStretch(3, 0)
        main_layout.setStretch(4, 0)

        self.setLayout(main_layout)

        reset_image_button.clicked.connect(self.on_reset_image)
        current_image_button.clicked.connect(self.set_current_image)
        fit_image_button.clicked.connect(self.image_editor.fit_image)

    def on_reset_image(self):
        self.image_scale_control.reset()
        self.image_x_pos_control.reset()
        self.image_y_pos_control.reset()
        self.image_rotation_control.reset()
        self.image_weight_slider.setValue(1.0)
        self.image_noise_slider.setValue(0.0)

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
                self.show_image(path)

    def on_load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg)", options=options)
        if path:
            self.clear_image()
            self.show_image(path)

    def show_image(self, path):
        self.image_path = path
        pixmap = QPixmap(path)
        self.image_editor.set_pixmap(pixmap)
        self.add_image_button.setEnabled(True)
        self.new_image_button.setEnabled(True)

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

    def set_image_parameters(self, image_id, scale, x, y, angle, weight, noise):
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
        self.image_weight_slider.setValue(weight)
        self.image_noise_slider.setValue(noise)

    def set_current_image(self):
        if self.image_viewer.pixmap_item is not None:
            pixmap = self.image_viewer.pixmap_item.pixmap()
            self.image_editor.set_pixmap(pixmap)

    def clear_image(self):
        self.image_scale_control.reset()
        self.image_x_pos_control.reset()
        self.image_y_pos_control.reset()
        self.image_rotation_control.reset()
        self.image_weight_slider.setValue(1.0)
        self.image_noise_slider.setValue(0.0)
        self.image_editor.clear()

    def on_add_image(self):
        if self.image_editor.pixmap_item is not None:
            self.add_image_button.setEnabled(False)
            QTimer.singleShot(10, self.image_added.emit)

    def on_image_updated(self):
        self.add_image_button.setEnabled(True)

    def on_new_image(self):
        self.image_id = None
        self.image_path = None
        self.clear_image()
        self.add_image_button.setText("Add image")
        self.add_image_button.setDisabled(True)
        self.delete_image_button.setDisabled(True)
        self.new_image_button.setDisabled(True)

    def set_editor_image_by_path(self, image_path):
        self.image_editor.set_image(image_path)
        self.image_path = image_path

    def create_original_image(self, path):
        self.clear_image()

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"ip_{timestamp}_original.png"
        image_path = os.path.join("tmp/", filename)
        pil_image = Image.open(path)
        pil_image.save(image_path, format="PNG")

        self.set_editor_image_by_path(image_path)

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

        self.image_copy_thread = SaveMergedImageThread(self.editor_width, self.editor_height, image_data, None, save_path=save_path)
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
