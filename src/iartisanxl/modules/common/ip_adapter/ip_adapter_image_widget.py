import os
from datetime import datetime

from PIL import Image
from PyQt6.QtCore import QMimeData, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QImageReader
from PyQt6.QtWidgets import QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from superqt import QLabeledDoubleSlider

from iartisanxl.layouts.aspect_ratio_layout import AspectRatioLayout
from iartisanxl.modules.common.image.image_editor import ImageEditor
from iartisanxl.modules.common.image.layer_manager_widget import LayerManagerWidget
from iartisanxl.modules.common.image_control import ImageControl
from iartisanxl.modules.common.image_viewer_simple import ImageViewerSimple
from iartisanxl.threads.image.save_merged_image_thread import SaveMergedImageThread


class IPAdapterImageWidget(QWidget):
    image_loaded = pyqtSignal()
    image_changed = pyqtSignal()
    widget_updated = pyqtSignal()
    image_added = pyqtSignal()
    new_image = pyqtSignal()

    def __init__(
        self, text: str, image_viewer: ImageViewerSimple, save_directory: str, target_width: int, target_height: int
    ):
        super().__init__()

        self.setObjectName("ip_adapter_image_widget")
        self.text = text
        self.image_viewer = image_viewer
        self.image_path = None
        self.save_directory = save_directory
        self.image_id = None

        self.setAcceptDrops(True)

        self.target_width = target_width
        self.target_height = target_height
        self.aspect_ratio = float(self.target_width) / float(self.target_height)

        self.image_copy_thread = None

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        top_layout = QHBoxLayout()
        fit_image_button = QPushButton("Fit")
        top_layout.addWidget(fit_image_button)
        current_image_button = QPushButton("Current")
        top_layout.addWidget(current_image_button)
        load_image_button = QPushButton("Load")
        load_image_button.clicked.connect(self.on_load_image)
        top_layout.addWidget(load_image_button)
        main_layout.addLayout(top_layout)

        middle_layout = QHBoxLayout()
        image_widget = QWidget()
        self.image_editor = ImageEditor(self.target_width, self.target_height, self.aspect_ratio, self.save_directory)
        self.image_editor.image_changed.connect(self.on_image_changed)
        self.image_editor.image_scaled.connect(self.update_image_scale)
        self.image_editor.image_moved.connect(self.update_image_position)
        self.image_editor.image_rotated.connect(self.update_image_angle)
        self.image_editor.image_pasted.connect(self.set_image)
        self.image_editor.image_copy.connect(self.on_image_copy)
        self.image_editor.image_save.connect(self.on_image_copy)
        editor_layout = AspectRatioLayout(image_widget, self.aspect_ratio)
        editor_layout.addWidget(self.image_editor)
        middle_layout.addWidget(image_widget, 4)

        layers_layout = QVBoxLayout()
        self.layer_manager_widget = LayerManagerWidget(True)
        self.layer_manager_widget.layer_selected.connect(self.on_layer_selected)
        self.layer_manager_widget.layers_reordered.connect(self.image_editor.edit_all_layers_order)
        self.layer_manager_widget.layer_lock_changed.connect(self.on_layer_lock_changed)
        self.layer_manager_widget.add_layer_clicked.connect(self.on_add_layer_clicked)
        self.layer_manager_widget.delete_layer_clicked.connect(self.on_delete_layer_clicked)
        layers_layout.addWidget(self.layer_manager_widget)
        middle_layout.addLayout(layers_layout, 0)
        main_layout.addLayout(middle_layout)

        image_bottom_layout = QHBoxLayout()
        self.reset_image_button = QPushButton("Reset Layer")
        self.reset_image_button.setToolTip("Reset zoom and position of the image from the last update.")
        image_bottom_layout.addWidget(self.reset_image_button)
        self.reset_view_button = QPushButton("Reset view")
        self.reset_view_button.setToolTip("Reset zoom and position of the viewport.")
        image_bottom_layout.addWidget(self.reset_view_button)
        main_layout.addLayout(image_bottom_layout)

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
        self.image_weight_slider.valueChanged.connect(self.on_image_changed)
        image_actions_layout.addWidget(self.image_weight_slider)

        image_noise_label = QLabel("Noise:")
        image_actions_layout.addWidget(image_noise_label)
        self.image_noise_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.image_noise_slider.setRange(0.0, 1.0)
        self.image_noise_slider.setValue(0.0)
        self.image_noise_slider.valueChanged.connect(self.on_image_changed)
        image_actions_layout.addWidget(self.image_noise_slider)
        self.noise_type_combo = QComboBox()
        self.noise_type_combo.addItem("Default noise", "default")
        self.noise_type_combo.addItem("Mandelbrot noise", "mandelbrot")
        self.noise_type_combo.addItem("Perlin noise", "perlin")
        self.noise_type_combo.addItem("Simplex noise", "simplex")
        self.noise_type_combo.addItem("Uniform noise", "uniform")
        self.noise_type_combo.addItem("Gaussian noise", "gaussian")
        self.noise_type_combo.currentIndexChanged.connect(self.on_image_changed)
        image_actions_layout.addWidget(self.noise_type_combo)

        image_actions_layout.setStretch(0, 0)
        image_actions_layout.setStretch(1, 1)
        image_actions_layout.setStretch(2, 0)
        image_actions_layout.setStretch(3, 1)
        image_actions_layout.setStretch(4, 0)
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

        fit_image_button.clicked.connect(self.image_editor.fit_image)
        current_image_button.clicked.connect(self.set_current_image)
        self.reset_image_button.clicked.connect(self.image_editor.clear_and_restore)
        self.reset_view_button.clicked.connect(self.image_editor.reset_view)

    def update_image_scale(self, scale: float):
        self.image_scale_control.set_value(scale)
        self.widget_updated.emit()

    def update_image_position(self, x, y):
        self.image_x_pos_control.set_value(x)
        self.image_y_pos_control.set_value(y)
        self.widget_updated.emit()

    def update_image_angle(self, angle):
        self.image_rotation_control.set_value(angle)
        self.widget_updated.emit()

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
        path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg)", options=options)
        if path:
            self.set_image(path)

    def on_layer_selected(self, layer_id: int):
        self.image_editor.selected_layer_id = layer_id
        layer = self.image_editor.get_selected_layer()
        self.update_image_position(layer.pixmap_item.x(), layer.pixmap_item.y())

    def on_layer_lock_changed(self, layer_id: int, locked: bool):
        self.image_editor.set_layer_locked(layer_id, locked)

    def on_add_layer_clicked(self):
        self.add_empty_layer()

    def add_empty_layer(self, name: str = None):
        layer_id = self.image_editor.add_empty_layer()
        layer_name = name if name else f"Layer {layer_id}"
        self.layer_manager_widget.add_layer(layer_id, layer_name)
        self.image_editor.selected_layer_id = layer_id

    def on_delete_layer_clicked(self):
        self.image_editor.delete_layer()
        self.layer_manager_widget.delete_layer(self.image_editor.selected_layer_id)

    def set_image(self, path):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"ip_adapter_{timestamp}_{self.image_editor.selected_layer_id}_original.png"
        image_path = os.path.join("tmp/", filename)
        pil_image = Image.open(path)
        pil_image.save(image_path, format="PNG")
        self.image_editor.set_image(image_path)
        self.add_image_button.setEnabled(True)
        self.new_image_button.setEnabled(True)

        self.image_loaded.emit()

    def on_add_image(self):
        if self.image_editor.layer_manager.layers[0] is not None:
            self.add_image_button.setEnabled(False)
            QTimer.singleShot(10, self.image_added.emit)

    def on_new_image(self):
        self.clear_image()
        self.add_image_button.setText("Add image")
        self.add_image_button.setDisabled(True)
        self.delete_image_button.setDisabled(True)
        self.new_image_button.setDisabled(True)
        self.add_empty_layer()
        self.new_image.emit()

    def reload_image_layer(self, image_path: str, original_path: str, order: int):
        layer_id = self.image_editor.reload_image_layer(image_path, original_path, order)
        return layer_id

    def set_current_image(self):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"ip_adapter_{timestamp}_{self.image_editor.selected_layer_id}_original.png"
        filepath = os.path.join("tmp/", filename)

        pixmap = self.image_viewer.pixmap_item.pixmap()
        pixmap.save(filepath, format="PNG")
        self.image_editor.set_image(filepath)
        self.reset_controls()

        self.image_loaded.emit()

    def on_image_scale(self, value: float):
        self.image_editor.set_image_scale(value)
        self.widget_updated.emit()

    def on_set_image_x(self, value: int):
        self.image_editor.set_image_x(value)
        self.widget_updated.emit()

    def on_set_image_y(self, value: int):
        self.image_editor.set_image_y(value)
        self.widget_updated.emit()

    def on_rotate_image(self, value: float):
        self.image_editor.rotate_image(value)
        self.widget_updated.emit()

    def set_layer_parameters(self, image_layer_id, scale, x, y, angle):
        self.image_editor.selected_layer_id = image_layer_id

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
        self.image_weight_slider.setValue(1.0)
        self.image_noise_slider.setValue(0.0)
        self.image_editor.clear_all()

    def on_image_changed(self):
        self.add_image_button.setEnabled(True)
        self.image_changed.emit()

    def on_image_copy(self, save_path: str = None):
        layers = self.image_editor.get_all_layers()

        if len(layers) == 0:
            return

        self.image_copy_thread = SaveMergedImageThread(layers, self.target_width, self.target_height, save_path)
        self.image_copy_thread.image_done.connect(self.on_copy_image_done)
        self.image_copy_thread.start()

    def on_copy_thread_finished(self):
        self.image_copy_thread = None

    def on_copy_image_done(self, image_path):
        if image_path is not None:
            clipboard = QGuiApplication.clipboard()
            mime_data = QMimeData()
            mime_data.setUrls([QUrl.fromLocalFile(image_path)])
            clipboard.setMimeData(mime_data)

    def set_erase_mode(self, value: bool):
        self.image_editor.erasing = value
