import os
from datetime import datetime
from importlib.resources import files

from PIL import Image
from PyQt6.QtCore import QMimeData, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QImageReader
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from iartisanxl.buttons.transparent_button import TransparentButton
from iartisanxl.layouts.aspect_ratio_layout import AspectRatioLayout
from iartisanxl.modules.common.image.image_data_object import ImageDataObject
from iartisanxl.modules.common.image.image_editor import ImageEditor
from iartisanxl.modules.common.image.image_editor_layer import ImageEditorLayer
from iartisanxl.modules.common.image_control import ImageControl
from iartisanxl.modules.common.image_viewer_simple import ImageViewerSimple


class LayerWidget(QWidget):
    LINK_IMG = files("iartisanxl.theme.icons").joinpath("link.png")
    UNLINK_IMG = files("iartisanxl.theme.icons").joinpath("unlink.png")

    lock_changed = pyqtSignal(int, bool)

    def __init__(self, layer_id: int, name: str):
        super().__init__()

        self.layer_id = layer_id
        self.name = name
        self.lock = True
        self.lock_parent = None

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(3, 0, 0, 0)
        main_layout.setSpacing(0)

        self.layer_name_label = QLabel(self.name)
        main_layout.addWidget(self.layer_name_label, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.lock_button = TransparentButton(self.LINK_IMG, 25, 25)
        self.lock_button.clicked.connect(self.on_lock_clicked)
        main_layout.addWidget(self.lock_button, alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)

        self.setLayout(main_layout)

    def on_lock_clicked(self):
        self.lock = not self.lock
        self.lock_button.icon = self.LINK_IMG if self.lock else self.UNLINK_IMG
        self.lock_changed.emit(self.layer_id, self.lock)


class LayerListWidget(QListWidget):
    layers_reordered = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setDragDropMode(self.DragDropMode.InternalMove)

    def dropEvent(self, event):
        pre_move_index = self.currentRow()
        super().dropEvent(event)
        post_move_index = self.currentRow()

        if pre_move_index != post_move_index:
            layers_list = []

            for i in range(self.count()):
                item = self.item(i)
                widget = self.itemWidget(item)
                inverted_index = self.count() - 1 - i
                layers_list.append((widget.layer_id, inverted_index))

            self.layers_reordered.emit(layers_list)


class LayerManagerWidget(QWidget):
    layer_selected = pyqtSignal(int)
    layer_lock_changed = pyqtSignal(int, bool)
    layers_reordered = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setMaximumWidth(150)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(2)

        self.list_widget = LayerListWidget()
        self.list_widget.currentItemChanged.connect(self.handle_item_selected)
        self.list_widget.layers_reordered.connect(self.on_layers_reordered)
        self.list_widget.setSpacing(0)
        main_layout.addWidget(self.list_widget)

        self.setLayout(main_layout)

    def add_layer(self, layer_id: int, name: str):
        item = QListWidgetItem()
        widget = LayerWidget(layer_id, name)
        item.setSizeHint(widget.sizeHint())
        widget.lock_changed.connect(self.on_lock_changed)

        self.list_widget.insertItem(0, item)
        self.list_widget.setItemWidget(item, widget)
        self.list_widget.setCurrentItem(item)

    def delete_layer(self, layer_id: int):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            widget = self.list_widget.itemWidget(item)
            if widget.layer_id == layer_id:
                self.list_widget.takeItem(i)
                break

    def get_layer_name(self, layer_id):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            widget = self.list_widget.itemWidget(item)
            if widget.layer_id == layer_id:
                return widget.name
        return None

    def handle_item_selected(self, current_item, previous_item):
        if current_item is not None:
            widget = self.list_widget.itemWidget(current_item)
            self.layer_selected.emit(widget.layer_id)

    def on_lock_changed(self, layer_id: int, locked: bool):
        self.layer_lock_changed.emit(layer_id, locked)

    def on_layers_reordered(self, layers: list):
        self.layers_reordered.emit(layers)


class ImageWidget(QWidget):
    ADD_LAYER_IMG = files("iartisanxl.theme.icons").joinpath("add_layer.png")
    DELETE_LAYER_IMG = files("iartisanxl.theme.icons").joinpath("delete_layer.png")

    image_loaded = pyqtSignal()
    image_changed = pyqtSignal()
    widget_updated = pyqtSignal()

    def __init__(
        self,
        text: str,
        prefix: str,
        image_viewer: ImageViewerSimple,
        editor_width: int,
        editor_height: int,
        show_layer_manager: bool = False,
        layer_manager_to_right: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.text = text
        self.image_viewer = image_viewer

        self.setAcceptDrops(True)

        self.prefix = prefix
        self.editor_width = editor_width
        self.editor_height = editor_height
        self.aspect_ratio = float(self.editor_width) / float(self.editor_height)

        self.image_copy_thread = None

        self.show_layer_manager = show_layer_manager
        self.layer_manager_to_right = layer_manager_to_right

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        top_layout = QHBoxLayout()
        source_text_label = QLabel(self.text)
        top_layout.addWidget(source_text_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.fit_image_button = QPushButton("Fit")
        top_layout.addWidget(self.fit_image_button)
        self.current_image_button = QPushButton("Current")
        self.current_image_button.setToolTip("Copy the current generated image an set it as an image.")
        top_layout.addWidget(self.current_image_button)
        load_image_button = QPushButton("Load")
        load_image_button.setToolTip("Load an image from your computer.")
        load_image_button.clicked.connect(self.on_load_image)
        top_layout.addWidget(load_image_button)
        main_layout.addLayout(top_layout)

        middle_layout = QHBoxLayout()
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
        middle_layout.addWidget(image_widget, 4)

        # set layer manager to the corresponding side
        if self.show_layer_manager:
            layers_layout = QVBoxLayout()
            self.layer_manager_widget = LayerManagerWidget()
            self.layer_manager_widget.layer_selected.connect(self.on_layer_selected)
            self.layer_manager_widget.layers_reordered.connect(self.image_editor.edit_all_layers_order)
            self.layer_manager_widget.layer_lock_changed.connect(self.on_layer_lock_changed)
            layers_layout.addWidget(self.layer_manager_widget)

            layers_controls_layout = QHBoxLayout()
            self.add_layer_button = TransparentButton(self.ADD_LAYER_IMG, 28, 28)
            self.add_layer_button.setObjectName("bottom_layer_control")
            self.add_layer_button.clicked.connect(self.on_add_layer_clicked)
            layers_controls_layout.addWidget(self.add_layer_button, alignment=Qt.AlignmentFlag.AlignLeft)
            self.delete_layer_button = TransparentButton(self.DELETE_LAYER_IMG, 28, 28)
            self.delete_layer_button.clicked.connect(self.on_delete_layer_clicked)
            self.delete_layer_button.setObjectName("bottom_layer_control")
            layers_controls_layout.addWidget(self.delete_layer_button, alignment=Qt.AlignmentFlag.AlignRight)
            layers_layout.addLayout(layers_controls_layout)

            if self.layer_manager_to_right:
                middle_layout.addLayout(layers_layout, 2)
            else:
                middle_layout.insertLayout(0, layers_layout, 2)

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
        self.image_scale_control = ImageControl("Scale: ", 1.0, 3)
        self.image_scale_control.value_changed.connect(self.on_image_scale)
        image_controls_layout.addWidget(self.image_scale_control)
        self.image_x_pos_control = ImageControl("X Pos: ", 0, 0)
        self.image_x_pos_control.value_changed.connect(self.on_set_image_x)
        image_controls_layout.addWidget(self.image_x_pos_control)
        self.image_y_pos_control = ImageControl("Y Pos: ", 0, 0)
        self.image_y_pos_control.value_changed.connect(self.on_set_image_y)
        image_controls_layout.addWidget(self.image_y_pos_control)
        self.image_rotation_control = ImageControl("Rotation: ", 0, 0)
        self.image_rotation_control.value_changed.connect(self.on_rotate_image)
        image_controls_layout.addWidget(self.image_rotation_control)
        main_layout.addLayout(image_controls_layout)

        main_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding))

        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 1)
        main_layout.setStretch(2, 0)
        main_layout.setStretch(3, 0)

        self.setLayout(main_layout)

        self.fit_image_button.clicked.connect(self.image_editor.fit_image)
        self.current_image_button.clicked.connect(self.set_current_image)
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
        selected_path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg)", options=options)
        if selected_path:
            self.set_image(selected_path)

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

    def set_image(self, path: str):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{self.prefix}_{timestamp}_{self.image_editor.selected_layer_id}_original.png"
        image_path = os.path.join("tmp/", filename)
        pil_image = Image.open(path)
        pil_image.save(image_path, format="PNG")
        self.image_editor.set_image(image_path)
        self.reset_controls()

        self.image_loaded.emit()

    def reload_image_layer(self, image_path: str, original_path: str, order: int):
        layer_id = self.image_editor.reload_image_layer(image_path, original_path, order)
        return layer_id

    def set_current_image(self):
        if self.image_viewer.pixmap_item is not None:
            self.clear_image()

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{self.prefix}_{timestamp}_original.png"
            filepath = os.path.join("tmp/", filename)

            pixmap = self.image_viewer.pixmap_item.pixmap()
            pixmap.save(filepath)

            self.set_image(filepath)

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
        self.image_editor.clear_all()

    def on_image_changed(self):
        self.image_changed.emit()

    def on_image_copy(self):
        layer = self.image_editor.get_selected_layer()

        if layer.image_path is not None:
            self.prepare_copy_thread(layer)
            self.image_copy_thread.image_done.connect(self.on_copy_image_done)
            self.image_copy_thread.start()

    def prepare_copy_thread(self, layer: ImageEditorLayer, save_path: str = None):
        image_data = ImageDataObject()
        image_data.image_original = layer.image_path
        image_data.image_scale = self.image_scale_control.value
        image_data.image_x_pos = self.image_x_pos_control.value
        image_data.image_y_pos = self.image_y_pos_control.value
        image_data.image_rotation = self.image_rotation_control.value

        # drawings_layer = self.image_editor.layer_manager.get_layer_by_id(self.drawing_layer_id)
        # drawings_pixmap = drawings_layer.pixmap_item.pixmap()

        # self.image_copy_thread = SaveMergedImageThread(
        #     self.editor_width, self.editor_height, image_data, drawings_pixmap, save_path=save_path
        # )
        # self.image_copy_thread.finished.connect(self.on_copy_thread_finished)

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

    def set_enabled(self, enabled: bool):
        self.image_editor.setEnabled(enabled)

        self.fit_image_button.setEnabled(enabled)
        self.reset_image_button.setEnabled(enabled)
        self.reset_view_button.setEnabled(enabled)

        self.image_scale_control.setEnabled(enabled)
        self.image_x_pos_control.setEnabled(enabled)
        self.image_y_pos_control.setEnabled(enabled)
        self.image_rotation_control.setEnabled(enabled)

    def set_erase_mode(self, value: bool):
        self.image_editor.erasing = value
