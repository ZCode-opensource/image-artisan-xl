import os
import shutil

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QMenu
from PyQt6.QtGui import QPixmap, QAction
from PyQt6.QtCore import pyqtSignal, Qt

from iartisanxl.layouts.simple_flow_layout import SimpleFlowLayout
from iartisanxl.threads.dataset_items_loader_thread import DatasetItemsLoaderThread
from iartisanxl.modules.dataset.dataset_item import DatasetItem
from iartisanxl.modules.common.drop_lightbox import DropLightBox


class DatasetItemsView(QWidget):
    finished_loading = pyqtSignal()
    item_selected = pyqtSignal()
    items_changed = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, thumb_width: int, thumb_height: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.thumb_width = thumb_width
        self.thumb_height = thumb_height
        self.path = None
        self.dataset_items_loader_thread = None
        self.selected_path = None
        self.current_item = None
        self.current_item_index = None
        self.item_count = None
        self.originals_dir = None

        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.flow_widget = QWidget()
        self.flow_widget.setObjectName("flow_widget")
        self.flow_layout = SimpleFlowLayout(self.flow_widget)
        self.scroll_area.setWidget(self.flow_widget)
        main_layout.addWidget(self.scroll_area)

        self.drop_lightbox = DropLightBox(self)
        self.drop_lightbox.setText("Drop file here")

    def load_items(self, path):
        self.selected_path = None
        self.current_item = None
        self.current_item_index = None
        self.dataset_items_loader_thread = None
        images = []

        self.flow_layout.clear()
        self.path = path

        for image_file in os.listdir(self.path):
            if image_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                images.append(image_file)

        self.load_dataset_images_thread(images)

    def load_dataset_images_thread(self, images):
        self.dataset_items_loader_thread = DatasetItemsLoaderThread(images, self.path, self.thumb_width, self.thumb_height)
        self.dataset_items_loader_thread.image_loaded.connect(self.add_item)
        self.dataset_items_loader_thread.finished.connect(self.on_loading_finished)
        self.dataset_items_loader_thread.start()

    def add_item(self, path: str, thumbnail: QPixmap):
        dataset_item = DatasetItem(self.thumb_width, self.thumb_height, path, thumbnail)
        dataset_item.clicked.connect(self.on_item_selected)
        self.flow_layout.addWidget(dataset_item)
        self.item_count = self.flow_layout.count()

        if self.selected_path is None:
            self.selected_path = path
            self.current_item_index = 0
            self.current_item = dataset_item
            dataset_item.set_selected(True)

    def on_loading_finished(self):
        self.item_count = self.flow_layout.count()
        self.finished_loading.emit()

    def on_item_selected(self, item: DatasetItem):
        for i in range(self.flow_layout.count()):
            widget: DatasetItem = self.flow_layout.itemAt(i).widget()

            if widget == item:
                self.current_item_index = i
                self.current_item = widget
                self.selected_path = item.path
                self.item_selected.emit()
            else:
                widget.set_selected(False)

        self.setFocus()

    def get_first_item(self):
        self.current_item_index = 0
        item: DatasetItem = self.flow_layout.itemAt(self.current_item_index).widget()

        if item is not None:
            self.current_item.set_selected(False)
            self.current_item = item
            self.selected_path = item.path
            item.set_selected(True)
            self.item_selected.emit()
            self.scroll_area.ensureWidgetVisible(item)

        return item

    def get_prev_item(self):
        if self.current_item_index > 0:
            self.current_item_index -= 1
            item: DatasetItem = self.flow_layout.itemAt(self.current_item_index).widget()

            if item is not None:
                self.current_item.set_selected(False)
                self.current_item = item
                self.selected_path = item.path
                item.set_selected(True)
                self.item_selected.emit()
                self.scroll_area.ensureWidgetVisible(item)
            return item
        return None

    def get_next_item(self):
        if self.current_item_index < self.flow_layout.count() - 1:
            self.current_item_index += 1
            item: DatasetItem = self.flow_layout.itemAt(self.current_item_index).widget()

            if item is not None:
                self.current_item.set_selected(False)
                self.current_item = item
                self.selected_path = item.path
                item.set_selected(True)
                self.item_selected.emit()
                self.scroll_area.ensureWidgetVisible(item)
            return item
        return None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Left:
            self.get_prev_item()
        elif event.key() == Qt.Key.Key_Right:
            self.get_next_item()

    def update_current_item(self, path: str, thumbnail: QPixmap):
        self.current_item.set_image(path, thumbnail)

    def contextMenuEvent(self, event):
        pos = self.flow_widget.mapFrom(self, event.pos())
        item = self.flow_layout.itemAtPosition(pos)

        if item is not None:
            context_menu = QMenu(self)
            delete_action: QAction | None = context_menu.addAction("Delete")
            delete_action.triggered.connect(lambda: self.on_delete_item(item.widget()))
            duplicate_action: QAction | None = context_menu.addAction("Duplicate")
            duplicate_action.triggered.connect(lambda: self.on_duplicate_item(item.widget()))
            context_menu.exec(event.globalPos())

    def on_delete_item(self, item: DatasetItem):
        delete_index = self.flow_layout.index_of(item)
        captions_file = os.path.splitext(item.path)[0] + ".txt"

        filename = os.path.basename(item.path)
        name = os.path.splitext(filename)[0]
        original_path = os.path.join(self.originals_dir, filename)
        json_path = os.path.join(self.originals_dir, f"{name}.json")

        os.remove(item.path)

        if os.path.isfile(original_path):
            os.remove(original_path)

        if os.path.isfile(json_path):
            os.remove(json_path)

        if os.path.isfile(captions_file):
            os.remove(captions_file)

        self.flow_layout.remove_item(item)
        self.item_count = self.flow_layout.count()

        if self.current_item_index == delete_index:
            if self.current_item_index > self.item_count - 1:
                self.current_item_index = self.current_item_index - 1

            if self.current_item_index > 0:
                item: DatasetItem = self.flow_layout.itemAt(self.current_item_index).widget()
                self.current_item = item
                self.selected_path = item.path
            else:
                self.current_item = None
                self.selected_path = None
                self.current_item_index = None

        self.items_changed.emit()

    def on_duplicate_item(self, item: DatasetItem):
        index = self.flow_layout.index_of(item)

        path = item.path
        filename = os.path.basename(path)
        name, extension = os.path.splitext(filename)
        new_name = f"{name}_{index + 1}"

        # copy the original if it has one
        original_path = os.path.join(self.originals_dir, filename)
        new_original_path = os.path.join(self.originals_dir, f"{new_name}.{extension}")
        shutil.copy2(original_path, new_original_path)

        # copy the dataset image
        new_image_path = os.path.join(self.path, f"{new_name}.{extension}")
        shutil.copy2(path, new_image_path)

        if self.current_item is not None:
            self.current_item.set_selected(False)
            self.current_item = None
            self.selected_path = None
            self.current_item_index = None

        self.add_item(new_image_path, item.pixmap)

        self.item_count = self.flow_layout.count()
        self.items_changed.emit()

    def clear_selection(self):
        if self.current_item is not None:
            self.current_item.set_selected(False)
            self.current_item = None
            self.selected_path = None
            self.current_item_index = None
