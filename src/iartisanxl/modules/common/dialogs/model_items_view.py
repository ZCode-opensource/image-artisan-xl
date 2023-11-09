import os
import glob
import logging
import shutil

from io import BytesIO
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QScrollArea,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QMenu,
    QLineEdit,
    QFileDialog,
)
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QAction

from iartisanxl.layouts.flow_layout import FlowLayout
from iartisanxl.threads.model_items_loader_thread import ModelItemsLoaderThread
from iartisanxl.modules.common.model_item import ModelItem
from iartisanxl.modules.common.item_selector import ItemSelector
from iartisanxl.modules.common.drop_lightbox import DropLightBox


class ModelItemsView(QWidget):
    model_item_clicked = pyqtSignal(dict)
    finished_loading = pyqtSignal()
    item_imported = pyqtSignal(str)

    def __init__(self, directories: tuple, default_image: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.thumb_width = 150
        self.thumb_height = 150
        self.directories = directories
        self.default_image = default_image
        self.model_items_loader_thread = None
        self.loading_images = False
        self.tags = None
        self.logger = logging.getLogger()

        self.setAcceptDrops(True)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        filters_layout = QHBoxLayout()
        tag_filter_label = QLabel("Tags:")
        filters_layout.addWidget(tag_filter_label)
        self.tags_selector = ItemSelector()
        self.tags_selector.item_changed.connect(self.on_filter_changed)
        filters_layout.addWidget(self.tags_selector)
        name_filter_label = QLabel("Name:")
        filters_layout.addWidget(name_filter_label)
        self.name_line_edit = QLineEdit()
        self.name_line_edit.textChanged.connect(self.on_filter_changed)
        filters_layout.addWidget(self.name_line_edit)
        self.order_combo_box = QComboBox()
        self.order_combo_box.addItem("Ascending", "asc")
        self.order_combo_box.addItem("Descending", "desc")
        self.order_combo_box.currentIndexChanged.connect(self.change_order_direction)
        filters_layout.addWidget(self.order_combo_box)
        reload_button = QPushButton("Reload")
        reload_button.clicked.connect(self.on_reload_items)
        filters_layout.addWidget(reload_button)
        import_button = QPushButton("Import")
        import_button.clicked.connect(self.on_import_model)
        filters_layout.addWidget(import_button)

        filters_layout.setStretch(0, 0)
        filters_layout.setStretch(1, 4)
        filters_layout.setStretch(2, 1)
        filters_layout.setStretch(3, 3)
        filters_layout.setStretch(4, 1)
        filters_layout.setStretch(5, 1)
        filters_layout.setStretch(6, 1)

        main_layout.addLayout(filters_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.flow_widget = QWidget()
        self.flow_widget.setObjectName("flow_widget")
        self.flow_layout = FlowLayout(self.flow_widget)
        scroll_area.setWidget(self.flow_widget)
        main_layout.addWidget(scroll_area)

        self.drop_lightbox = DropLightBox(self)
        self.drop_lightbox.setText("Drop file here")

    def load_items(self):
        self.model_items_loader_thread = None
        self.flow_layout.clear()
        model_files = []
        self.tags = []

        for directory in self.directories:
            if directory["type"] == "diffusers":
                if directory["path"] and os.path.isdir(directory["path"]):
                    for dirname in os.listdir(directory["path"]):
                        dirpath = os.path.join(directory["path"], dirname)
                        if os.path.isdir(dirpath):
                            data = {
                                "root_filename": dirname,
                                "filepath": dirpath,
                                "type": "diffusers",
                            }
                            model_files.append(data)
            else:
                for filepath in glob.iglob(
                    os.path.join(directory["path"], "*.safetensors")
                ):
                    filename = os.path.basename(filepath)
                    root_filename, _ = os.path.splitext(filename)

                    data = {
                        "root_filename": root_filename,
                        "filepath": filepath,
                        "type": "safetensors",
                    }
                    model_files.append(data)

        model_files.sort(key=lambda x: x["root_filename"])
        self.load_items_thread(model_files)

    def load_items_thread(self, model_files: list):
        self.model_items_loader_thread = ModelItemsLoaderThread(
            model_files,
            self.thumb_width,
            self.thumb_height,
            self.default_image,
        )
        self.model_items_loader_thread.model_item_loaded.connect(self.add_model_item)
        self.model_items_loader_thread.finished.connect(self.on_loading_finished)
        self.model_items_loader_thread.start()

    def add_single_item_from_path(self, filepath: str, item_type: str):
        filename = os.path.basename(filepath)
        root_filename, _ = os.path.splitext(filename)
        data = {
            "root_filename": root_filename,
            "filepath": filepath,
            "type": item_type,
        }
        self.load_items_thread([data])

    def add_model_item(self, data: dict, buffer: BytesIO):
        model_item = ModelItem(data, buffer, self.thumb_width, self.thumb_height)
        model_item.clicked.connect(
            lambda: self.model_item_clicked.emit(model_item.data)
        )
        self.flow_layout.addWidget(model_item)

        if data.get("tags") is not None:
            tags = data["tags"].split(", ")
            self.tags.extend(tags)

    def on_import_model(self):
        dialog = QFileDialog()
        options = QFileDialog.Option.ReadOnly | QFileDialog.Option.HideNameFilterDetails
        dialog.setOptions(options)

        filepath, _ = dialog.getOpenFileName(
            None, "Select a a model", "", "*.safetensors", options=options
        )
        self.item_imported.emit(filepath)

    def on_loading_finished(self):
        tags = list(set(self.tags))
        tags.sort()
        self.tags_selector.add_items(tags)

        self.loading_images = False
        self.finished_loading.emit()

    def on_reload_items(self):
        self.tags_selector.clear_selected_items()
        self.name_line_edit.setText("")
        self.order_combo_box.setCurrentIndex(0)
        self.load_items()

    def change_order_direction(self):
        self.flow_layout.sort_direction = self.order_combo_box.currentData()
        self.flow_layout.order_by()

    def contextMenuEvent(self, event):
        pos = self.flow_widget.mapFrom(self, event.pos())
        item = self.flow_layout.itemAtPosition(pos)

        if item is not None:
            context_menu = QMenu(self)
            delete_action: QAction | None = context_menu.addAction("Delete")
            delete_action.triggered.connect(lambda: self.on_delete_item(item.widget()))
            context_menu.exec(event.globalPos())

    def on_delete_item(self, widget: ModelItem):
        model_type = widget.data.get("type")

        if model_type is not None:
            filepath = widget.data.get("filepath")

            if model_type == "diffusers":
                shutil.rmtree(filepath)
                self.flow_layout.remove_item(widget)
            else:
                os.remove(filepath)
                self.flow_layout.remove_item(widget)

    def on_filter_changed(self):
        tags = None
        if len(self.tags_selector.line_edit.text()) > 0:
            tags = self.tags_selector.line_edit.text().split(", ")
        name = self.name_line_edit.text()
        self.flow_layout.set_filters(tags, name)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_lightbox.show()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.drop_lightbox.hide()
        event.accept()

    def dropEvent(self, event):
        self.drop_lightbox.hide()

        for url in event.mimeData().urls():
            path = url.toLocalFile()
            self.item_imported.emit(path)
