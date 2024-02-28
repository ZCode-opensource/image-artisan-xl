import os
from io import BytesIO

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QImage, QPixmap
from PyQt6.QtWidgets import QMenu, QScrollArea, QVBoxLayout, QWidget

from iartisanxl.layouts.simple_flow_layout import SimpleFlowLayout
from iartisanxl.modules.common.drop_lightbox import DropLightBox
from iartisanxl.modules.common.ip_adapter.ip_adapter_data_object import IPAdapterDataObject
from iartisanxl.modules.common.ip_adapter.ip_adapter_image import IPAdapterImage
from iartisanxl.modules.common.ip_adapter.ip_adapter_image_item import IPAdapterImageItem
from iartisanxl.threads.images_loader_thread import ImagesLoaderThread
from iartisanxl.utilities.image.operations import remove_image_data_files


class IpAdapterImageItemsView(QWidget):
    finished_loading = pyqtSignal()
    item_selected = pyqtSignal(IPAdapterImage)
    item_deleted = pyqtSignal(IPAdapterImage, bool)
    error = pyqtSignal(str)

    def __init__(self, ip_adapter_data: IPAdapterDataObject, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ip_adapter_data = ip_adapter_data
        self.dataset_items_loader_thread = None
        self.image_data = None
        self.current_item = None
        self.current_item_index = None
        self.item_count = None

        self.thumb_width = 80
        self.thumb_height = 80

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

    def load_items(self):
        if self.ip_adapter_data.images is not None and len(self.ip_adapter_data.images) > 0:
            self.load_images_thread(self.ip_adapter_data.images)

    def load_images_thread(self, images):
        self.dataset_items_loader_thread = ImagesLoaderThread(images)
        self.dataset_items_loader_thread.image_loaded.connect(self.add_item)
        self.dataset_items_loader_thread.finished.connect(self.on_loading_finished)
        self.dataset_items_loader_thread.start()

    def add_item(self, buffer: BytesIO, ip_adapter_image: IPAdapterImage):
        qimage = QImage.fromData(buffer.getvalue())
        pixmap = QPixmap.fromImage(qimage)

        dataset_item = IPAdapterImageItem(pixmap, ip_adapter_image=ip_adapter_image)
        dataset_item.clicked.connect(self.on_item_selected)
        self.flow_layout.addWidget(dataset_item)

    def add_item_data_object(self, ip_adapter_image: IPAdapterImage):
        pixmap = QPixmap(ip_adapter_image.thumb)
        image_item = IPAdapterImageItem(pixmap, ip_adapter_image=ip_adapter_image)
        image_item.clicked.connect(self.on_item_selected)

        self.flow_layout.addWidget(image_item)
        self.item_count = self.flow_layout.count()

        return image_item

    def update_current_item(self, ip_adapter_image: IPAdapterImage):
        pixmap = QPixmap(ip_adapter_image.thumb)
        self.current_item.ip_adapter_image = ip_adapter_image
        self.current_item.set_image(pixmap)

    def on_loading_finished(self):
        self.item_count = self.flow_layout.count()
        self.current_item_index = 0
        widget: IPAdapterImageItem = self.flow_layout.itemAt(0).widget()
        self.current_item = widget
        widget.set_selected(True)
        self.ip_adapter_image = widget.ip_adapter_image
        self.finished_loading.emit()

    def clear_selection(self):
        if self.current_item is not None:
            self.current_item.set_selected(False)
            self.current_item_index = None
            self.current_item = None
            self.ip_adapter_image = None

    def on_item_selected(self, item: IPAdapterImageItem):
        for i in range(self.flow_layout.count()):
            widget: IPAdapterImageItem = self.flow_layout.itemAt(i).widget()

            if widget == item:
                self.current_item_index = i
                self.current_item = widget
                self.ip_adapter_image = item.ip_adapter_image
                widget.set_selected(True)
                self.item_selected.emit(item.ip_adapter_image)
            else:
                widget.set_selected(False)

        self.setFocus()

    def set_current_item(self, ip_adapter_image_item: IPAdapterImageItem):
        if self.current_item:
            self.current_item.set_selected(False)

        self.current_item = ip_adapter_image_item
        self.ip_adapter_image = ip_adapter_image_item.ip_adapter_image

        self.scroll_area.ensureWidgetVisible(ip_adapter_image_item)

    def get_first_item(self):
        self.current_item_index = 0
        item: IPAdapterImageItem = self.flow_layout.itemAt(self.current_item_index).widget()

        if item is not None:
            self.set_current_item(item)
            self.item_selected.emit(item.ip_adapter_image)

        return item

    def get_prev_item(self):
        if self.current_item_index > 0:
            self.current_item_index -= 1
            item: IPAdapterImageItem = self.flow_layout.itemAt(self.current_item_index).widget()

            if item is not None:
                self.set_current_item(item)
                self.item_selected.emit(item.ip_adapter_image)

            return item
        return None

    def get_next_item(self):
        if self.current_item_index < self.flow_layout.count() - 1:
            self.current_item_index += 1
            item: IPAdapterImageItem = self.flow_layout.itemAt(self.current_item_index).widget()

            if item is not None:
                self.set_current_item(item)
                self.item_selected.emit(item.ip_adapter_image)
            return item
        return None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Left:
            self.get_prev_item()
        elif event.key() == Qt.Key.Key_Right:
            self.get_next_item()

    def contextMenuEvent(self, event):
        pos = self.flow_widget.mapFrom(self, event.pos())
        item = self.flow_layout.itemAtPosition(pos)

        if item is not None:
            context_menu = QMenu(self)
            delete_action: QAction | None = context_menu.addAction("Delete")
            delete_action.triggered.connect(lambda: self.on_delete_item(item.widget()))
            context_menu.exec(event.globalPos())

    def on_delete_item(self, item: IPAdapterImageItem):
        delete_index = self.flow_layout.index_of(item)

        os.remove(item.ip_adapter_image.thumb)
        os.remove(item.ip_adapter_image.image)

        for image_data in item.ip_adapter_image.images:
            remove_image_data_files(image_data)

        self.flow_layout.remove_item(item)
        self.item_count = self.flow_layout.count()

        clear_view = False

        if self.current_item_index == delete_index:
            if self.current_item_index > self.item_count - 1:
                self.current_item_index = self.current_item_index - 1

            self.clear_selection()
            clear_view = True

        self.item_deleted.emit(item.ip_adapter_image, clear_view)

    def clear(self):
        self.clear_selection()
        self.flow_layout.clear()
