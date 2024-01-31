import os
from datetime import datetime

from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFileDialog, QMenu, QApplication
from PyQt6.QtCore import Qt, QSize, QPointF, pyqtSignal
from PyQt6.QtGui import QPixmap, QAction, QPainter, QGuiApplication

from iartisanxl.modules.common.drop_lightbox import DropLightBox


class ImageAdderPreview(QGraphicsView):
    image_moved = pyqtSignal(float, float)
    image_scaled = pyqtSignal(float)
    image_updated = pyqtSignal()
    image_pasted = pyqtSignal(str)
    image_copy = pyqtSignal()
    image_save = pyqtSignal(str)

    def __init__(self, target_width: int, target_height: int, aspect_ratio: float, save_directory=None):
        super(ImageAdderPreview, self).__init__()

        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.save_directory = save_directory
        self.original_scale = 1.000
        self.original_x = 0
        self.original_y = 0
        self.original_rotation = 0

        self._mouse_press_pos = None

        self.original_pixmap = None
        self.pixmap_item = None
        self.target_width = target_width
        self.target_height = target_height
        self.aspect_ratio = aspect_ratio

        self.cropped_image = None
        self.image_was_translated = False
        self.image_was_scaled = False
        self.image_was_rotated = False
        self.image_convert_thread = None

        self.setSceneRect(0, 0, self.target_width, self.target_height)

        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.scene = QGraphicsScene(0, 0, 0, 0)
        self.setScene(self.scene)

        self.drop_lightbox = DropLightBox(self)
        self.drop_lightbox.setText("Drop file here")

    def set_aspect(self, width: int, height: int, ratio: float):
        self.target_width = width
        self.target_height = height
        self.aspect_ratio = ratio

        self.setSceneRect(0, 0, self.target_width, self.target_height)

    def sizeHint(self):
        return QSize(self.target_width, self.target_height)

    def resizeEvent(self, event):
        rect = self.sceneRect()
        self.resetTransform()
        self.scale(self.viewport().width() / rect.width(), self.viewport().height() / rect.height())
        super().resizeEvent(event)

    def fit_image(self):
        if self.pixmap_item is not None:
            pixmap_size = self.pixmap_item.pixmap().size()

            # Calculate scale factors for both dimensions, prioritizing height to fill the view
            width_scale = (self.mapToScene(self.viewport().rect()).boundingRect().width()) / pixmap_size.width()
            height_scale = (self.mapToScene(self.viewport().rect()).boundingRect().height()) / pixmap_size.height()
            scale_factor = min(width_scale, height_scale)

            # Scale the image
            self.set_image_scale(scale_factor)

            # Calculate the new top-left position after scaling
            scaled_width = pixmap_size.width() * scale_factor
            scaled_height = pixmap_size.height() * scale_factor
            width_diff = pixmap_size.width() - scaled_width
            height_diff = pixmap_size.height() - scaled_height

            new_x = width_diff / 2
            new_y = height_diff / 2

            # Move the image to the new top-left corner
            self.pixmap_item.setPos(-new_x, -new_y)

            # Emit signals for potential updates
            self.image_scaled.emit(scale_factor)
            self.image_moved.emit(-new_x, -new_y)
            self.image_updated.emit()

    def set_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.set_pixmap(pixmap)

    def set_pixmap(self, pixmap):
        if self.pixmap_item is not None:
            self.scene.removeItem(self.pixmap_item)
            self.pixmap_item = None

        self.scene.clear()

        self.original_pixmap = pixmap
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene.addItem(self.pixmap_item)

    def set_image_scale(self, scale_factor):
        if self.pixmap_item is not None:
            self.pixmap_item.setTransformOriginPoint(self.pixmap_item.boundingRect().center())
            self.pixmap_item.setScale(scale_factor)
            self.image_was_scaled = True
            self.image_was_translated = True
            self.image_updated.emit()

    def set_image_x(self, x_position):
        if self.pixmap_item is not None:
            self.pixmap_item.setX(x_position)
            self.image_was_translated = True
            self.image_updated.emit()

    def set_image_y(self, y_position):
        if self.pixmap_item is not None:
            self.pixmap_item.setY(y_position)
            self.image_was_translated = True
            self.image_updated.emit()

    def rotate_image(self, angle):
        if self.pixmap_item is not None:
            self.pixmap_item.setTransformOriginPoint(self.pixmap_item.boundingRect().center())
            self.pixmap_item.setRotation(angle)
            self.image_was_rotated = True
            self.image_was_translated = True
            self.image_updated.emit()

    @property
    def image_modified(self):
        return (
            self.pixmap_item.scale() != self.original_scale
            or self.pixmap_item.x() != self.original_x
            or self.pixmap_item.y() != self.original_y
            or self.pixmap_item.rotation() != self.original_rotation
        )

    def mousePressEvent(self, event):
        self._mouse_press_pos = event.pos()
        super(ImageAdderPreview, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            # Calculate the difference between the current mouse position and the last mouse press position
            diff = event.pos() - self._mouse_press_pos
            self._mouse_press_pos = event.pos()

            # Update the position of the pixmap_item
            if self.pixmap_item is not None:
                new_pos = self.pixmap_item.pos() + QPointF(diff)
                self.pixmap_item.setPos(new_pos)
                self.image_moved.emit(new_pos.x(), new_pos.y())

        super(ImageAdderPreview, self).mouseMoveEvent(event)

    def clear(self):
        if self.pixmap_item is not None:
            self.scene.removeItem(self.pixmap_item)
            self.pixmap_item = None

        self.scene.clear()

    def contextMenuEvent(self, event):
        context_menu = QMenu(self)

        copy_action = QAction("Copy Image", self)
        copy_action.triggered.connect(self.copy_image)
        if self.pixmap_item is None:
            copy_action.setDisabled(True)
        context_menu.addAction(copy_action)

        paste_action = QAction("Paste Image", self)
        paste_action.triggered.connect(self.paste_image)

        clipboard = QGuiApplication.clipboard()
        mime_data = clipboard.mimeData()
        paste_action.setDisabled(True)
        if mime_data.hasUrls():
            url = mime_data.urls()[0]
            if url.isLocalFile():
                file_path = url.toLocalFile()
                if file_path.endswith(".png") and os.path.isfile(file_path):
                    paste_action.setEnabled(True)

        context_menu.addAction(paste_action)

        save_action = QAction("Save Image", self)
        save_action.triggered.connect(self.on_save_image)
        if self.pixmap_item is None:
            save_action.setDisabled(True)
        context_menu.addAction(save_action)

        context_menu.exec(event.globalPos())

    def paste_image(self):
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()

        mime_data = clipboard.mimeData()
        url = mime_data.urls()[0]
        file_path = url.toLocalFile()

        self.image_pasted.emit(file_path)

    def copy_image(self):
        self.image_copy.emit()

    def on_save_image(self):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        home_dir = os.path.expanduser("~")
        pictures_dir = os.path.join(home_dir, "Pictures")

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save image",
            os.path.join(pictures_dir, f"{timestamp}.png"),
            "Images (*.png *.jpg)",
        )

        self.image_save.emit(save_path)

    def get_current_view_as_pixmap(self):
        size = self.size()
        pixmap = QPixmap(size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        self.render(painter)
        painter.end()
        return pixmap

    def dragMoveEvent(self, event):
        pass

    def dragEnterEvent(self, event):
        pass

    def dropEvent(self, event):
        pass
