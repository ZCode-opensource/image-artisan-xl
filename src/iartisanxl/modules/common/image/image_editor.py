import os
from datetime import datetime
from importlib.resources import files

from PyQt6.QtCore import QPoint, QPointF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QBrush, QColor, QCursor, QGuiApplication, QPainter, QPen, QPixmap, QRadialGradient
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QApplication, QFileDialog, QGraphicsEllipseItem, QGraphicsScene, QGraphicsView, QMenu

from iartisanxl.modules.common.drop_lightbox import DropLightBox
from iartisanxl.modules.common.image.image_editor_layer import ImageEditorLayer
from iartisanxl.modules.common.image.layer_manager import LayerManager


class ImageEditor(QGraphicsView):
    BRUSH_BLACK = files("iartisanxl.theme.cursors").joinpath("brush_black.svg")
    BRUSH_WHITE = files("iartisanxl.theme.cursors").joinpath("brush_white.svg")
    CROSSHAIR_BLACK = files("iartisanxl.theme.cursors").joinpath("crosshair_black.svg")
    CROSSHAIR_WHITE = files("iartisanxl.theme.cursors").joinpath("crosshair_white.svg")

    image_changed = pyqtSignal()
    image_moved = pyqtSignal(float, float)
    image_scaled = pyqtSignal(float)
    image_rotated = pyqtSignal(float)
    image_pasted = pyqtSignal(str)
    image_copy = pyqtSignal()
    image_save = pyqtSignal(str)

    def __init__(self, target_width: int, target_height: int, aspect_ratio: float, save_directory=None):
        super(ImageEditor, self).__init__()

        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.save_directory = save_directory
        self.target_width = target_width
        self.target_height = target_height
        self.aspect_ratio = aspect_ratio

        self.setSceneRect(0, 0, self.target_width, self.target_height)

        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.scene = QGraphicsScene(0, 0, 0, 0)
        self.setScene(self.scene)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        self.layer_manager = LayerManager()
        self.selected_layer_id = 0

        self.drawing = False
        self.last_point = QPoint()
        self.brush_color = QColor(0, 0, 0, 255)
        self.brush_size = 32
        self.hardness = 0
        self.brush_preview = None
        self.last_cursor_size = None

        self.erasing = False

        self.moving = False
        self.last_mouse_position = None
        self.setMouseTracking(True)
        self.current_scale_factor = 1.0
        self.total_translation = QPointF(0, 0)

        self.drop_lightbox = DropLightBox(self)
        self.drop_lightbox.setText("Drop file here")

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_cursor)

    def sizeHint(self):
        return QSize(self.target_width, self.target_height)

    def resizeEvent(self, event):
        rect = self.sceneRect()
        self.resetTransform()
        self.scale(self.viewport().width() / rect.width(), self.viewport().height() / rect.height())
        super().resizeEvent(event)

    def enterEvent(self, event):
        self.setFocus()
        self.drawing = False
        self.moving = False
        self.timer.start(100)
        super().enterEvent(event)

    def add_empty_layer(self):
        pixmap = QPixmap(self.target_width, self.target_height)
        pixmap.fill(Qt.GlobalColor.transparent)

        layer_id = self.set_pixmap(pixmap)

        return layer_id

    def set_image(self, image_path: str, delete_prev_image: bool = True):
        pixmap = QPixmap(image_path)

        layer_id = self.set_pixmap(pixmap, self.selected_layer_id, image_path, delete_prev_image=delete_prev_image)

        return layer_id

    def set_pixmap(
        self,
        pixmap: QPixmap,
        layer_id: int = None,
        image_path: str = None,
        order: int = None,
        delete_prev_image: bool = True,
    ):
        layer = self.layer_manager.get_layer_by_id(layer_id)

        alpha_pixmap = QPixmap(pixmap.size())
        alpha_pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(alpha_pixmap)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

        if layer is not None:
            layer.pixmap_item.setPixmap(alpha_pixmap)

            if delete_prev_image and layer.image_path is not None and os.path.isfile(layer.image_path):
                os.remove(layer.image_path)

            layer.original_path = image_path
        else:
            layer = self.layer_manager.add_new_layer(alpha_pixmap, image_path=image_path, order=order)
            self.scene.addItem(layer.pixmap_item)

        layer.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        layer.pixmap_item.setZValue(layer.order)
        self.image_changed.emit()

        return layer.layer_id

    def reload_image_layer(self, image_path: str, original_path: str, order: int):
        pixmap = QPixmap(image_path)
        layer = self.layer_manager.reload_layer(pixmap, image_path, original_path, order)
        self.scene.addItem(layer.pixmap_item)

        layer.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        layer.pixmap_item.setZValue(layer.order)

        return layer.layer_id

    def get_selected_layer(self):
        layer = self.layer_manager.get_layer_by_id(self.selected_layer_id)
        return layer

    def get_all_layers(self) -> list[ImageEditorLayer]:
        return self.layer_manager.layers

    def set_layer_locked(self, layer_id: int, locked: bool):
        layer = self.layer_manager.get_layer_by_id(layer_id)

        if layer is not None:
            layer.locked = locked

    def set_layer_order(self, layer_id: int, order: int):
        layer = self.layer_manager.get_layer_by_id(layer_id)

        if layer is not None and layer.order != order:
            self.layer_manager.move_layer(layer_id, order)

            for layer in self.layer_manager.get_layers():
                layer.pixmap_item.setZValue(layer.order)

    def edit_all_layers_order(self, layers: list):
        for layer_id, order in layers:
            for layer in self.layer_manager.layers:
                if layer.layer_id == layer_id:
                    layer.order = order
                    layer.pixmap_item.setZValue(layer.order)

    def fit_image(self):
        layer = self.layer_manager.get_layer_by_id(self.selected_layer_id)

        if layer is not None:
            pixmap_size = layer.pixmap_item.pixmap().size()

            layer.pixmap_item.setRotation(0)

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
            layer.pixmap_item.setPos(-new_x, -new_y)

            self.image_scaled.emit(scale_factor)
            self.image_moved.emit(-new_x, -new_y)
            self.image_rotated.emit(0)

    def set_image_scale(self, scale_factor):
        layer = self.layer_manager.get_layer_by_id(self.selected_layer_id)

        if layer is not None:
            if layer.pixmap_item is not None:
                scale_ratio = scale_factor / layer.pixmap_item.scale()

                layer.pixmap_item.setTransformOriginPoint(layer.pixmap_item.boundingRect().center())
                layer.pixmap_item.setScale(scale_factor)

                if layer.locked:
                    for layer in self.layer_manager.get_layers():
                        if layer.layer_id != self.selected_layer_id and layer.locked:
                            layer.pixmap_item.setTransformOriginPoint(layer.pixmap_item.boundingRect().center())
                            layer.pixmap_item.setScale(layer.pixmap_item.scale() * scale_ratio)

    def set_image_x(self, x_position):
        layer = self.layer_manager.get_layer_by_id(self.selected_layer_id)

        if layer is not None:
            x_delta = x_position - layer.pixmap_item.x()

            if layer.pixmap_item is not None:
                layer.pixmap_item.setX(x_position)

                if layer.locked:
                    for layer in self.layer_manager.get_layers():
                        if layer.layer_id != self.selected_layer_id and layer.locked:
                            layer.pixmap_item.setX(layer.pixmap_item.x() + x_delta)

    def set_image_y(self, y_position):
        layer = self.layer_manager.get_layer_by_id(self.selected_layer_id)

        if layer is not None and layer.pixmap_item is not None:
            y_delta = y_position - layer.pixmap_item.y()

            layer.pixmap_item.setY(y_position)

            if layer.locked:
                for layer in self.layer_manager.get_layers():
                    if layer.layer_id != self.selected_layer_id and layer.locked:
                        layer.pixmap_item.setY(layer.pixmap_item.y() + y_delta)

    def rotate_image(self, angle):
        layer = self.layer_manager.get_layer_by_id(self.selected_layer_id)

        if layer is not None:
            if layer.pixmap_item is not None:
                rotation_delta = angle - layer.pixmap_item.rotation()

                layer.pixmap_item.setTransformOriginPoint(layer.pixmap_item.boundingRect().center())
                layer.pixmap_item.setRotation(angle)

                if layer.locked:
                    for layer in self.layer_manager.get_layers():
                        if layer.layer_id != self.selected_layer_id and layer.locked:
                            layer.pixmap_item.setTransformOriginPoint(layer.pixmap_item.boundingRect().center())
                            layer.pixmap_item.setRotation(layer.pixmap_item.rotation() + rotation_delta)

    def clear_and_restore(self):
        layer = self.layer_manager.get_layer_by_id(self.selected_layer_id)

        if layer is not None and layer.pixmap_item is not None:
            pixmap = QPixmap(layer.image_path)
            layer.pixmap_item.setPixmap(pixmap)
            layer.pixmap_item.setScale(1)
            layer.pixmap_item.setX(0)
            layer.pixmap_item.setY(0)
            layer.pixmap_item.setRotation(0)
            self.image_scaled.emit(1)
            self.image_moved.emit(0, 0)
            self.image_rotated.emit(0)

            self.image_changed.emit()

    def delete_layer(self):
        layer = self.layer_manager.get_layer_by_id(self.selected_layer_id)

        if layer is not None and layer.pixmap_item is not None:
            if layer.image_path is not None and os.path.isfile(layer.image_path):
                os.remove(layer.image_path)
            if layer.original_path is not None and os.path.isfile(layer.original_path):
                os.remove(layer.original_path)

            self.scene.removeItem(layer.pixmap_item)
            self.layer_manager.delete_layer(layer.layer_id)
            del layer

    def clear_all(self):
        self.scene.clear()
        self.layer_manager.delete_all()
        self.selected_layer_id = None

    def draw(self, point):
        layer = self.layer_manager.get_layer_by_id(self.selected_layer_id)

        if layer is not None:
            pixmap = layer.pixmap_item.pixmap()
            painter = QPainter(pixmap)

            if self.erasing:
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            else:
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

            gradient = QRadialGradient(point, self.brush_size / 2)
            gradient.setColorAt(
                0, QColor(self.brush_color.red(), self.brush_color.green(), self.brush_color.blue(), 255)
            )
            gradient.setColorAt(self.hardness, self.brush_color)
            gradient.setColorAt(
                1, QColor(self.brush_color.red(), self.brush_color.green(), self.brush_color.blue(), 0)
            )

            brush = QBrush(gradient)
            painter.setBrush(brush)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(point, self.brush_size / 2, self.brush_size / 2)
            painter.end()

            layer.pixmap_item.setPixmap(pixmap)
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.moving:
                self.timer.stop()
                self.last_mouse_position = event.pos()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            else:
                layer = self.layer_manager.get_layer_by_id(self.selected_layer_id)
                if layer.pixmap_item is not None:
                    self.drawing = True
                    self.last_point = layer.pixmap_item.mapFromScene(self.mapToScene(event.pos()))
                    self.draw(self.last_point)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            if self.moving:
                self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
                delta = self.mapToScene(self.last_mouse_position) - self.mapToScene(event.pos())
                self.setSceneRect(self.sceneRect().translated(delta.x(), delta.y()))
                self.translate(delta.x() / self.current_scale_factor, delta.y() / self.current_scale_factor)
                self.total_translation += delta
                self.last_mouse_position = event.pos()
            elif self.drawing:
                layer = self.layer_manager.get_layer_by_id(self.selected_layer_id)
                current_point = layer.pixmap_item.mapFromScene(self.mapToScene(event.pos()))
                self.draw(current_point)
                self.last_point = current_point
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.moving:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            elif self.drawing:
                self.image_changed.emit()
                self.drawing = False
                self.timer.start(100)
            else:
                self.timer.start(100)

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor

        # Set the anchor point to the mouse position
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # Zoom
        if event.angleDelta().y() > 0:
            zoomFactor = zoomInFactor
        else:
            zoomFactor = zoomOutFactor
        self.scale(zoomFactor, zoomFactor)
        self.current_scale_factor *= zoomFactor

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            if not self.moving:
                self.timer.stop()
                self.setCursor(Qt.CursorShape.OpenHandCursor)
                self.moving = True

            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        key = event.key()

        if key == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.moving = False
            self.timer.start(100)
        else:
            super().keyReleaseEvent(event)

    def contextMenuEvent(self, event):
        context_menu = QMenu(self)

        copy_action = QAction("Copy Image", self)
        copy_action.triggered.connect(self.copy_image)
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
        save_action.triggered.connect(self.save_image)
        context_menu.addAction(save_action)

        context_menu.exec(event.globalPos())

    def save_image(self):
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

    def paste_image(self):
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()

        mime_data = clipboard.mimeData()
        url = mime_data.urls()[0]
        file_path = url.toLocalFile()

        self.image_pasted.emit(file_path)

    def copy_image(self):
        self.image_copy.emit()

    def create_cursor(self, svg_path, use_crosshair, pixmap_size):
        if use_crosshair:
            pixmap_size *= 10

        pixmap = QPixmap(pixmap_size, pixmap_size)
        pixmap.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pixmap)
        renderer = QSvgRenderer(svg_path)
        renderer.render(painter)
        painter.end()
        return QCursor(pixmap)

    def update_cursor(self):
        zoom_factor = self.transform().m11()
        layer = self.layer_manager.get_layer_by_id(self.selected_layer_id)
        scale_factor = 1 if layer is None else layer.pixmap_item.scale()
        pixmap_size = int(self.brush_size * 1 * zoom_factor * scale_factor)

        use_crosshair = pixmap_size < 6

        # Determine the color of the cursor
        bg_color = self.get_color_under_cursor()
        brightness = (bg_color.red() * 299 + bg_color.green() * 587 + bg_color.blue() * 114) / 1000

        # Determine the cursor type based on the brightness and whether we should use the crosshair
        if brightness < 128:
            cursor_type = self.CROSSHAIR_WHITE if use_crosshair else self.BRUSH_WHITE
        else:
            cursor_type = self.CROSSHAIR_BLACK if use_crosshair else self.BRUSH_BLACK

        self.setCursor(self.create_cursor(str(cursor_type), use_crosshair, pixmap_size))

    def get_color_under_cursor(self):
        screen = QGuiApplication.primaryScreen()
        cursor_pos = QCursor.pos()
        pixmap = screen.grabWindow(0, cursor_pos.x(), cursor_pos.y(), 1, 1)
        image = pixmap.toImage()
        color = QColor(image.pixelColor(0, 0))

        return color

    def get_color_under_point(self, point):
        # Convert the point from scene coordinates to global coordinates
        global_point = self.mapToGlobal(point)

        screen = QGuiApplication.primaryScreen()
        pixmap = screen.grabWindow(0, global_point.x(), global_point.y(), 1, 1)
        image = pixmap.toImage()
        color = QColor(image.pixelColor(0, 0))

        return color

    def set_brush_color(self, color: tuple):
        self.brush_color = QColor(int(color[0]), int(color[1]), int(color[2]), 255)

    def set_brush_size(self, value):
        self.brush_size = value
        self.show_brush_preview()

    def set_brush_hardness(self, value):
        self.hardness = value
        self.show_brush_preview()

    def show_brush_preview(self):
        if self.brush_preview is not None:
            self.scene.removeItem(self.brush_preview)
            self.brush_preview = None

        center = self.sceneRect().center()

        brightness = (
            self.brush_color.red() * 299 + self.brush_color.green() * 587 + self.brush_color.blue() * 114
        ) / 1000

        if brightness < 128:
            brush_preview_color = Qt.GlobalColor.white
        else:
            brush_preview_color = Qt.GlobalColor.black

        self.brush_preview = QGraphicsEllipseItem(
            center.x() - self.brush_size / 2, center.y() - self.brush_size / 2, self.brush_size, self.brush_size
        )
        self.brush_preview.setPen(QPen(brush_preview_color))

        # Create a gradient for the brush preview
        gradient = QRadialGradient(center, self.brush_size / 2)
        gradient.setColorAt(0, QColor(self.brush_color.red(), self.brush_color.green(), self.brush_color.blue(), 255))
        gradient.setColorAt(self.hardness, self.brush_color)
        gradient.setColorAt(1, QColor(self.brush_color.red(), self.brush_color.green(), self.brush_color.blue(), 0))

        # Set the brush preview's brush to the gradient
        self.brush_preview.setBrush(QBrush(gradient))

        z_value = len(self.layer_manager.layers) + 1
        self.brush_preview.setZValue(z_value)  # Make sure the preview is drawn on top
        self.scene.addItem(self.brush_preview)

    def hide_brush_preview(self):
        # Remove the brush preview
        if self.brush_preview is not None:
            self.scene.removeItem(self.brush_preview)
            self.brush_preview = None

    def reset_view(self):
        self.scale(1 / self.current_scale_factor, 1 / self.current_scale_factor)
        self.setSceneRect(self.sceneRect().translated(-self.total_translation.x(), -self.total_translation.y()))
        self.current_scale_factor = 1.0
        self.total_translation = QPointF(0, 0)

    def get_scene_as_pixmap(self):
        pixmap = QPixmap(self.sceneRect().size().toSize())
        pixmap.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.render(painter)
        painter.end()

        return pixmap

    def dragMoveEvent(self, event):
        pass

    def dragEnterEvent(self, event):
        pass

    def dropEvent(self, event):
        pass
