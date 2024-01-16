from importlib.resources import files

from PyQt6.QtWidgets import QGraphicsScene, QGraphicsPixmapItem, QGraphicsView, QGraphicsPathItem, QApplication, QMenu
from PyQt6.QtCore import Qt, QPoint, QTimer, QSize, pyqtSignal, QLineF
from PyQt6.QtGui import QPixmap, QPainter, QPainterPath, QColor, QRadialGradient, QBrush, QPen, QCursor, QGuiApplication, QAction
from PyQt6.QtSvg import QSvgRenderer

from iartisanxl.modules.common.drop_lightbox import DropLightBox


class ImageEditor(QGraphicsView):
    BRUSH_BLACK = files("iartisanxl.theme.cursors").joinpath("brush_black.svg")
    BRUSH_WHITE = files("iartisanxl.theme.cursors").joinpath("brush_white.svg")
    CROSSHAIR_BLACK = files("iartisanxl.theme.cursors").joinpath("crosshair_black.svg")
    CROSSHAIR_WHITE = files("iartisanxl.theme.cursors").joinpath("crosshair_white.svg")

    image_changed = pyqtSignal()
    image_moved = pyqtSignal(float, float)
    image_scaled = pyqtSignal(float)

    def __init__(self, target_width: int, target_height: int, aspect_ratio: float, save_directory=None):
        super(ImageEditor, self).__init__()

        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.save_directory = save_directory
        self.original_scale = 1.000
        self.original_x = 0
        self.original_y = 0
        self.original_rotation = 0

        self.original_pixmap = None
        self.pixmap_item = None
        self.target_width = target_width
        self.target_height = target_height
        self.aspect_ratio = aspect_ratio

        self.setSceneRect(0, 0, self.target_width, self.target_height)

        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.scene = QGraphicsScene(0, 0, 0, 0)
        self.setScene(self.scene)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        self.drawing = False
        self.last_point = QPoint()
        self.brush_color = QColor(0, 0, 0, 255)
        self.brush_size = 32
        self.last_cursor_size = None
        self.hardness = 0
        self.undo_stack = []
        self.redo_stack = []
        self.current_drawing = []

        self.moving = False
        self.setMouseTracking(True)

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
        self.timer.start(100)
        super().enterEvent(event)

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

            self.image_scaled.emit(scale_factor)
            self.image_moved.emit(-new_x, -new_y)
            self.image_changed.emit()

    def set_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.set_pixmap(pixmap)

    def set_pixmap(self, pixmap: QPixmap):
        if self.pixmap_item is not None:
            self.scene.removeItem(self.pixmap_item)
            self.pixmap_item = None

        self.scene.clear()

        self.original_pixmap = pixmap

        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.pixmap_item.setZValue(0)
        self.scene.addItem(self.pixmap_item)
        self.update_cursor()
        self.image_changed.emit()

    def set_color_pixmap(self, width, height):
        color_pixmap = QPixmap(width, height)
        color_pixmap.fill(self.brush_color)
        self.set_pixmap(color_pixmap)

    def set_image_scale(self, scale_factor):
        if self.pixmap_item is not None:
            self.pixmap_item.setTransformOriginPoint(self.pixmap_item.boundingRect().center())
            self.pixmap_item.setScale(scale_factor)
            self.image_changed.emit()

    def set_image_x(self, x_position):
        if self.pixmap_item is not None:
            self.pixmap_item.setX(x_position)
            self.image_changed.emit()

    def set_image_y(self, y_position):
        if self.pixmap_item is not None:
            self.pixmap_item.setY(y_position)
            self.image_changed.emit()

    def rotate_image(self, angle):
        if self.pixmap_item is not None:
            self.pixmap_item.setTransformOriginPoint(self.pixmap_item.boundingRect().center())
            self.pixmap_item.setRotation(angle)
            self.image_changed.emit()

    def draw(self, point):
        gradient = QRadialGradient(point, self.brush_size / 2)
        gradient.setColorAt(0, QColor(0, 0, 0, 255))  # Inner color (black)
        gradient.setColorAt(self.hardness, self.brush_color)  # Outer color (black)
        gradient.setColorAt(1, QColor(0, 0, 0, 0))  # Beyond hardness (transparent)
        brush = QBrush(gradient)

        path_item = QGraphicsPathItem(self.pixmap_item)
        path_item.setZValue(1)
        path_item.setBrush(brush)
        path_item.setPen(QPen(Qt.GlobalColor.transparent))
        path = QPainterPath()
        path.addEllipse(point, self.brush_size / 2, self.brush_size / 2)
        path_item.setPath(path)
        self.current_drawing.append(path_item)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.pixmap_item:
            self.timer.stop()

            if self.moving:
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            else:
                self.drawing = True
                self.last_point = self.pixmap_item.mapFromScene(self.mapToScene(event.pos()))
                self.draw(self.last_point)

            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self.drawing and self.pixmap_item:
            current_point = self.pixmap_item.mapFromScene(self.mapToScene(event.pos()))

            if QLineF(current_point, self.last_point).length() > self.brush_size / 4:
                self.draw(current_point)
                self.last_point = current_point
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.drawing = False
            self.update_cursor()
            self.undo_stack.append(self.current_drawing)
            self.current_drawing = []
            self.timer.start(100)
            self.image_changed.emit()
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.timer.stop()
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.moving = True

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()
        if key == Qt.Key.Key_Z and modifiers == Qt.KeyboardModifier.ControlModifier:
            self.undo()
        elif key == Qt.Key.Key_Z and modifiers == (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
            self.redo()
        elif key == Qt.Key.Key_Space:
            self.moving = False
            self.timer.start(100)
        else:
            super().keyReleaseEvent(event)

    def undo(self):
        if self.undo_stack:
            drawing = self.undo_stack.pop()
            redo_drawing = []
            for path_item in drawing:
                self.scene.removeItem(path_item)
                redo_drawing.append(path_item)

            self.redo_stack.append(redo_drawing)
            self.image_changed.emit()

    def redo(self):
        if self.redo_stack:
            drawing = self.redo_stack.pop()
            undo_drawing = []
            for path_item in drawing:
                self.scene.addItem(path_item)
                undo_drawing.append(path_item)
            self.undo_stack.append(undo_drawing)
            self.image_changed.emit()

    def clear_and_restore(self):
        self.scene.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        original_image_item = QGraphicsPixmapItem(self.original_pixmap)
        self.pixmap_item = original_image_item
        self.scene.addItem(original_image_item)

    def clear(self):
        self.scene.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.pixmap_item = None
        self.original_pixmap = None

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
        scale_factor = 1 if self.pixmap_item is None else self.pixmap_item.scale()
        pixmap_size = int(self.brush_size * 0.8 * zoom_factor * scale_factor)

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

    def set_brush_color(self, color: tuple):
        self.brush_color = QColor(int(color[0]), int(color[1]), int(color[2]), 255)

    def set_brush_size(self, value):
        self.brush_size = value

    def set_brush_hardness(self, value):
        self.hardness = value

    def dragMoveEvent(self, event):
        pass

    def dragEnterEvent(self, event):
        pass

    def dropEvent(self, event):
        pass

    def contextMenuEvent(self, event):
        context_menu = QMenu(self)

        copy_action = QAction("Copy Image", self)
        copy_action.triggered.connect(self.copy_image)
        context_menu.addAction(copy_action)

        paste_action = QAction("Paste Image", self)
        paste_action.triggered.connect(self.paste_image)
        context_menu.addAction(paste_action)

        save_action = QAction("Save Image", self)
        save_action.triggered.connect(self.save_image)
        context_menu.addAction(save_action)

        context_menu.exec(event.globalPos())

    def paste_image(self):
        clipboard = QApplication.clipboard()
        pixmap = clipboard.pixmap()
        if not pixmap.isNull():
            self.set_pixmap(pixmap)
            self.image_changed.emit()

    def copy_image(self):
        pass

    def save_image(self):
        pass

    def get_layer(self, layer_z):
        layer_items = [item for item in self.scene.items() if item.zValue() == layer_z]

        layer_scene = QGraphicsScene()
        layer_scene.setSceneRect(0, 0, self.target_width, self.target_height)
        for item in layer_items:
            layer_scene.addItem(item)

        layer_pixmap = QPixmap(self.target_width, self.target_height)
        layer_pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(layer_pixmap)
        layer_scene.render(painter)
        painter.end()

        for item in layer_items:
            self.scene.addItem(item)
            item.setParentItem(self.pixmap_item)

        return layer_pixmap
