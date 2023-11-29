from importlib.resources import files

from PyQt6.QtWidgets import QGraphicsScene, QGraphicsPixmapItem, QGraphicsView, QGraphicsPathItem

from PyQt6.QtCore import Qt, QRectF, QPoint, QTimer, QSize
from PyQt6.QtGui import QPixmap, QPainter, QPainterPath, QColor, QRadialGradient, QBrush, QPen, QCursor, QGuiApplication
from PyQt6.QtSvg import QSvgRenderer

from iartisanxl.modules.common.drop_lightbox import DropLightBox


class ImageEditor(QGraphicsView):
    BRUSH_BLACK = files("iartisanxl.theme.cursors").joinpath("brush_black.svg")
    BRUSH_WHITE = files("iartisanxl.theme.cursors").joinpath("brush_white.svg")
    CROSSHAIR_BLACK = files("iartisanxl.theme.cursors").joinpath("crosshair_black.svg")
    CROSSHAIR_WHITE = files("iartisanxl.theme.cursors").joinpath("crosshair_white.svg")

    def __init__(self, parent=None):
        super(ImageEditor, self).__init__(parent)

        self.original_width = 300
        self.original_height = 300
        self.aspect_ratio = 1.0

        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self._zoom = 0
        self._empty = True
        self._scene = QGraphicsScene(0, 0, 0, 0)
        self._photo = QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.original_pixmap = None

        self.drawing = False
        self.last_point = QPoint()
        self.brush_color = QColor(0, 0, 0, 255)
        self.brush_size = 32
        self.last_cursor_size = None
        self.hardness = 0
        self.pressure = 0
        self.undo_stack = []
        self.redo_stack = []
        self.current_drawing = []

        self.moving = False
        self.setMouseTracking(True)

        self.drop_lightbox = DropLightBox(self)
        self.drop_lightbox.setText("Drop file here")

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_cursor)

        self.pressure_timer = QTimer()
        self.pressure_timer.timeout.connect(self.draw)

    def set_original_size(self, width, height):
        self.original_width = width
        self.original_height = height
        self.setSceneRect(0, 0, self.original_width, self.original_height)
        self.aspect_ratio = float(width) / float(height)

    def sizeHint(self):
        return QSize(self.original_width, self.original_height)

    def set_pixmap(self, pixmap: QPixmap):
        self.original_pixmap = pixmap
        self._empty = False
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._photo.setPixmap(pixmap)
        self.update_cursor()
        self.fit_in_view()

    def set_white_pixmap(self, width, height):
        # Create a white QPixmap
        white_pixmap = QPixmap(width, height)
        white_pixmap.fill(Qt.GlobalColor.white)

        # Set the QPixmap as the photo
        self.original_pixmap = white_pixmap
        self._empty = False
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._photo.setPixmap(self.original_pixmap)
        self.update_cursor()
        self.fit_in_view()

    def set_image_scale(self, scale_factor):
        if self._photo is not None:
            self._photo.setScale(scale_factor)

    def set_image_x(self, x_position):
        if self._photo is not None:
            self._photo.setX(x_position)

    def set_image_y(self, y_position):
        if self._photo is not None:
            self._photo.setY(y_position)

    def has_photo(self):
        return not self._empty

    def fit_in_view(self):
        rect = QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.has_photo():
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(
                    viewrect.width() / scenerect.width(),
                    viewrect.height() / scenerect.height(),
                )
                self.scale(factor, factor)
            self._zoom = 0

    def enterEvent(self, event):
        self.setFocus()
        self.drawing = False
        self.timer.start(100)
        super().enterEvent(event)

    def wheelEvent(self, event):
        if self.has_photo():
            if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                if event.angleDelta().y() > 0:
                    factor = 1.25
                    self._zoom += 1
                else:
                    factor = 0.8
                    self._zoom -= 1
                if self._zoom > 0:
                    self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
                    self.scale(factor, factor)
                elif self._zoom == 0:
                    self.fit_in_view()
                else:
                    self._zoom = 0
            else:
                super().wheelEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.moving:
                self.timer.stop()
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                super().mousePressEvent(event)
            else:
                self.timer.stop()
                self.drawing = True
                self.pressure = 0
                self.last_point = self.mapToScene(event.pos())
                self.pressure_timer.start(100)

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self.drawing:
            current_point = self.mapToScene(event.pos())

            hardness = self.hardness * self.pressure

            gradient = QRadialGradient(self.last_point, self.brush_size / 2)
            gradient.setColorAt(0, QColor(0, 0, 0, 255))  # Inner color (black)
            gradient.setColorAt(hardness, self.brush_color)  # Outer color (black)
            gradient.setColorAt(1, QColor(0, 0, 0, 0))  # Beyond hardness (transparent)
            brush = QBrush(gradient)

            path_item = QGraphicsPathItem()
            path_item.setBrush(brush)
            path_item.setPen(QPen(Qt.GlobalColor.transparent))
            path = QPainterPath()
            path.addEllipse(self.last_point, self.brush_size / 2, self.brush_size / 2)
            path_item.setPath(path)
            self._scene.addItem(path_item)
            self.current_drawing.append(path_item)

            self.last_point = current_point

        super().mouseMoveEvent(event)

    def draw(self):
        if self.drawing:
            # Increase pressure over time
            self.pressure = min(self.pressure + 0.01, 1)

            # Use pressure to adjust hardness
            hardness = self.hardness * self.pressure

            # Create a radial gradient as the brush
            gradient = QRadialGradient(self.last_point, self.brush_size / 2)
            gradient.setColorAt(0, QColor(0, 0, 0, 255))  # Inner color (black)
            gradient.setColorAt(hardness, self.brush_color)  # Outer color (black)
            gradient.setColorAt(1, QColor(0, 0, 0, 0))  # Beyond hardness (transparent)
            brush = QBrush(gradient)

            # Create a path item with the brush
            path_item = QGraphicsPathItem()
            path_item.setBrush(brush)
            path_item.setPen(QPen(Qt.GlobalColor.transparent))
            path = QPainterPath()
            path.addEllipse(self.last_point, self.brush_size / 2, self.brush_size / 2)
            path_item.setPath(path)
            self._scene.addItem(path_item)
            self.current_drawing.append(path_item)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.drawing = False
            self.update_cursor()
            self.undo_stack.append(self.current_drawing)
            self.current_drawing = []
            self.pressure_timer.stop()
            self.timer.start(100)
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
                self._scene.removeItem(path_item)
                redo_drawing.append(path_item)

            self.redo_stack.append(redo_drawing)

    def redo(self):
        if self.redo_stack:
            drawing = self.redo_stack.pop()
            undo_drawing = []
            for path_item in drawing:
                self._scene.addItem(path_item)
                undo_drawing.append(path_item)
            self.undo_stack.append(undo_drawing)

    def clear_and_restore(self):
        self._scene.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        original_image_item = QGraphicsPixmapItem(self.original_pixmap)
        self._photo = original_image_item
        self._scene.addItem(original_image_item)

    def clear(self):
        self._scene.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        original_image_item = QGraphicsPixmapItem()
        self._photo = original_image_item
        self._scene.addItem(original_image_item)

        self.original_pixmap = None
        self._empty = True

    def get_painted_image(self):
        pixmap = QPixmap(self.original_width, self.original_height)
        painter = QPainter(pixmap)
        self.render(painter)
        painter.end()

        return pixmap.toImage()

    def create_cursor(self, svg_path, use_crosshair):
        # Check if we should use the crosshair cursor
        if use_crosshair:
            # If it is, use the last pixmap size
            pixmap_size = self.last_cursor_size
        else:
            zoom_factor = self.transform().m11()
            # If it's not, calculate the pixmap size
            pixmap_size = int(self.brush_size * 0.8 * zoom_factor)
            # Store the calculated pixmap size
            self.last_cursor_size = pixmap_size

        pixmap = QPixmap(pixmap_size, pixmap_size)
        pixmap.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pixmap)
        renderer = QSvgRenderer(svg_path)
        renderer.render(painter)
        painter.end()
        return QCursor(pixmap)

    def update_cursor(self):
        # Check if the zoom factor is less than 7 and the brush size is smaller than 15
        use_crosshair = self._zoom < 7 and self.brush_size < 15

        # Determine the color of the cursor
        bg_color = self.get_color_under_cursor()
        brightness = (bg_color.red() * 299 + bg_color.green() * 587 + bg_color.blue() * 114) / 1000

        # Determine the cursor type based on the brightness and whether we should use the crosshair
        if brightness < 128:
            cursor_type = self.CROSSHAIR_WHITE if use_crosshair else self.BRUSH_WHITE
        else:
            cursor_type = self.CROSSHAIR_BLACK if use_crosshair else self.BRUSH_BLACK

        self.setCursor(self.create_cursor(str(cursor_type), use_crosshair))

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
