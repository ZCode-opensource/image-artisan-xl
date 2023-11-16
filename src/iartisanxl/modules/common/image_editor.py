from PyQt6.QtWidgets import (
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsView,
    QSizePolicy,
    QApplication,
    QGraphicsPathItem,
)
from PyQt6.QtCore import Qt, QRectF, QPoint
from PyQt6.QtGui import (
    QPixmap,
    QPainter,
    QPainterPath,
    QColor,
    QRadialGradient,
    QBrush,
    QPen,
    QImage,
)

from iartisanxl.modules.common.drop_lightbox import DropLightBox


class ImageEditor(QGraphicsView):
    def __init__(self, parent=None):
        super(ImageEditor, self).__init__(parent)

        self._zoom = 0
        self._empty = True
        self._scene = QGraphicsScene(self)
        self._photo = QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.original_pixmap = None

        self.drawing = False
        self.last_point = QPoint()
        self.brush_color = QColor(0, 0, 0, 255)
        self.brush_size = 32
        self.hardness = 0
        self.undo_stack = []
        self.redo_stack = []
        self.current_drawing = []

        self.drop_lightbox = DropLightBox(self)
        self.drop_lightbox.setText("Drop file here")

    def set_pixmap(self, pixmap: QPixmap):
        self.original_pixmap = pixmap
        self._empty = False
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._photo.setPixmap(pixmap)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.fitInView()

    def set_white_pixmap(self, width, height):
        # Create a white QPixmap
        white_pixmap = QPixmap(width, height)
        white_pixmap.fill(Qt.GlobalColor.white)

        # Set the QPixmap as the photo
        self.original_pixmap = white_pixmap
        self._empty = False
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._photo.setPixmap(self.original_pixmap)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.fitInView()

    def has_photo(self):
        return not self._empty

    def fitInView(self):
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
                    self.setTransformationAnchor(
                        QGraphicsView.ViewportAnchor.AnchorUnderMouse
                    )
                    self.scale(factor, factor)
                elif self._zoom == 0:
                    self.fitInView()
                else:
                    self._zoom = 0
            else:
                super().wheelEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            modifiers = QApplication.keyboardModifiers()
            if modifiers == Qt.KeyboardModifier.ControlModifier:
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                super().mousePressEvent(event)
            else:
                self.drawing = True
                self.last_point = self.mapToScene(event.pos())
                path = QPainterPath()
                path.moveTo(self.last_point)
                path_item = self._scene.addPath(path)
                self.current_drawing.append(path_item)
                self.setCursor(Qt.CursorShape.BlankCursor)

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self.drawing:
            current_point = self.mapToScene(event.pos())

            # Create a radial gradient as the brush
            gradient = QRadialGradient(self.last_point, self.brush_size / 2)
            gradient.setColorAt(0, QColor(0, 0, 0, 255))  # Inner color (black)
            gradient.setColorAt(self.hardness, self.brush_color)  # Outer color (black)
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

            self.last_point = current_point
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.drawing = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.undo_stack.append(self.current_drawing)
            self.current_drawing = []
        else:
            super().mouseReleaseEvent(event)

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

    def keyReleaseEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()
        if key == Qt.Key.Key_Z and modifiers == Qt.KeyboardModifier.ControlModifier:
            self.undo()
        elif key == Qt.Key.Key_Z and modifiers == (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
        ):
            self.redo()
        else:
            super().keyReleaseEvent(event)

    def clear_and_restore(self):
        self._scene.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        original_image_item = QGraphicsPixmapItem(self.original_pixmap)
        self._photo = original_image_item
        self._scene.addItem(original_image_item)

    def get_painted_image(self):
        image = QImage(
            self._scene.sceneRect().size().toSize(), QImage.Format.Format_ARGB32
        )
        image.fill(Qt.GlobalColor.transparent)

        painter = QPainter(image)
        self._scene.render(painter)
        painter.end()

        return image

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
