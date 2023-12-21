from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtCore import Qt, QSize, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QPixmap


class ImageCropper(QGraphicsView):
    imageMoved = pyqtSignal(float, float)
    imageScaled = pyqtSignal(float)

    def __init__(self, original_width: int, original_height: int, aspect_ratio: float):
        super(ImageCropper, self).__init__()

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.original_scale = 1.000
        self.original_x = 0
        self.original_y = 0
        self.original_rotation = 0

        self._mouse_press_pos = None

        self.pixmap_item = None
        self.original_width = original_width
        self.original_height = original_height
        self.aspect_ratio = aspect_ratio

        self.setSceneRect(0, 0, self.original_width, self.original_height)

        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.scene = QGraphicsScene(0, 0, 0, 0)
        self.setScene(self.scene)

    def sizeHint(self):
        return QSize(self.original_width, self.original_height)

    def resizeEvent(self, event):
        rect = self.sceneRect()
        rect.adjust(7, 7, 0, 0)  # Adjust the rect to remove the margin
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        super().resizeEvent(event)

    def set_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.set_pixmap(pixmap)

    def set_pixmap(self, pixmap):
        if self.pixmap_item is not None:
            self.scene.removeItem(self.pixmap_item)
            self.pixmap_item = None

        self.scene.clear()

        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene.addItem(self.pixmap_item)

    def set_image_scale(self, scale_factor):
        if self.pixmap_item is not None:
            self.pixmap_item.setTransformOriginPoint(self.pixmap_item.boundingRect().center())
            self.pixmap_item.setScale(scale_factor)

    def set_image_x(self, x_position):
        if self.pixmap_item is not None:
            self.pixmap_item.setX(x_position)

    def set_image_y(self, y_position):
        if self.pixmap_item is not None:
            self.pixmap_item.setY(y_position)

    def rotate_image(self, angle):
        if self.pixmap_item is not None:
            self.pixmap_item.setTransformOriginPoint(self.pixmap_item.boundingRect().center())
            self.pixmap_item.setRotation(angle)

    def get_current_view_as_pixmap(self):
        pixmap = QPixmap(self.original_width, self.original_height)
        painter = QPainter(pixmap)
        self.render(painter)
        painter.end()
        return pixmap

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
        super(ImageCropper, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            # Calculate the difference between the current mouse position and the last mouse press position
            diff = event.pos() - self._mouse_press_pos
            self._mouse_press_pos = event.pos()

            # Update the position of the pixmap_item
            if self.pixmap_item is not None:
                new_pos = self.pixmap_item.pos() + QPointF(diff)
                self.pixmap_item.setPos(new_pos)
                self.imageMoved.emit(new_pos.x(), new_pos.y())

        super(ImageCropper, self).mouseMoveEvent(event)

    def clear(self):
        if self.pixmap_item is not None:
            self.scene.removeItem(self.pixmap_item)
            self.pixmap_item = None

        self.scene.clear()
