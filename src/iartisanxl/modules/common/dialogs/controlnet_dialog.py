from importlib.resources import files
from PyQt6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QWidget,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsView,
    QSizePolicy,
    QApplication,
    QSlider,
)
from PyQt6.QtCore import QSettings, Qt, QRectF, QPoint
from PyQt6.QtGui import (
    QPixmap,
    QImageReader,
    QPainter,
    QPainterPath,
    QCursor,
    QPen,
    QColor,
)
from PyQt6.QtSvg import QSvgRenderer
from superqt import QLabeledDoubleRangeSlider

from iartisanxl.modules.common.dialogs.base_dialog import BaseDialog
from iartisanxl.modules.common.drop_lightbox import DropLightBox


class ImageEditor(QGraphicsView):
    CIRCLE_CURSOR = files("iartisanxl.theme.cursors").joinpath("circle.svg")

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

        self.drawing = False
        self.lastPoint = QPoint()
        self.path = None
        self.pathItem = None

        self.drop_lightbox = DropLightBox(self)
        self.drop_lightbox.setText("Drop file here")

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(100)
        self.slider.setValue(32)  # Default value
        self.slider.valueChanged.connect(self.updateCursor)
        layout = QVBoxLayout()
        layout.addWidget(self.slider)
        self.setLayout(layout)

    def setPhoto(self, path):
        reader = QImageReader(path)
        if reader.canRead():
            pixmap = QPixmap(path)
            self._empty = False
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self._photo.setPixmap(QPixmap())
        self.fitInView()

    def hasPhoto(self):
        return not self._empty

    def fitInView(self):
        rect = QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
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
        if self.hasPhoto():
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
                self.lastPoint = self.mapToScene(event.pos())
                self.path = QPainterPath()
                self.path.moveTo(self.lastPoint)
                self.pathItem = self._scene.addPath(self.path)
                size = self.slider.value()
                pixmap = QPixmap(size, size)
                pixmap.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pixmap)
                renderer = QSvgRenderer(str(self.CIRCLE_CURSOR))
                renderer.render(painter)
                painter.end()
                circle_cursor = QCursor(pixmap)
                self.setCursor(circle_cursor)

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self.drawing:
            currentPoint = self.mapToScene(event.pos())
            self.path.lineTo(currentPoint)
            pen = QPen(QColor("black"), self.slider.value())
            self.pathItem.setPen(pen)
            self.pathItem.setPath(self.path)
            self.lastPoint = currentPoint
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.drawing = False
            self.unsetCursor()  # Reset the cursor when you stop drawing
        else:
            super().mouseReleaseEvent(event)

    def updateCursor(self):
        if self.drawing:
            size = self.slider.value()
            pixmap = QPixmap(size, size)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            renderer = QSvgRenderer(str(self.CIRCLE_CURSOR))
            renderer.render(painter)
            painter.end()
            circle_cursor = QCursor(pixmap)
            self.setCursor(circle_cursor)


class ControlImageWidget(QWidget):
    def __init__(
        self,
        text: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.setObjectName("control_image_widget")
        self.text = text
        self.image_path = ""

        self.setAcceptDrops(True)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        source_text_label = QLabel(self.text)
        main_layout.addWidget(source_text_label)
        self.image_editor = ImageEditor()
        main_layout.addWidget(self.image_editor)

        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 11)
        self.setLayout(main_layout)

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
                self.image_path = path
                self.image_editor.setPhoto(self.image_path)


class ControlNetDialog(BaseDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle("ControlNet")
        self.setMinimumSize(1160, 800)

        self.settings = QSettings("ZCode", "ImageArtisanXL")
        self.settings.beginGroup("controlnet_dialog")
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        self.settings.endGroup()

        self.control_guidance_start = 0.0
        self.control_guidance_end = 1.0

        self.init_ui()

    def init_ui(self):
        content_layout = QVBoxLayout()

        top_layout = QHBoxLayout()

        slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 1)
        slider.setValue((self.control_guidance_start, self.control_guidance_end))
        slider.valueChanged.connect(self.on_guidance_changed)

        top_layout.addWidget(slider)
        content_layout.addLayout(top_layout)

        images_layout = QHBoxLayout()
        source_widget = ControlImageWidget("Source image")
        images_layout.addWidget(source_widget)

        annotator_widget = ControlImageWidget("Annotator")
        images_layout.addWidget(annotator_widget)

        content_layout.addLayout(images_layout)

        content_layout.setStretch(0, 2)
        content_layout.setStretch(1, 6)

        self.main_layout.addLayout(content_layout)

    def closeEvent(self, event):
        self.settings.beginGroup("controlnet_dialog")
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.endGroup()
        super().closeEvent(event)

    def on_guidance_changed(self, values):
        print(f"{values=}")
