from io import BytesIO

from PyQt6.QtWidgets import QWidget, QLabel, QHBoxLayout
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QSize, QPropertyAnimation, QPoint


class ModelImageWidget(QWidget):
    def __init__(
        self,
        image_bytes: BytesIO,
        model_name: str,
        model_version: str,
        model_type: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.model_name = model_name
        self.model_version = model_version
        self.model_type = model_type
        qimage = QImage.fromData(image_bytes.getvalue())
        self.pixmap = QPixmap.fromImage(qimage)

        self.init_ui()

        self.type_animation = QPropertyAnimation(self.type_label, b"pos")
        self.version_animation = QPropertyAnimation(self.version_label, b"pos")
        self.name_animation = QPropertyAnimation(self.name_label, b"pos")

        self.setMouseTracking(True)

    def init_ui(self):
        self.image_label = QLabel(self)
        self.image_label.setPixmap(self.pixmap)

        self.top_widget = QWidget(self)
        top_layout = QHBoxLayout(self.top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)
        self.type_label = QLabel(self.model_type)
        self.type_label.setObjectName("item_type")
        top_layout.addWidget(self.type_label, alignment=Qt.AlignmentFlag.AlignLeft)
        self.version_label = QLabel(self.model_version)
        self.version_label.setObjectName("item_version")
        top_layout.addWidget(self.version_label, alignment=Qt.AlignmentFlag.AlignRight)
        self.top_widget.raise_()

        self.name_label = QLabel(self.model_name, self)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setObjectName("item_name")
        self.name_label.raise_()

        if self.model_version is None or len(self.model_version) == 0:
            self.version_label.setVisible(False)

        self.top_widget.move(0, 0)
        self.name_label.move(0, self.height() - self.name_label.height())

    def enterEvent(self, _event):
        self.type_animation.stop()
        self.version_animation.stop()
        self.name_animation.stop()

        self.type_animation.setDuration(250)
        self.version_animation.setDuration(250)
        self.name_animation.setDuration(250)

        self.type_animation.setEndValue(
            QPoint(
                -self.type_label.width(),
                0,
            )
        )
        self.version_animation.setEndValue(
            QPoint(
                self.top_widget.width(),
                0,
            )
        )
        self.name_animation.setEndValue(QPoint(0, self.height()))

        self.type_animation.start()
        self.version_animation.start()
        self.name_animation.start()

    def leaveEvent(self, _event):
        self.type_animation.stop()
        self.version_animation.stop()
        self.name_animation.stop()

        self.type_animation.setDuration(250)
        self.version_animation.setDuration(250)
        self.name_animation.setDuration(250)

        self.type_animation.setEndValue(QPoint(0, 0))
        self.version_animation.setEndValue(
            QPoint(self.top_widget.width() - self.version_label.width(), 0)
        )
        self.name_animation.setEndValue(
            QPoint(0, self.height() - self.name_label.height())
        )

        self.type_animation.start()
        self.version_animation.start()
        self.name_animation.start()

    def resizeEvent(self, event):
        self.top_widget.setFixedWidth(self.width())
        self.name_label.setFixedSize(QSize(self.height(), 20))
        self.name_label.move(0, self.height() - self.name_label.height())
        self.image_label.resize(event.size())
        pixmap = self.image_label.pixmap()
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled_pixmap)

        if not self.underMouse():
            self.top_widget.move(0, 0)
            self.name_label.move(0, self.height() - self.name_label.height())

    def set_model_version(self, version: str):
        if len(version) > 0:
            self.version_label.setText(version)
            self.version_label.setVisible(True)
