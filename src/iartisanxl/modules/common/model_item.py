from io import BytesIO
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal, QSize

from iartisanxl.modules.common.model_image_widget import ModelImageWidget


class ModelItem(QWidget):
    clicked = pyqtSignal()

    def __init__(
        self,
        data: dict,
        image_bytes: BytesIO,
        item_width: int,
        item_height: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.setFixedSize(item_width, item_height)

        self.data = data
        self.image_bytes = image_bytes

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.image_widget = ModelImageWidget(
            self.image_bytes,
            self.data.get("name"),
            self.data.get("version"),
            self.data.get("type"),
        )
        self.image_widget.setFixedSize(QSize(150, 150))
        layout.addWidget(self.image_widget)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)
