from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import pyqtSignal, Qt

from iartisanxl.modules.common.image.image_data_object import ImageDataObject


class ImageItem(QFrame):
    clicked = pyqtSignal(object)

    COLORS = ["#252629", "#374344", "#56585f"]

    def __init__(self, image_data: ImageDataObject, pixmap: QPixmap, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_data = image_data
        self.pixmap = pixmap
        self.selected = False

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(0)

        self.image_label = QLabel()
        self.image_label.setPixmap(self.pixmap)
        main_layout.addWidget(self.image_label)

        self.setLayout(main_layout)
        self.set_background_color(0)

    def set_background_color(self, index=None):
        if index is None:
            index = 2 if self.selected else 0
        color = self.COLORS[index]
        self.setStyleSheet(f"background-color: {color}")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.set_selected(True)
            self.clicked.emit(self)
        super().mousePressEvent(event)

    def enterEvent(self, event):
        if not self.selected:
            self.set_background_color(1)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if not self.selected:
            self.set_background_color(0)
        super().leaveEvent(event)

    def set_selected(self, selected: bool):
        self.selected = selected
        self.set_background_color()

    def set_image(self, pixmap):
        self.image_label.setPixmap(pixmap)
