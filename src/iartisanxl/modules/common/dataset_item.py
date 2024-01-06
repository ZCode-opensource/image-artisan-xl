from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import pyqtSignal, Qt


class DatasetItem(QFrame):
    clicked = pyqtSignal(object)

    COLORS = ["#252629", "#374344", "#56585f"]

    def __init__(self, width: int, height: int, path: str, pixmap: QPixmap, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setFixedWidth(width)
        self.setFixedHeight(height)

        self.path = path
        self.pixmap = pixmap
        self.selected = False

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(0)

        self.image_label = QLabel()
        self.image_label.setPixmap(self.pixmap)
        main_layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignVCenter)

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

    def set_image(self, path: str, pixmap: QPixmap):
        self.path = path
        self.image_label.setPixmap(pixmap)
