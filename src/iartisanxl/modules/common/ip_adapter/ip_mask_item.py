from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget


class IpMaskItem(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()

        thumb_layout = QVBoxLayout()

        mask_label = QLabel("Mask")
        thumb_layout.addWidget(mask_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.mask_image_label = QLabel()
        self.mask_image_label.setObjectName("mask_label")
        self.mask_image_label.setFixedSize(80, 80)
        thumb_layout.addWidget(self.mask_image_label)

        main_layout.addLayout(thumb_layout)
        self.setLayout(main_layout)

    def set_pixmap(self, thumb_pixmap: QPixmap):
        self.mask_image_label.setPixmap(thumb_pixmap)

    def set_thumb_image(self, thumb_path: str):
        thumb_pixmap = QPixmap(thumb_path)
        self.mask_image_label.setPixmap(thumb_pixmap)
