import os

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap


class DatasetItemsLoaderThread(QThread):
    image_loaded = pyqtSignal(str, QPixmap)
    finished_loading = pyqtSignal()

    def __init__(self, images: list, path, item_width: int, item_height: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.images = images
        self.path = path
        self.item_width = item_width
        self.item_height = item_height

    def run(self):
        for image in self.images:
            image_path = os.path.join(self.path, image)

            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(
                self.item_width, self.item_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )

            self.image_loaded.emit(image_path, scaled_pixmap)

        self.finished_loading.emit()
