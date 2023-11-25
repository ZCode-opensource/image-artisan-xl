from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QPixmap

from iartisanxl.formats.image import ImageProcessor


class ImageProcesorThread(QThread):
    image_loaded = pyqtSignal(QPixmap)
    serialized_data_obtained = pyqtSignal(str)
    image_error = pyqtSignal(str)
    status_changed = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()

        self.path = path

    def run(self):
        self.status_changed.emit("Getting generation data from image...")

        image = ImageProcessor()
        image.open_image(self.path)

        self.status_changed.emit(
            "Setting up generation from metada found in the image..."
        )

        self.serialized_data_obtained.emit(image.serialized_data)
        pixmap = image.get_qpixmap()
        self.image_loaded.emit(pixmap)
