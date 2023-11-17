from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QPixmap

from iartisanxl.formats.image import ImageProcessor
from iartisanxl.generation.generation_data_object import ImageGenData


class ImageProcesorThread(QThread):
    image_loaded = pyqtSignal(QPixmap)
    generation_data_obtained = pyqtSignal(ImageGenData)
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

        if image.serialized_data is None:
            self.image_error.emit("No metadata found in the image.")
            self.status_changed.emit("No metadata found in the image.")
        else:
            try:
                image_generation_data = image.get_image_generation_data()
                self.generation_data_obtained.emit(image_generation_data)

                pixmap = image.get_qpixmap()
                self.image_loaded.emit(pixmap)
            except ValueError as e:
                self.image_error.emit(f"{e}")
