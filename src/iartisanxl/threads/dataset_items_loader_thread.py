import os
from io import BytesIO

from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal


class DatasetItemsLoaderThread(QThread):
    image_loaded = pyqtSignal(BytesIO, str)
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

            pil_image = Image.open(image_path)
            pil_image.thumbnail((self.item_width, self.item_height), Image.Resampling.LANCZOS)

            _, ext = os.path.splitext(image)
            ext = ext[1:]
            ext = ext.upper()
            if ext == "JPG":
                ext = "JPEG"

            buffer = BytesIO()
            pil_image.save(buffer, format=ext)
            self.image_loaded.emit(buffer, image_path)

        self.finished_loading.emit()
