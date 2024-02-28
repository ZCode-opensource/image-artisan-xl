import os
from io import BytesIO

from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

from iartisanxl.modules.common.ip_adapter.ip_adapter_image import IPAdapterImage


class ImagesLoaderThread(QThread):
    image_loaded = pyqtSignal(BytesIO, IPAdapterImage)
    finished_loading = pyqtSignal()

    def __init__(self, images: list[IPAdapterImage], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.images = images

    def run(self):
        for image_data in self.images:
            pil_image = Image.open(image_data.thumb)

            _, ext = os.path.splitext(image_data.image)
            ext = ext[1:]
            ext = ext.upper()
            if ext == "JPG":
                ext = "JPEG"

            buffer = BytesIO()
            pil_image.save(buffer, format=ext)
            self.image_loaded.emit(buffer, image_data)

        self.finished_loading.emit()
