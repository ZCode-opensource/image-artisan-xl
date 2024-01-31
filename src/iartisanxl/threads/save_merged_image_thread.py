import os
import math
import io
from datetime import datetime

from PIL import Image
from PyQt6.QtCore import pyqtSignal, QThread, QByteArray, QBuffer, QIODevice
from PyQt6.QtGui import QPixmap

from iartisanxl.modules.common.image.image_data_object import ImageDataObject


class SaveMergedImageThread(QThread):
    error = pyqtSignal(str)
    image_done = pyqtSignal(str)

    def __init__(self, image_width, image_height, image_data: ImageDataObject, drawings_pixmap: QPixmap, save_path: str = None):
        super().__init__()

        self.image_width = image_width
        self.image_height = image_height
        self.image_data = image_data
        self.drawings_pixmap = drawings_pixmap
        self.save_path = save_path

    def run(self):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        original_pil_image = Image.open(self.image_data.image_original)
        width, height = original_pil_image.size

        if original_pil_image.mode not in ("RGBA", "LA", "P"):
            original_pil_image = original_pil_image.convert("RGBA")

        center = (width / 2, height / 2)
        original_pil_image = original_pil_image.rotate(-self.image_data.image_rotation, Image.Resampling.BICUBIC, center=center, expand=True)

        new_width = round(width * self.image_data.image_scale)
        new_height = round(height * self.image_data.image_scale)
        original_pil_image = original_pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        left = math.floor(new_width / 2 - (width / 2 + self.image_data.image_x_pos))
        top = math.floor(new_height / 2 - (height / 2 + self.image_data.image_y_pos))
        right = self.image_width + left
        bottom = self.image_height + top
        original_pil_image = original_pil_image.crop((left, top, right, bottom))

        qimage = self.drawings_pixmap.toImage()
        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        qimage.save(buffer, "PNG")
        drawing_image = Image.open(io.BytesIO(byte_array.data()))

        merged_image = Image.alpha_composite(original_pil_image, drawing_image)

        if self.save_path:
            merged_image.save(self.save_path)
        else:
            image_filename = f"copy_image_{timestamp}.png"
            image_path = os.path.join("tmp/", image_filename)
            merged_image.save(image_path)
            self.image_done.emit(image_path)
