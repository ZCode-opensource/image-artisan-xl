import os
from datetime import datetime
from typing import List, Union

from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QPixmap

from iartisanxl.utilities.image.converters import convert_pixmap_to_pillow, convert_to_alpha_image
from iartisanxl.utilities.image.operations import generate_thumbnail, merge_images, rotate_scale_crop_image


class TransformedImagesSaverThread(QThread):
    error = pyqtSignal(str)
    merge_finished = pyqtSignal(str, str)

    def __init__(
        self,
        images: List[Union[str, Image.Image, QPixmap]],
        target_width: int,
        target_height: int,
        angle: float,
        horizontal_scale: int,
        vertical_scale: int,
        x_pos: int,
        y_pos: int,
        prefix: str = "img",
    ):
        super().__init__()

        self.images = images
        self.target_width = target_width
        self.target_height = target_height
        self.angle = angle
        self.horizontal_scale = horizontal_scale
        self.vertical_scale = vertical_scale
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.prefix = prefix

    def run(self):
        # make the list of images all pillow images
        self.images = [self.process_image(image) for image in self.images]

        source_path = None
        thumb_path = None

        try:
            merged_image = merge_images(self.images)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            source_filename = f"{self.prefix}_{timestamp}.png"
            source_path = os.path.join("tmp/", source_filename)
            merged_image.save(source_path)

            thumb_filename = f"{self.prefix}_{timestamp}_thumb.png"
            thumb_path = os.path.join("tmp/", thumb_filename)
            generate_thumbnail(merged_image, 80, 80, thumb_path)
        except ValueError as e:
            self.error.emit(str(e))

        self.merge_finished.emit(source_path, thumb_path)

    def process_image(self, image: Union[str, Image.Image, QPixmap]) -> Image.Image:
        if isinstance(image, str):
            image = convert_to_alpha_image(Image.open(image))
        elif isinstance(image, QPixmap):
            image = convert_pixmap_to_pillow(image)

        image = rotate_scale_crop_image(
            image,
            self.target_width,
            self.target_height,
            self.angle,
            self.horizontal_scale,
            self.vertical_scale,
            self.x_pos,
            self.y_pos,
        )

        return image
