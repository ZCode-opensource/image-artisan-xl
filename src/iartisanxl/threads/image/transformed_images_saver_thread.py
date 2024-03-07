import os
from datetime import datetime

from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

from iartisanxl.modules.common.image.image_editor_layer import ImageEditorLayer
from iartisanxl.utilities.image.converters import convert_pixmap_to_pillow
from iartisanxl.utilities.image.operations import generate_thumbnail, merge_images, transform_image


class TransformedImagesSaverThread(QThread):
    error = pyqtSignal(str)
    merge_finished = pyqtSignal(str, str)

    def __init__(
        self,
        layers: list[ImageEditorLayer],
        target_width: int,
        target_height: int,
        prefix: str = "img",
    ):
        super().__init__()

        self.layers = layers
        self.target_width = target_width
        self.target_height = target_height
        self.prefix = prefix
        self.temp_path = "tmp"

    def run(self):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # make the list of images all pillow images
        self.images = [
            self.process_image(layer, timestamp) for layer in sorted(self.layers, key=lambda layer: layer.order)
        ]

        source_path = None
        thumb_path = None

        try:
            merged_image = merge_images(self.images)
            source_filename = f"{self.prefix}_{timestamp}.png"
            source_path = os.path.join("tmp/", source_filename)
            merged_image.save(source_path)

            thumb_filename = f"{self.prefix}_{timestamp}_thumb.png"
            thumb_path = os.path.join(self.temp_path, thumb_filename)
            generate_thumbnail(merged_image, 80, 80, thumb_path)
        except ValueError as e:
            self.error.emit(str(e))

        self.merge_finished.emit(source_path, thumb_path)

    def process_image(self, layer: ImageEditorLayer, timestamp: str) -> Image.Image:
        image = convert_pixmap_to_pillow(layer.pixmap_item.pixmap())

        if layer.original_path is None:
            original_filename = f"{self.prefix}_{timestamp}_{layer.layer_id}_original.png"
            original_path = os.path.join(self.temp_path, original_filename)
            image.save(original_path)

        image = transform_image(
            image,
            self.target_width,
            self.target_height,
            layer.pixmap_item.rotation(),
            layer.pixmap_item.scale(),
            layer.pixmap_item.x(),
            layer.pixmap_item.y(),
        )
        filename = f"{self.prefix}_{timestamp}_{layer.layer_id}.png"
        image_path = os.path.join(self.temp_path, filename)
        image.save(image_path)

        if layer.image_path is not None and os.path.isfile(layer.image_path):
            os.remove(layer.image_path)
        layer.image_path = image_path

        return image
