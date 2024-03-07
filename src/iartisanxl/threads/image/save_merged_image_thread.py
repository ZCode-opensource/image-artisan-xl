import os
from datetime import datetime

from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

from iartisanxl.modules.common.image.image_editor_layer import ImageEditorLayer
from iartisanxl.utilities.image.converters import convert_pixmap_to_pillow
from iartisanxl.utilities.image.operations import merge_images, transform_image


class SaveMergedImageThread(QThread):
    error = pyqtSignal(str)
    image_done = pyqtSignal(str)

    def __init__(self, layers: list[ImageEditorLayer], target_width: int, target_height: int, save_path: str = None):
        super().__init__()

        self.layers = layers
        self.target_width = target_width
        self.target_height = target_height
        self.save_path = save_path

    def run(self):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # make the list of images all pillow images
        self.images = [self.process_image(layer) for layer in sorted(self.layers, key=lambda layer: layer.order)]

        image_path = None

        try:
            merged_image = merge_images(self.images)

            if self.save_path:
                merged_image.save(self.save_path)
            else:
                image_filename = f"copy_image_{timestamp}.png"
                image_path = os.path.join("tmp/", image_filename)
                merged_image.save(image_path)
        except ValueError as e:
            self.error.emit(str(e))

        self.image_done.emit(image_path)

    def process_image(self, layer: ImageEditorLayer) -> Image.Image:
        image = convert_pixmap_to_pillow(layer.pixmap_item.pixmap())

        image = transform_image(
            image,
            self.target_width,
            self.target_height,
            layer.pixmap_item.rotation(),
            layer.pixmap_item.scale(),
            layer.pixmap_item.x(),
            layer.pixmap_item.y(),
        )

        return image
