import os
import json
import math

from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap

from iartisanxl.modules.common.image.image_adder_preview import ImageAdderPreview


class DatasetItemSaverThread(QThread):
    image_saved = pyqtSignal(str, QPixmap)

    def __init__(
        self,
        dataset_dir: str,
        originals_dir: str,
        filename: str,
        aspect_index: int,
        thumb_width: int,
        thumb_height: int,
        image_editor: ImageAdderPreview,
        *args,
        has_original: bool = False,
        upscaled: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.filename = filename
        self.aspect_index = aspect_index
        self.dataset_dir = dataset_dir
        self.originals_dir = originals_dir
        self.thumb_width = thumb_width
        self.thumb_height = thumb_height
        self.image_editor = image_editor
        self.has_original = has_original
        self.upscaled = upscaled

    def run(self):
        pos_x = self.image_editor.pixmap_item.x()
        pos_y = self.image_editor.pixmap_item.y()
        angle = self.image_editor.pixmap_item.rotation()
        scale = self.image_editor.pixmap_item.scale()

        # check if image has an original
        if self.has_original:
            if self.upscaled:
                old_image = os.path.join(self.dataset_dir, self.filename)
                old_captions = os.path.join(self.dataset_dir, os.path.splitext(self.filename)[0] + ".txt")
                old_json = os.path.join(self.originals_dir, os.path.splitext(self.filename)[0] + ".json")
                os.remove(old_image)
                os.remove(old_json)

                if os.path.isfile(old_captions):
                    os.remove(old_captions)

                name = os.path.splitext(self.filename)[0]
                self.filename = f"{name}_upscaled.jpg"

            image_path = os.path.join(self.originals_dir, self.filename)
            pil_image = Image.open(image_path)

            # save the params in a json file
            json_path = os.path.splitext(image_path)[0] + ".json"
            params = {"pos_x": pos_x, "pos_y": pos_y, "angle": angle, "scale": scale, "aspect_index": self.aspect_index}

            with open(json_path, "w", encoding="utf-8") as json_file:
                json.dump(params, json_file)
        else:
            # if not we just open the dataset image
            pil_image = Image.open(os.path.join(self.dataset_dir, self.filename))

        width, height = pil_image.size
        if pil_image.mode not in ("RGBA"):
            pil_image = pil_image.convert("RGB")

        center = (width / 2, height / 2)
        pil_image = pil_image.rotate(-angle, Image.Resampling.BICUBIC, center=center, expand=True)

        new_width = round(self.image_editor.pixmap_item.sceneBoundingRect().width())
        new_height = round(self.image_editor.pixmap_item.sceneBoundingRect().height())
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

        left = math.floor(new_width / 2 - (width / 2 + pos_x))
        top = math.floor(new_height / 2 - (height / 2 + pos_y))
        right = round(self.image_editor.mapToScene(self.image_editor.viewport().rect()).boundingRect().width()) + left
        bottom = round(self.image_editor.mapToScene(self.image_editor.viewport().rect()).boundingRect().height()) + top
        pil_image = pil_image.crop((left, top, right, bottom))

        final_image_path = os.path.join(self.dataset_dir, self.filename)
        pil_image.save(final_image_path)

        # generate a thumb pixmap
        pixmap = QPixmap(final_image_path)
        scaled_pixmap = pixmap.scaled(self.thumb_width, self.thumb_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_saved.emit(final_image_path, scaled_pixmap)
