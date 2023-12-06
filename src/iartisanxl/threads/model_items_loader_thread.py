import base64
import io
import os
from io import BytesIO

from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

from iartisanxl.modules.common.model_utils import get_metadata_from_safetensors
from iartisanxl.app.preferences import PreferencesObject


class ModelItemsLoaderThread(QThread):
    model_item_loaded = pyqtSignal(dict, BytesIO)

    def __init__(
        self,
        model_files: list,
        item_width: int,
        item_height: int,
        default_image: str,
        preferences: PreferencesObject,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.model_files = model_files
        self.item_width = item_width
        self.item_height = item_height
        self.preferences = preferences

        pil_image = Image.open(default_image)
        pil_image.thumbnail((self.item_width, self.item_height), Image.Resampling.LANCZOS)
        self.default_image_buffer = BytesIO()
        pil_image.save(self.default_image_buffer, format="WEBP")

    def run(self):
        for model in self.model_files:
            buffer = None
            model_version = ""

            if model["type"] == "diffusers":
                model_directory = os.path.join(model["filepath"], "text_encoder")
                model_path = os.path.join(model_directory, "model.fp16.safetensors")
            else:
                model_path = model["filepath"]

            metadata = get_metadata_from_safetensors(model_path)
            name = metadata.get("iartisan_name")
            version = metadata.get("iartisan_version")
            image = metadata.get("iartisan_image")
            tags = metadata.get("iartisan_tags")

            if self.preferences.hide_nsfw and tags is not None and "nsfw" in tags:
                continue

            if image is not None:
                img_bytes = base64.b64decode(image)
                buffer = io.BytesIO(img_bytes)
                pil_image = Image.open(buffer)
                buffer.seek(0)
                pil_image.save(buffer, format="WEBP")
            else:
                buffer = self.default_image_buffer

            if name is not None:
                model["name"] = name
            else:
                root_filename = model["root_filename"]
                model["name"] = (root_filename[:20] + "...") if len(root_filename) > 20 else root_filename

            if version is not None:
                model_version = version

            data = {
                "name": model["name"],
                "root_filename": model["root_filename"],
                "version": model_version,
                "filepath": model["filepath"],
                "tags": tags,
                "type": model["type"],
            }

            self.model_item_loaded.emit(data, buffer)
