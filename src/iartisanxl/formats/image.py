import json
import io
import os
import logging

from datetime import datetime

import numpy as np
from PyQt6.QtCore import QByteArray, QBuffer, QIODevice
from PyQt6.QtGui import QPixmap, QImage
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from iartisanxl.generation.generation_data_object import ImageGenData
from iartisanxl.app.preferences import PreferencesObject


class ImageProcessor:
    def __init__(self):
        self.logger = logging.getLogger()
        self.image_data = None
        self.serialized_data = None

    def open_image(self, path: str):
        pil_image = Image.open(path)

        metadata = pil_image.info
        data = metadata.get("data")

        if data is not None:
            self.serialized_data = data

        self.set_pillow_image(pil_image)

    def set_pillow_image(self, pillow_image: Image):
        byte_arr = io.BytesIO()
        pillow_image.save(byte_arr, format="PNG")
        self.image_data = byte_arr.getvalue()

    def set_pixmap(self, pixmap: QPixmap):
        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        pixmap.save(buffer, "PNG")
        self.image_data = byte_array.data()

    def set_qimage(self, qimage: QImage):
        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        qimage.save(buffer, "PNG")
        self.image_data = byte_array.data()

    def set_numpy_array(self, np_array: np.ndarray):
        self.image_data = np_array.tobytes()

    def set_serialized_data(self, serialized_data: json):
        self.serialized_data = serialized_data

    def get_qimage(self) -> QImage:
        byte_array = QByteArray(self.image_data)
        buffer = QBuffer(byte_array)
        buffer.open(QIODevice.OpenModeFlag.ReadOnly)

        qimage = QImage()
        qimage.loadFromData(buffer.data(), "PNG")
        return qimage

    def get_qpixmap(self) -> QPixmap:
        qimage = self.get_qimage()
        qpixmap = QPixmap.fromImage(qimage)
        return qpixmap

    def get_pillow_image(self) -> Image:
        return Image.open(io.BytesIO(self.image_data))

    def save_to_png(self, output_path: str):
        if os.path.isdir(output_path):
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}.png"
            output_filepath = os.path.join(output_path, filename)
        else:
            output_filepath = output_path

        image = Image.open(io.BytesIO(self.image_data))

        if self.serialized_data is None:
            image.save(output_filepath)
            return

        metadata = PngInfo()
        metadata.add_text("data", self.serialized_data)

        image.save(output_filepath, pnginfo=metadata)

    def get_image_generation_data(self) -> ImageGenData:
        try:
            data = json.loads(self.serialized_data)
            data = {key.strip("_"): value for key, value in data.items()}

            image_generation_data = ImageGenData.from_dict(data)

            return image_generation_data
        except json.JSONDecodeError as json_error:
            self.logger.debug("JSONDecodeError exception", exc_info=True)
            raise ValueError("Error decoding JSON from image metadata") from json_error
        except ValueError as data_error:
            self.logger.debug("ValueError exception", exc_info=True)
            raise ValueError("Value error from image metadata") from data_error

    def serialize_image_data(
        self, rendering_generation_data: ImageGenData, preferences: PreferencesObject
    ) -> json:
        data = {
            attr.strip("_"): getattr(rendering_generation_data, attr)
            for attr in rendering_generation_data.__slots__
            if attr not in ("_loras", "_controlnets", "_model")
        }
        data["loras"] = [
            {
                "enabled": lora.enabled,
                "name": lora.name,
                "filename": lora.filename,
                "version": lora.version,
                "path": lora.path,
                "weight": lora.weight,
            }
            for lora in rendering_generation_data.loras
        ]

        data["controlnets"] = []
        for controlnet in rendering_generation_data.controlnets:
            controlnet_dict = {
                "enabled": controlnet.enabled,
                "model_path": controlnet.model_path,
                "name": controlnet.name,
                "guess_mode": controlnet.guess_mode,
                "conditioning_scale": controlnet.conditioning_scale,
                "guidance_start": controlnet.guidance_start,
                "guidance_end": controlnet.guidance_end,
            }

            if preferences.save_image_control_annotators:
                controlnet_dict["annotator_image"] = controlnet.annotator_image_filename

            if preferences.save_image_control_sources:
                controlnet_dict["source_image"] = controlnet.source_image_filename

            data["controlnets"].append(controlnet_dict)

        data["model"] = {
            "name": rendering_generation_data.model.name,
            "path": rendering_generation_data.model.path,
            "type": rendering_generation_data.model.type,
            "version": rendering_generation_data.model.version,
        }
        data["vae"] = {
            "name": rendering_generation_data.vae.name,
            "path": rendering_generation_data.vae.path,
        }
        serialized_data = json.dumps(data)
        return serialized_data
