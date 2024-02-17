import os
from typing import Union

import cv2
import numpy as np
from PIL import Image

from iartisanxl.modules.common.image.image_data_object import ImageDataObject

from .converters import convert_to_alpha_image


def transform_image(
    image: Image.Image,
    target_width: int,
    target_height: int,
    angle: float,
    scale: float,
    x_pos: int,
    y_pos: int,
) -> Image.Image:
    width, height = image.size
    original_center = (width / 2, height / 2)

    # scale the image
    new_width = round(width * scale)
    new_height = round(height * scale)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    scaled_center = (new_width / 2, new_height / 2)
    diff_center = (original_center[0] - scaled_center[0], original_center[1] - scaled_center[1])

    # crop the image
    left = -diff_center[0] - x_pos
    top = -diff_center[1] - y_pos
    right = target_width - diff_center[0] - x_pos
    bottom = target_height - diff_center[1] - y_pos
    image = image.crop((left, top, right, bottom))

    # Rotate the image
    center = (width / 2, height / 2)
    image = image.rotate(-angle, Image.Resampling.BICUBIC, center=center, expand=False)

    return image


def merge_images(images: list[Image.Image]) -> Image.Image:
    size = images[0].size

    for i, image in enumerate(images):
        if image.size != size:
            raise ValueError("All images must be the same size")

        images[i] = convert_to_alpha_image(image)

    merged_image = Image.new("RGBA", size)

    for image in images:
        merged_image = Image.alpha_composite(merged_image, image)

    return merged_image


def generate_thumbnail(
    image: Union[Image.Image, np.ndarray], thumbnail_width: int, thumbnail_height: int, save_path: str
):
    # Check if the input is a Pillow Image
    if isinstance(image, Image.Image):
        thumb_image = image.copy()
        thumb_image.thumbnail((thumbnail_width, thumbnail_height), Image.Resampling.LANCZOS)
        thumb_image.save(save_path)
    # Otherwise, assume it's a numpy array
    else:
        height, width = image.shape[:2]
        numpy_image = image.copy()

        aspect_ratio = width / height

        if width > height:
            new_width = thumbnail_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = thumbnail_height
            new_width = int(new_height * aspect_ratio)

        thumb_numpy_image = cv2.resize(numpy_image, (new_width, new_height), interpolation=cv2.INTER_AREA)  # pylint: disable=no-member
        cv2.imwrite(save_path, cv2.cvtColor(thumb_numpy_image, cv2.COLOR_RGBA2BGRA))  # pylint: disable=no-member


def remove_image_data_files(image_data: ImageDataObject):
    attributes = ["image_original", "image_filename", "image_thumb", "image_drawings"]
    for attr in attributes:
        file_path = getattr(image_data, attr, None)
        if file_path:
            os.remove(file_path)
