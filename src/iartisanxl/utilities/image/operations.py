import math
import os
from typing import Union

import cv2
import numpy as np
from PIL import Image

from iartisanxl.modules.common.image.image_data_object import ImageDataObject

from .converters import convert_to_alpha_image


def rotate_scale_crop_image(
    image: Image.Image,
    target_width: int,
    target_height: int,
    angle: float,
    horizontal_scale: float,
    vertical_scale: float,
    x_pos: int,
    y_pos: int,
) -> Image.Image:
    image = rotate_image(image, angle)
    image = scale_image(image, horizontal_scale, vertical_scale)
    image = crop_image(image, target_width, target_height, x_pos, y_pos)

    return image


def rotate_image(image: Image.Image, angle: float) -> Image.Image:
    width, height = image.size

    center = (width / 2, height / 2)
    image = image.rotate(-angle, Image.Resampling.BICUBIC, center=center, expand=True)

    return image


def scale_image(image: Image.Image, horizontal_scale: float, vertical_scale: float) -> Image.Image:
    width, height = image.size

    new_width = round(width * horizontal_scale)
    new_height = round(height * vertical_scale)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image


def crop_image(image: Image.Image, target_width: int, target_height: int, x_pos: int, y_pos: int) -> Image.Image:
    width, height = image.size

    left = math.floor(width / 2 - (target_width / 2 + x_pos))
    top = math.floor(height / 2 - (target_width / 2 + y_pos))
    right = target_width + left
    bottom = target_height + top
    image = image.crop((left, top, right, bottom))

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
