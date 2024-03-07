import cv2
import numpy as np
from PIL import Image
from PyQt6.QtGui import QImage, QPixmap


def convert_pixmap_to_pillow(qpixmap: QPixmap) -> Image.Image:
    qimage = qpixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)

    buffer = qimage.bits().asstring(qimage.sizeInBytes())
    img_size = qimage.size()

    pil_image = Image.frombytes("RGBA", (img_size.width(), img_size.height()), buffer)

    return pil_image


def convert_pillow_to_pixmap(pillow_img):
    raw_data = pillow_img.convert("RGBA").tobytes("raw", "RGBA")
    qimg = QImage(raw_data, pillow_img.width, pillow_img.height, QImage.Format.Format_RGBA8888)
    qpixmap = QPixmap.fromImage(qimg)

    return qpixmap


def convert_pixmap_to_numpy(pixmap: QPixmap) -> np.ndarray:
    image = pixmap.toImage()

    width = image.width()
    height = image.height()

    ptr = image.bits()
    ptr.setsize(height * width * 4)
    numpy_image = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))

    numpy_image = numpy_image.astype(np.uint8)

    return numpy_image


def convert_numpy_argb_to_bgr(arr: np.ndarray) -> np.ndarray:
    rgb_image = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)  # pylint: disable=no-member
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # pylint: disable=no-member

    return bgr_image


def convert_numpy_to_pixmap(numpy_image: np.array):
    qimage = QImage(numpy_image.tobytes(), numpy_image.shape[1], numpy_image.shape[0], QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qimage)

    return pixmap


def convert_to_alpha_image(image: Image.Image) -> Image.Image:
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    return image
