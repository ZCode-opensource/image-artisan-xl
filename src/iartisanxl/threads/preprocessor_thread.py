import os
import math
from datetime import datetime

import cv2
from PIL import Image
import numpy as np
from PyQt6.QtCore import pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap

from iartisanxl.preprocessors.canny.canny_edges_detector import CannyEdgesDetector
from iartisanxl.preprocessors.depth.depth_estimator import DepthEstimator
from iartisanxl.preprocessors.openpose.open_pose_detector import OpenPoseDetector
from iartisanxl.modules.common.controlnet.controlnet_data_object import ControlNetDataObject
from iartisanxl.modules.common.image.image_data_object import ImageDataObject


class PreprocessorThread(QThread):
    error = pyqtSignal(str)

    def __init__(
        self,
        controlnet: ControlNetDataObject,
        drawings_pixmap: QPixmap,
        source_changed: bool,
        preprocess: bool,
        save_preprocessor: bool = False,
        preprocessor_drawings: QPixmap = None,
    ):
        super().__init__()

        self.source_thumb_pixmap = None
        self.preprocessor_pixmap = None
        self.controlnet = controlnet
        self.drawings_pixmap = drawings_pixmap
        self.source_changed = source_changed
        self.preprocess = preprocess
        self.save_preprocessor = save_preprocessor
        self.preprocessor_drawings = preprocessor_drawings
        self.preprocessor_thumb_path = None

        self.canny_detector = None
        self.depth_estimator = None
        self.openpose_detector = None

    def run(self):
        source_image = None
        preprocessor_name = ""

        source_resolution = (
            int(self.controlnet.generation_width * self.controlnet.preprocessor_resolution),
            int(self.controlnet.generation_height * self.controlnet.preprocessor_resolution),
        )

        if self.source_changed:
            if self.controlnet.source_image.image_filename:
                os.remove(self.controlnet.source_image.image_filename)

                if self.controlnet.source_image.image_thumb and os.path.isfile(self.controlnet.source_image.image_thumb):
                    os.remove(self.controlnet.source_image.image_thumb)
                    self.controlnet.source_image.image_thumb = None

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            drawings_filename = f"cn_source_{timestamp}_drawings.png"
            drawings_path = os.path.join("tmp/", drawings_filename)
            self.drawings_pixmap.save(drawings_path)
            if self.controlnet.source_image.image_drawings and os.path.isfile(self.controlnet.source_image.image_drawings):
                os.remove(self.controlnet.source_image.image_drawings)
            self.controlnet.source_image.image_drawings = drawings_path

            source_filename = f"cn_source_{timestamp}.png"
            source_path = os.path.join("tmp/", source_filename)
            source_image = self.prepare_image(self.controlnet.source_image)
            source_image.save(source_path)
            self.controlnet.source_image.image_filename = source_path
            source_image = source_image.convert("RGB")
            numpy_image = np.array(source_image)
        else:
            if self.controlnet.source_image.image_filename:
                numpy_image = cv2.imread(self.controlnet.source_image.image_filename)  # pylint: disable=no-member
                numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGRA2RGB)  # pylint: disable=no-member

        if self.controlnet.source_image.image_filename and self.controlnet.source_image.image_thumb is None:
            thumb_filename = os.path.basename(self.controlnet.source_image.image_filename)
            thumb_name, thumb_extension = os.path.splitext(thumb_filename)
            thumb_final_filename = f"{thumb_name}_thumb{thumb_extension}"
            thumb_path = os.path.join("tmp/", thumb_final_filename)
            source_numpy_thumb = self.generate_thumbnail(numpy_image)
            cv2.imwrite(thumb_path, source_numpy_thumb)  # pylint: disable=no-member
            self.controlnet.source_image.image_thumb = thumb_path

        preprocessor_image = None
        preprocessor_loaded = False

        if self.preprocess:
            if self.controlnet.type_index == 0:
                preprocessor_name = "canny"
                if self.canny_detector is None:
                    self.depth_estimator = None
                    self.openpose_detector = None
                    self.canny_detector = CannyEdgesDetector()

                preprocessor_image = self.canny_detector.get_canny_edges(
                    numpy_image, self.controlnet.canny_low, self.controlnet.canny_high, resolution=source_resolution
                )
            elif self.controlnet.type_index == 1:
                preprocessor_name = "depth"
                try:
                    if self.depth_estimator is None:
                        self.canny_detector = None
                        self.openpose_detector = None
                        self.depth_estimator = DepthEstimator(self.controlnet.depth_type)

                    self.depth_estimator.change_model(self.controlnet.depth_type)
                except OSError:
                    self.error.emit("You need to download the preprocessors from the downloader menu first.")
                    return

                preprocessor_image = self.depth_estimator.get_depth_map(numpy_image, source_resolution)
            elif self.controlnet.type_index == 2:
                preprocessor_name = "pose"
                try:
                    if self.openpose_detector is None:
                        self.canny_detector = None
                        self.depth_estimator = None
                        self.openpose_detector = OpenPoseDetector()
                except FileNotFoundError:
                    self.error.emit("You need to download the preprocessors from the downloader menu first.")
                    return

                preprocessor_image = self.openpose_detector.get_open_pose(numpy_image, source_resolution)

            if preprocessor_image is not None:
                if self.save_preprocessor:
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    preprocessor_filename = f"cn_{preprocessor_name}_{timestamp}_original.png"
                    preprocessor_path = os.path.join("tmp/", preprocessor_filename)
                    cv2.imwrite(preprocessor_path, preprocessor_image)  # pylint: disable=no-member

                    if self.controlnet.preprocessor_image.image_original is not None and os.path.isfile(self.controlnet.preprocessor_image.image_original):
                        os.remove(self.controlnet.preprocessor_image.image_original)

                    self.controlnet.preprocessor_image.image_original = preprocessor_path
                    self.preprocessor_pixmap = QPixmap(preprocessor_path)
                else:
                    self.preprocessor_pixmap = self.convert_numpy_to_pixmap(preprocessor_image)
        else:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            drawings_filename = f"cn_{preprocessor_name}_{timestamp}_drawings.png"
            drawings_path = os.path.join("tmp/", drawings_filename)
            self.preprocessor_drawings.save(drawings_path)
            if self.controlnet.preprocessor_image.image_drawings and os.path.isfile(self.controlnet.preprocessor_image.image_drawings):
                os.remove(self.controlnet.preprocessor_image.image_drawings)
            self.controlnet.preprocessor_image.image_drawings = drawings_path

            preprocessor_filename = f"cn_{preprocessor_name}_{timestamp}.png"
            preprocessor_path = os.path.join("tmp/", preprocessor_filename)
            preprocessor_image = self.prepare_image(self.controlnet.preprocessor_image)
            preprocessor_image.save(preprocessor_path)
            self.controlnet.preprocessor_image.image_filename = preprocessor_path

            thumb_image = preprocessor_image.copy()
            thumb_image.thumbnail((80, 80))
            preprocessor_thumb_filename = f"cn_{preprocessor_name}_{timestamp}_thumb.png"
            self.controlnet.preprocessor_image.image_thumb = os.path.join("tmp/", preprocessor_thumb_filename)
            thumb_image.save(self.controlnet.preprocessor_image.image_thumb)

            preprocessor_loaded = True

        if self.preprocessor_pixmap and not preprocessor_loaded:
            if self.save_preprocessor:
                if self.controlnet.preprocessor_image.image_filename:
                    os.remove(self.controlnet.preprocessor_image.image_filename)

                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

                drawings_filename = f"cn_{preprocessor_name}_{timestamp}_drawings.png"
                drawings_path = os.path.join("tmp/", drawings_filename)
                self.preprocessor_drawings.save(drawings_path)
                if self.controlnet.preprocessor_image.image_drawings and os.path.isfile(self.controlnet.preprocessor_image.image_drawings):
                    os.remove(self.controlnet.preprocessor_image.image_drawings)
                self.controlnet.preprocessor_image.image_drawings = drawings_path

                preprocessor_filename = f"cn_{preprocessor_name}_{timestamp}.png"
                self.controlnet.preprocessor_image.image_filename = os.path.join("tmp/", preprocessor_filename)

                if preprocessor_image is not None:
                    preprocessor_drawing_image = Image.open(self.controlnet.preprocessor_image.image_drawings)
                    preprocessor_pil_image = Image.fromarray(preprocessor_image)
                    if preprocessor_pil_image.mode not in ("RGBA", "LA", "P"):
                        preprocessor_pil_image = preprocessor_pil_image.convert("RGBA")
                    merged_preprocessor = Image.alpha_composite(preprocessor_pil_image, preprocessor_drawing_image)
                    merged_preprocessor.save(self.controlnet.preprocessor_image.image_filename)

                if self.controlnet.preprocessor_image.image_thumb:
                    os.remove(self.controlnet.preprocessor_image.image_thumb)

                preprocessor_thumb_filename = f"cn_{preprocessor_name}_{timestamp}_thumb.png"
                self.controlnet.preprocessor_image.image_thumb = os.path.join("tmp/", preprocessor_thumb_filename)
                merged_preprocessor.thumbnail((80, 80))
                merged_preprocessor.save(self.controlnet.preprocessor_image.image_thumb)

    def generate_thumbnail(self, numpy_image: np.array):
        height, width = numpy_image.shape[:2]

        thumbnail_width = 80
        thumbnail_height = 80

        aspect_ratio = width / height

        if width > height:
            new_width = thumbnail_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = thumbnail_height
            new_width = int(new_height * aspect_ratio)

        thumb_numpy_image = cv2.resize(numpy_image, (new_width, new_height), interpolation=cv2.INTER_AREA)  # pylint: disable=no-member
        thumb_numpy_image = cv2.cvtColor(thumb_numpy_image, cv2.COLOR_BGRA2RGBA)  # pylint: disable=no-member

        return thumb_numpy_image

    def convert_numpy_to_pixmap(self, numpy_image: np.array):
        qimage = QImage(
            numpy_image.tobytes(),
            numpy_image.shape[1],
            numpy_image.shape[0],
            QImage.Format.Format_RGB888,
        )

        pixmap = QPixmap.fromImage(qimage)

        return pixmap

    def prepare_image(self, image_data: ImageDataObject):
        original_pil_image = Image.open(image_data.image_original)
        width, height = original_pil_image.size

        if original_pil_image.mode not in ("RGBA", "LA", "P"):
            original_pil_image = original_pil_image.convert("RGBA")

        center = (width / 2, height / 2)
        original_pil_image = original_pil_image.rotate(-image_data.image_rotation, Image.Resampling.BICUBIC, center=center, expand=True)

        new_width = round(width * image_data.image_scale)
        new_height = round(height * image_data.image_scale)
        original_pil_image = original_pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        left = math.floor(new_width / 2 - (width / 2 + image_data.image_x_pos))
        top = math.floor(new_height / 2 - (height / 2 + image_data.image_y_pos))
        right = self.controlnet.generation_width + left
        bottom = self.controlnet.generation_height + top
        original_pil_image = original_pil_image.crop((left, top, right, bottom))

        drawing_image = Image.open(image_data.image_drawings)
        merged_image = Image.alpha_composite(original_pil_image, drawing_image)

        return merged_image
