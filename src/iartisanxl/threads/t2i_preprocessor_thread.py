import os
import math
from datetime import datetime

import cv2
from PIL import Image, ImageQt
import numpy as np
from PyQt6.QtCore import pyqtSignal, QThread, Qt
from PyQt6.QtGui import QImage, QPixmap

from iartisanxl.preprocessors.canny.canny_edges_detector import CannyEdgesDetector
from iartisanxl.preprocessors.depth.depth_estimator import DepthEstimator
from iartisanxl.preprocessors.openpose.open_pose_detector import OpenPoseDetector
from iartisanxl.preprocessors.lineart.lineart_generator import LineArtGenerator
from iartisanxl.preprocessors.pidinet.pidinet_generator import PidinetGenerator
from iartisanxl.modules.common.t2i_adapter.t2i_adapter_data_object import T2IAdapterDataObject


class T2IPreprocessorThread(QThread):
    error = pyqtSignal(str)

    def __init__(
        self,
        adapter: T2IAdapterDataObject,
        drawings_pixmap: QPixmap,
        source_changed: bool,
        preprocess: bool,
        save_preprocessor: bool = False,
        preprocessor_drawings: QPixmap = None,
    ):
        super().__init__()

        self.source_thumb_pixmap = None
        self.preprocessor_pixmap = None
        self.adapter = adapter
        self.drawings_pixmap = drawings_pixmap
        self.source_changed = source_changed
        self.preprocess = preprocess
        self.save_preprocessor = save_preprocessor
        self.preprocessor_drawings = preprocessor_drawings
        self.preprocessor_thumb_path = None

        self.canny_detector = None
        self.depth_estimator = None
        self.openpose_detector = None
        self.lineart_generator = None
        self.pidinet_generator = None

    def run(self):
        source_image = None
        preprocessor_name = ""

        source_resolution = (
            int(self.adapter.generation_width * self.adapter.preprocessor_resolution),
            int(self.adapter.generation_height * self.adapter.preprocessor_resolution),
        )

        if self.source_changed:
            if self.adapter.source_image.image_filename:
                os.remove(self.adapter.source_image.image_filename)

                if self.adapter.source_image.image_thumb and os.path.isfile(self.adapter.source_image.image_thumb):
                    os.remove(self.adapter.source_image.image_thumb)
                    self.adapter.source_image.image_thumb = None

            original_pil_image = Image.open(self.adapter.source_image.image_original)
            width, height = original_pil_image.size
            if original_pil_image.mode not in ("RGBA", "LA", "P"):
                original_pil_image = original_pil_image.convert("RGBA")

            center = (width / 2, height / 2)
            original_pil_image = original_pil_image.rotate(-self.adapter.source_image.image_rotation, Image.Resampling.BICUBIC, center=center, expand=True)

            new_width = round(width * self.adapter.source_image.image_scale)
            new_height = round(height * self.adapter.source_image.image_scale)
            original_pil_image = original_pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            left = math.floor(new_width / 2 - (width / 2 + self.adapter.source_image.image_x_pos))
            top = math.floor(new_height / 2 - (height / 2 + self.adapter.source_image.image_y_pos))
            right = self.adapter.generation_width + left
            bottom = self.adapter.generation_height + top
            original_pil_image = original_pil_image.crop((left, top, right, bottom))

            drawing_image = self.drawings_pixmap.toImage()
            drawing_pil_image = ImageQt.fromqimage(drawing_image)
            merged_image = Image.alpha_composite(original_pil_image, drawing_pil_image)

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            source_filename = f"t2i_source_{timestamp}.png"
            source_path = os.path.join("tmp/", source_filename)
            merged_image.save(source_path)

            self.adapter.source_image.image_filename = source_path
            source_image = merged_image
            source_image = source_image.convert("RGB")
            numpy_image = np.array(source_image)
        else:
            if self.adapter.source_image.image_filename:
                numpy_image = cv2.imread(self.adapter.source_image.image_filename)  # pylint: disable=no-member
                numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGRA2RGB)  # pylint: disable=no-member

        if self.adapter.source_image.image_filename and self.adapter.source_image.image_thumb is None:
            thumb_filename = os.path.basename(self.adapter.source_image.image_filename)
            thumb_name, thumb_extension = os.path.splitext(thumb_filename)
            thumb_final_filename = f"{thumb_name}_thumb{thumb_extension}"
            thumb_path = os.path.join("tmp/", thumb_final_filename)
            source_numpy_thumb = self.generate_thumbnail(numpy_image)
            cv2.imwrite(thumb_path, source_numpy_thumb)  # pylint: disable=no-member
            self.adapter.source_image.image_thumb = thumb_path

        preprocessor_image = None
        preprocessor_loaded = False

        # if there's not preprocessor path, we must create it
        if self.preprocess:
            if self.adapter.type_index == 0:
                preprocessor_name = "canny"
                if self.canny_detector is None:
                    self.depth_estimator = None
                    self.openpose_detector = None
                    self.lineart_generator = None
                    self.pidinet_generator = None
                    self.canny_detector = CannyEdgesDetector()

                preprocessor_image = self.canny_detector.get_canny_edges(
                    numpy_image, self.adapter.canny_low, self.adapter.canny_high, resolution=source_resolution
                )
            elif self.adapter.type_index == 1:
                preprocessor_name = "depth"
                try:
                    if self.depth_estimator is None:
                        self.canny_detector = None
                        self.openpose_detector = None
                        self.depth_estimator = DepthEstimator(self.adapter.depth_type)

                    self.depth_estimator.change_model(self.adapter.depth_type)
                except OSError:
                    self.error.emit("You need to download the preprocessors from the downloader menu first.")
                    return

                preprocessor_image = self.depth_estimator.get_depth_map(numpy_image, source_resolution)
            elif self.adapter.type_index == 2:
                preprocessor_name = "pose"
                try:
                    if self.openpose_detector is None:
                        self.canny_detector = None
                        self.depth_estimator = None
                        self.lineart_generator = None
                        self.pidinet_generator = None
                        self.openpose_detector = OpenPoseDetector()
                except FileNotFoundError:
                    self.error.emit("You need to download the preprocessors from the downloader menu first.")
                    return

                preprocessor_image = self.openpose_detector.get_open_pose(numpy_image, source_resolution)
            elif self.adapter.type_index == 3:
                try:
                    if self.lineart_generator is None:
                        self.canny_detector = None
                        self.depth_estimator = None
                        self.openpose_detector = None
                        self.pidinet_generator = None
                        self.lineart_generator = LineArtGenerator(model_type=self.adapter.lineart_type)

                    self.lineart_generator.change_model(self.adapter.lineart_type)
                except FileNotFoundError:
                    self.error.emit("You need to download the preprocessors from the downloader menu first.")
                    return

                preprocessor_image = self.lineart_generator.get_lines(numpy_image, source_resolution)
            elif self.adapter.type_index == 4:
                try:
                    if self.pidinet_generator is None:
                        self.canny_detector = None
                        self.depth_estimator = None
                        self.openpose_detector = None
                        self.lineart_generator = None
                        self.pidinet_generator = PidinetGenerator(self.adapter.sketch_type)
                except FileNotFoundError:
                    self.error.emit("You need to download the preprocessors from the downloader menu first.")
                    return

                self.pidinet_generator.change_model(self.adapter.sketch_type)
                preprocessor_image = self.pidinet_generator.get_edges(numpy_image, source_resolution)

            if preprocessor_image is not None:
                self.preprocessor_pixmap = self.convert_numpy_to_pixmap(preprocessor_image)
        else:
            # The preprocessor image was dropped or loaded, use it as is
            self.preprocessor_pixmap = QPixmap(self.adapter.preprocessor_image.image_filename)

            if self.adapter.preprocessor_image.image_thumb is None:
                preprocessor_thumb_filename = f"t2i_{preprocessor_name}_{timestamp}_thumb.png"
                self.adapter.preprocessor_image.image_thumb = os.path.join("tmp/", preprocessor_thumb_filename)
                thumbnail_pixmap = self.preprocessor_pixmap.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                thumbnail_pixmap.save(self.adapter.preprocessor_image.image_thumb)

            preprocessor_loaded = True

        if self.preprocessor_pixmap and not preprocessor_loaded:
            if self.save_preprocessor:
                if self.adapter.preprocessor_image.image_filename:
                    os.remove(self.adapter.preprocessor_image.image_filename)

                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                preprocessor_filename = f"t2i_{preprocessor_name}_{timestamp}.png"
                self.adapter.preprocessor_image.image_filename = os.path.join("tmp/", preprocessor_filename)

                if preprocessor_image is not None:
                    preprocessor_drawing_image = self.preprocessor_drawings.toImage()
                    preprocessor_drawing_pip_image = ImageQt.fromqimage(preprocessor_drawing_image)
                    preprocessor_pil_image = Image.fromarray(preprocessor_image)
                    if preprocessor_pil_image.mode not in ("RGBA", "LA", "P"):
                        preprocessor_pil_image = preprocessor_pil_image.convert("RGBA")
                    merged_preprocessor = Image.alpha_composite(preprocessor_pil_image, preprocessor_drawing_pip_image)
                    merged_preprocessor.save(self.adapter.preprocessor_image.image_filename)

                if self.adapter.preprocessor_image.image_thumb:
                    os.remove(self.adapter.preprocessor_image.image_thumb)

                preprocessor_thumb_filename = f"t2i_{preprocessor_name}_{timestamp}_thumb.png"
                self.adapter.preprocessor_image.image_thumb = os.path.join("tmp/", preprocessor_thumb_filename)
                merged_preprocessor.thumbnail((80, 80))
                merged_preprocessor.save(self.adapter.preprocessor_image.image_thumb)

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
