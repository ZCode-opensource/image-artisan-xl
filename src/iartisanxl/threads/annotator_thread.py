import os
import math
from datetime import datetime

import cv2
from PIL import Image, ImageQt
import numpy as np
from PyQt6.QtCore import pyqtSignal, QThread, Qt
from PyQt6.QtGui import QImage, QPixmap

from iartisanxl.annotators.canny.canny_edges_detector import CannyEdgesDetector
from iartisanxl.annotators.depth.depth_estimator import DepthEstimator
from iartisanxl.annotators.openpose.open_pose_detector import OpenPoseDetector
from iartisanxl.modules.common.controlnet.controlnet_data_object import ControlNetDataObject


class AnnotatorThread(QThread):
    error = pyqtSignal(str)

    def __init__(
        self,
        controlnet: ControlNetDataObject,
        drawings_pixmap: QPixmap,
        source_changed: bool,
        annotate: bool,
        save_annotator: bool = False,
        annotator_drawings: QPixmap = None,
    ):
        super().__init__()

        self.source_thumb_pixmap = None
        self.annotator_pixmap = None
        self.controlnet = controlnet
        self.drawings_pixmap = drawings_pixmap
        self.source_changed = source_changed
        self.annotate = annotate
        self.save_annotator = save_annotator
        self.annotator_drawings = annotator_drawings
        self.annotator_thumb_path = None

        self.canny_detector = None
        self.depth_estimator = None
        self.openpose_detector = None

    def run(self):
        source_image = None
        annotator_name = ""

        source_resolution = (
            int(self.controlnet.generation_width * self.controlnet.annotator_resolution),
            int(self.controlnet.generation_height * self.controlnet.annotator_resolution),
        )

        if self.source_changed:
            if self.controlnet.source_image.image_filename:
                os.remove(self.controlnet.source_image.image_filename)

                if self.controlnet.source_image.image_thumb and os.path.isfile(self.controlnet.source_image.image_thumb):
                    os.remove(self.controlnet.source_image.image_thumb)
                    self.controlnet.source_image.image_thumb = None

            original_pil_image = Image.open(self.controlnet.source_image.image_original)
            width, height = original_pil_image.size
            if original_pil_image.mode not in ("RGBA", "LA", "P"):
                original_pil_image = original_pil_image.convert("RGBA")

            center = (width / 2, height / 2)
            original_pil_image = original_pil_image.rotate(
                -self.controlnet.source_image.image_rotation, Image.Resampling.BICUBIC, center=center, expand=True
            )

            new_width = round(width * self.controlnet.source_image.image_scale)
            new_height = round(height * self.controlnet.source_image.image_scale)
            original_pil_image = original_pil_image.resize((new_width, new_height), Image.LANCZOS)

            left = math.floor(new_width / 2 - (width / 2 + self.controlnet.source_image.image_x_pos))
            top = math.floor(new_height / 2 - (height / 2 + self.controlnet.source_image.image_y_pos))
            right = self.controlnet.generation_width + left
            bottom = self.controlnet.generation_height + top
            original_pil_image = original_pil_image.crop((left, top, right, bottom))

            drawing_image = self.drawings_pixmap.toImage()
            drawing_pil_image = ImageQt.fromqimage(drawing_image)
            merged_image = Image.alpha_composite(original_pil_image, drawing_pil_image)

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            source_filename = f"cn_source_{timestamp}.png"
            source_path = os.path.join("tmp/", source_filename)
            merged_image.save(source_path)

            self.controlnet.source_image.image_filename = source_path
            source_image = merged_image
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

        annotator_image = None
        annotator_loaded = False

        # if there's not annotator path, we must create it
        if self.annotate:
            if self.controlnet.type_index == 0:
                annotator_name = "canny"
                if self.canny_detector is None:
                    self.depth_estimator = None
                    self.openpose_detector = None
                    self.canny_detector = CannyEdgesDetector()

                annotator_image = self.canny_detector.get_canny_edges(
                    numpy_image, self.controlnet.canny_low, self.controlnet.canny_high, resolution=source_resolution
                )
            elif self.controlnet.type_index == 1:
                annotator_name = "depth"
                try:
                    if self.depth_estimator is None:
                        self.canny_detector = None
                        self.openpose_detector = None
                        self.depth_estimator = DepthEstimator(self.controlnet.depth_type)

                    self.depth_estimator.change_model(self.controlnet.depth_type)
                except OSError:
                    self.error.emit("You need to download the annotators from the downloader menu first.")
                    return

                annotator_image = self.depth_estimator.get_depth_map(numpy_image, source_resolution)
            elif self.controlnet.type_index == 2:
                annotator_name = "pose"
                try:
                    if self.openpose_detector is None:
                        self.canny_detector = None
                        self.depth_estimator = None
                        self.openpose_detector = OpenPoseDetector()
                except FileNotFoundError:
                    self.error.emit("You need to download the annotators from the downloader menu first.")
                    return

                annotator_image = self.openpose_detector.get_open_pose(numpy_image, source_resolution)

            if annotator_image is not None:
                self.annotator_pixmap = self.convert_numpy_to_pixmap(annotator_image)
        else:
            # The annotator image was dropped or loaded, use it as is
            self.annotator_pixmap = QPixmap(self.controlnet.annotator_image.image_filename)

            if self.controlnet.annotator_image.image_thumb is None:
                annotator_thumb_filename = f"cn_{annotator_name}_{timestamp}_thumb.png"
                self.controlnet.annotator_image.image_thumb = os.path.join("tmp/", annotator_thumb_filename)
                thumbnail_pixmap = self.annotator_pixmap.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                thumbnail_pixmap.save(self.controlnet.annotator_image.image_thumb)

            annotator_loaded = True

        if self.annotator_pixmap and not annotator_loaded:
            if self.save_annotator:
                if self.controlnet.annotator_image.image_filename:
                    os.remove(self.controlnet.annotator_image.image_filename)

                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                annotator_filename = f"cn_{annotator_name}_{timestamp}.png"
                self.controlnet.annotator_image.image_filename = os.path.join("tmp/", annotator_filename)

                if annotator_image is not None:
                    annotator_drawing_image = self.annotator_drawings.toImage()
                    annotator_drawing_pip_image = ImageQt.fromqimage(annotator_drawing_image)
                    annotator_pil_image = Image.fromarray(annotator_image)
                    if annotator_pil_image.mode not in ("RGBA", "LA", "P"):
                        annotator_pil_image = annotator_pil_image.convert("RGBA")
                    merged_annotator = Image.alpha_composite(annotator_pil_image, annotator_drawing_pip_image)
                    merged_annotator.save(self.controlnet.annotator_image.image_filename)

                if self.controlnet.annotator_image.image_thumb:
                    os.remove(self.controlnet.annotator_image.image_thumb)

                annotator_thumb_filename = f"cn_{annotator_name}_{timestamp}_thumb.png"
                self.controlnet.annotator_image.image_thumb = os.path.join("tmp/", annotator_thumb_filename)
                merged_annotator.thumbnail((80, 80))
                merged_annotator.save(self.controlnet.annotator_image.image_thumb)

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
