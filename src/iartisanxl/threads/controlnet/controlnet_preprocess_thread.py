import os
from datetime import datetime

import cv2
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

from iartisanxl.modules.common.controlnet.controlnet_data import ControlNetData
from iartisanxl.modules.common.image.image_editor_layer import ImageEditorLayer
from iartisanxl.preprocessors.canny.canny_edges_detector import CannyEdgesDetector
from iartisanxl.preprocessors.depth.depth_estimator import DepthEstimator
from iartisanxl.utilities.image.converters import convert_numpy_to_pixmap, convert_pillow_to_pixmap


preprocessors = ["canny", "depth"]


class ControlnetPreprocessThread(QThread):
    error = pyqtSignal(str)
    preprocessor_finished = pyqtSignal(object, str)

    def __init__(
        self,
        controlnet_data: ControlNetData,
        layer: ImageEditorLayer,
        prefix: str = "img",
    ):
        super().__init__()

        self.controlnet_data = controlnet_data
        self.target_width = self.controlnet_data.generation_width
        self.target_height = self.controlnet_data.generation_height
        self.resolution = self.controlnet_data.preprocessor_resolution
        self.layer = layer
        self.prefix = prefix

        self.canny_detector = None
        self.depth_estimator = None

    def run(self):
        if self.layer.image_path is not None and os.path.isfile(self.layer.image_path):
            os.remove(self.layer.image_path)
        if self.layer.original_path is not None and os.path.isfile(self.layer.original_path):
            os.remove(self.layer.original_path)

        preprocessor_resolution = (int(self.target_width * self.resolution), int(self.target_height * self.resolution))
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        preprocessor_filename = f"{self.prefix}_{timestamp}_{self.layer.layer_id}_original.png"
        preprocessor_path = os.path.join("tmp/", preprocessor_filename)

        if self.controlnet_data.type_index == 0:
            numpy_image = cv2.imread(self.controlnet_data.source_image)
            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGRA2RGB)

            if self.canny_detector is None:
                self.depth_estimator = None
                self.canny_detector = CannyEdgesDetector()

            preprocessor_image = self.canny_detector.get_canny_edges(
                numpy_image,
                self.controlnet_data.canny_low,
                self.controlnet_data.canny_high,
                resolution=preprocessor_resolution,
            )
            cv2.imwrite(preprocessor_path, preprocessor_image)
            pixmap = convert_numpy_to_pixmap(preprocessor_image)
        elif self.controlnet_data.type_index == 1:
            pil_image = Image.open(self.controlnet_data.source_image).convert("RGB")

            try:
                if self.depth_estimator is None:
                    self.canny_detector = None
                    self.depth_estimator = DepthEstimator(self.controlnet_data.depth_type)

                self.depth_estimator.change_model(self.controlnet_data.depth_type)
            except OSError:
                self.error.emit("You need to download the preprocessors from the downloader menu first.")
                return

            preprocessor_image = self.depth_estimator.get_depth_map(pil_image, preprocessor_resolution)
            preprocessor_image.save(preprocessor_path)
            pixmap = convert_pillow_to_pixmap(preprocessor_image)

        self.preprocessor_finished.emit(pixmap, preprocessor_path)
