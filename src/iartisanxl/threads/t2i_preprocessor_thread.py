import os
from datetime import datetime

import cv2
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

from iartisanxl.modules.common.image.image_editor_layer import ImageEditorLayer
from iartisanxl.modules.common.t2i_adapter.t2i_adapter_data_object import T2IAdapterDataObject
from iartisanxl.preprocessors.canny.canny_edges_detector import CannyEdgesDetector
from iartisanxl.preprocessors.depth.depth_estimator import DepthEstimator
from iartisanxl.preprocessors.lineart.lineart_generator import LineArtGenerator
from iartisanxl.preprocessors.pidinet.pidinet_generator import PidinetGenerator
from iartisanxl.utilities.image.converters import convert_numpy_to_pixmap, convert_pillow_to_pixmap


preprocessors = ["canny", "depth", "lineart", "pidinet"]


class T2IPreprocessorThread(QThread):
    error = pyqtSignal(str)
    preprocessor_finished = pyqtSignal(object, str)

    def __init__(
        self,
        t2i_adapter_data: T2IAdapterDataObject,
        layer: ImageEditorLayer,
        prefix: str = "img",
    ):
        super().__init__()

        self.t2i_adapter_data = t2i_adapter_data
        self.target_width = self.t2i_adapter_data.generation_width
        self.target_height = self.t2i_adapter_data.generation_height
        self.resolution = self.t2i_adapter_data.preprocessor_resolution
        self.layer = layer
        self.prefix = prefix

        self.canny_detector = None
        self.depth_estimator = None
        self.openpose_detector = None
        self.lineart_generator = None
        self.pidinet_generator = None

    def run(self):
        if self.layer.image_path is not None and os.path.isfile(self.layer.image_path):
            os.remove(self.layer.image_path)
        if self.layer.original_path is not None and os.path.isfile(self.layer.original_path):
            os.remove(self.layer.original_path)

        preprocessor_resolution = (int(self.target_width * self.resolution), int(self.target_height * self.resolution))
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        preprocessor_filename = f"{self.prefix}_{timestamp}_{self.layer.layer_id}_original.png"
        preprocessor_path = os.path.join("tmp/", preprocessor_filename)

        if self.t2i_adapter_data.type_index == 0:
            numpy_image = cv2.imread(self.t2i_adapter_data.source_image)
            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGRA2RGB)

            if self.canny_detector is None:
                self.canny_detector = CannyEdgesDetector()
                self.depth_estimator = None
                self.lineart_generator = None
                self.pidinet_generator = None

            preprocessor_image = self.canny_detector.get_canny_edges(
                numpy_image,
                self.t2i_adapter_data.canny_low,
                self.t2i_adapter_data.canny_high,
                resolution=preprocessor_resolution,
            )
            cv2.imwrite(preprocessor_path, preprocessor_image)
            pixmap = convert_numpy_to_pixmap(preprocessor_image)
        elif self.t2i_adapter_data.type_index == 1:
            pil_image = Image.open(self.t2i_adapter_data.source_image).convert("RGB")

            try:
                if self.depth_estimator is None:
                    self.canny_detector = None
                    self.depth_estimator = DepthEstimator(self.t2i_adapter_data.depth_type)
                    self.lineart_generator = None
                    self.pidinet_generator = None

                self.depth_estimator.change_model(self.t2i_adapter_data.depth_type)
            except OSError:
                self.error.emit("You need to download the preprocessors from the downloader menu first.")
                return

            preprocessor_image = self.depth_estimator.get_depth_map(pil_image, preprocessor_resolution)
            preprocessor_image.save(preprocessor_path)
            pixmap = convert_pillow_to_pixmap(preprocessor_image)
        elif self.t2i_adapter_data.type_index == 2:
            numpy_image = cv2.imread(self.t2i_adapter_data.source_image)
            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGRA2RGB)

            try:
                if self.lineart_generator is None:
                    self.canny_detector = None
                    self.depth_estimator = None
                    self.openpose_detector = None
                    self.lineart_generator = LineArtGenerator(model_type=self.t2i_adapter_data.lineart_type)
                    self.pidinet_generator = None

                self.lineart_generator.change_model(self.t2i_adapter_data.lineart_type)
            except FileNotFoundError:
                self.error.emit("You need to download the preprocessors from the downloader menu first.")
                return

            preprocessor_image = self.lineart_generator.get_lines(numpy_image, preprocessor_resolution)
            cv2.imwrite(preprocessor_path, preprocessor_image)
            pixmap = convert_numpy_to_pixmap(preprocessor_image)
        elif self.t2i_adapter_data.type_index == 3:
            numpy_image = cv2.imread(self.t2i_adapter_data.source_image)
            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGRA2RGB)

            try:
                if self.pidinet_generator is None:
                    self.canny_detector = None
                    self.depth_estimator = None
                    self.openpose_detector = None
                    self.pidinet_generator = PidinetGenerator(self.t2i_adapter_data.sketch_type)
                    self.lineart_generator = None
            except FileNotFoundError:
                self.error.emit("You need to download the preprocessors from the downloader menu first.")
                return

            self.pidinet_generator.change_model(self.t2i_adapter_data.sketch_type)
            preprocessor_image = self.pidinet_generator.get_edges(numpy_image, preprocessor_resolution)
            cv2.imwrite(preprocessor_path, preprocessor_image)
            pixmap = convert_numpy_to_pixmap(preprocessor_image)

        self.preprocessor_finished.emit(pixmap, preprocessor_path)
