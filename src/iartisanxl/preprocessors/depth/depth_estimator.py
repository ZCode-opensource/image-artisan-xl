import os

import numpy as np
import torch
from PIL import Image
from transformers import DPTForDepthEstimation, DPTImageProcessor


class DepthEstimator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.load_model()

    def load_model(self):
        model_path = os.path.join("./models/preprocessors/depth", self.model_name)
        self.depth_estimator = DPTForDepthEstimation.from_pretrained(model_path).to("cuda")
        self.image_processor = DPTImageProcessor.from_pretrained(model_path)

    def change_model(self, new_model_name):
        if new_model_name != self.model_name:
            self.model_name = new_model_name
            self.load_model()

    def get_depth_map(self, image: Image.Image, resolution: float = None):
        original_size = image.size

        if resolution:
            image = image.resize(resolution, Image.Resampling.LANCZOS)

        inputs = self.image_processor(images=image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.depth_estimator(**inputs)
            predicted_depth = outputs.predicted_depth

        # get back the image to the original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1), size=original_size[::-1], mode="bicubic", align_corners=False
        )

        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depthmap = Image.fromarray(formatted)

        return depthmap
