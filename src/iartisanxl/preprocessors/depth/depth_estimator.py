import os

import cv2
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation


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

    # pylint: disable=no-member
    def get_depth_map(self, image, resolution=None):
        original_resolution = (image.shape[0], image.shape[1])
        if resolution:
            image = cv2.resize(image, (resolution[1], resolution[0]))
        image = self.image_processor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = self.depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=original_resolution,
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        depth_map = (depth_map * 255.0).clip(0, 255).to(torch.uint8)
        image = torch.cat([depth_map] * 3, dim=1)

        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]

        return image
