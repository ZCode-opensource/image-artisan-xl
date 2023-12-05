import os

import numpy as np
import cv2
import torch

from iartisanxl.annotators.pidinet.pidinet import pidinet


class Args:
    def __init__(self, config, dil, sa):
        self.config = config
        self.dil = dil
        self.sa = sa


class PidinetGenerator:
    def __init__(self, model_type: str = "table5"):
        self.model_type = model_type
        self.model = None
        self.load_model()

    def load_model(self):
        args = Args("carv4", True, True)
        self.model = pidinet(args)

        checkpoint_filename = f"{self.model_type}_pidinet.pth"
        checkpoint = torch.load(os.path.join("./models/annotators/pidinet/", checkpoint_filename), map_location="cpu")
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}

        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v

        self.model.load_state_dict(new_state_dict)
        self.model.to("cuda")
        self.model.eval()

    def change_model(self, model_type):
        if model_type != self.model_type:
            self.model_type = model_type
            self.load_model()

    # pylint: disable=no-member
    def get_edges(self, image, resolution=None):
        original_resolution = (image.shape[0], image.shape[1])

        if resolution:
            image = cv2.resize(image, (resolution[1], resolution[0]))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))
        image = image / 255.0
        tensor_image = torch.from_numpy(image).unsqueeze(0).float().to("cuda")

        with torch.no_grad():
            image = self.model(tensor_image)[-1]
            image = image.cpu().numpy()
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
            image = image[0, 0]

        image = cv2.resize(image, (original_resolution[1], original_resolution[0]))

        image = np.stack([image] * 3, axis=-1)

        return image
