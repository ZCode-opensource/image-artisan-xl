import os

import numpy as np
import cv2
import torch

from iartisanxl.preprocessors.lineart.model import Generator


class LineArtGenerator:
    def __init__(self, model_type: str = "anime_style"):
        self.model_type = model_type
        self.model = None
        self.load_model()

    def load_model(self):
        self.model = 0
        self.model = Generator(3, 1, 3)
        self.model.cuda()

        self.model.load_state_dict(torch.load(os.path.join("./models/preprocessors/lineart", self.model_type, "netG_A_latest.pth")))
        self.model.eval()

    def change_model(self, model_type):
        if model_type != self.model_type:
            self.model_type = model_type
            self.load_model()

    # pylint: disable=no-member
    def get_lines(self, image, resolution=None):
        original_resolution = (image.shape[0], image.shape[1])

        if resolution:
            image = cv2.resize(image, (resolution[1], resolution[0]))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))
        image = image / 255.0
        tensor_image = torch.from_numpy(image).unsqueeze(0).float().to("cuda")

        with torch.no_grad():
            image = self.model(tensor_image)[0][0]
            image = image.cpu().numpy()
            image = (image * 255.0).clip(0, 255).astype(np.uint8)

        image = cv2.resize(image, (original_resolution[1], original_resolution[0]))
        inverted_image = 255 - image

        inverted_image = np.stack([inverted_image] * 3, axis=-1)

        return inverted_image
