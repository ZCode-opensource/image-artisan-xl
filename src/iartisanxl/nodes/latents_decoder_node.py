# pylint: disable=no-member
import torch
import numpy as np
from PIL import Image

from iartisanxl.nodes.node import Node


class LatentsDecoderNode(Node):
    REQUIRED_ARGS = [
        "vae",
        "device",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.torch_dtype = kwargs.get("torch_dtype", torch.float16)
        self.vae = kwargs.get("vae", None)

    def __call__(self, latents) -> Image:
        image = None

        if self.vae is not None:
            needs_upcasting = (
                self.vae.config.force_upcast and self.vae.dtype == torch.float16
            )

            if needs_upcasting:
                self.vae.to(dtype=torch.float32)
                latents = latents.to(dtype=torch.float32)

            if self.vae.device != latents.device and str(self.vae.device) != "meta":
                self.vae.to(latents.device)

            decoded = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]

            if needs_upcasting:
                self.vae.to(dtype=self.torch_dtype)

            image = decoded[0]
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(1, 2, 0).float().numpy()
            image = Image.fromarray(np.uint8(image * 255))

        return image
