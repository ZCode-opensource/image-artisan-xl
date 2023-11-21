# pylint: disable=no-member
import torch
import numpy as np
from PIL import Image

from iartisanxl.nodes.node import Node


class LatentsDecoderNode(Node):
    PRIORITY = 6
    REQUIRED_ARGS = []
    INPUTS = ["vae", "latents"]
    OUTPUTS = ["image"]

    def __call__(self, vae, latents) -> Image:
        image = None

        if vae is not None:
            needs_upcasting = vae.config.force_upcast and vae.dtype == torch.float16

            if needs_upcasting:
                vae.to(dtype=torch.float32)
                latents = latents.to(dtype=torch.float32)

            decoded = vae.decode(
                latents / vae.config.scaling_factor, return_dict=False
            )[0]

            if needs_upcasting:
                vae.to(dtype=self.torch_dtype)

            image = decoded[0]
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(1, 2, 0).float().numpy()
            image = Image.fromarray(np.uint8(image * 255))

        return image
