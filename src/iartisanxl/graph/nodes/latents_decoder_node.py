import torch
import numpy as np
from PIL import Image

from iartisanxl.graph.nodes.node import Node


class LatentsDecoderNode(Node):
    REQUIRED_INPUTS = ["vae", "latents"]
    OUTPUTS = ["image"]

    def __call__(self):
        super().__call__()
        image = None

        needs_upcasting = (
            self.vae.config.force_upcast and self.vae.dtype == torch.float16
        )

        latents = self.latents

        if self.cpu_offload:
            self.vae.to("cuda:0")

        if needs_upcasting:
            self.vae.to(dtype=torch.float32)
            latents = latents.to(dtype=torch.float32)

        decoded = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]

        if needs_upcasting:
            self.vae.to(dtype=self.torch_dtype)

        if self.cpu_offload:
            self.vae.to(self.device)

        image = decoded[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(1, 2, 0).float().numpy()
        image = Image.fromarray(np.uint8(image * 255))

        torch.cuda.empty_cache()

        self.values["image"] = image

        return self.values
