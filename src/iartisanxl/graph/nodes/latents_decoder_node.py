import gc

import numpy as np
import torch
from PIL import Image

from iartisanxl.graph.nodes.node import Node


class LatentsDecoderNode(Node):
    REQUIRED_INPUTS = ["vae", "latents"]
    OUTPUTS = ["image"]

    def __call__(self):
        image = None

        needs_upcasting = self.vae.config.force_upcast and self.vae.dtype == torch.float16

        latents = self.latents

        if self.cpu_offload:
            self.vae.to("cuda:0")

        if needs_upcasting:
            self.vae.to(dtype=torch.float32)
            latents = latents.to(dtype=torch.float32)

        has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
        has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None

        if has_latents_mean and has_latents_std:
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            )
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
        else:
            latents = latents / self.vae.config.scaling_factor

        decoded = self.vae.decode(latents, return_dict=False)[0]

        if needs_upcasting:
            self.vae.to(dtype=self.torch_dtype)

        if self.cpu_offload:
            self.vae.to("cpu")

        image = decoded[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(1, 2, 0).float().numpy()
        self.values["image"] = Image.fromarray(np.uint8(image * 255))

        del image
        gc.collect()
        torch.cuda.empty_cache()

        return self.values
