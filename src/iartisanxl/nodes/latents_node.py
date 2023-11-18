# pylint: disable=no-member

import torch
from diffusers.utils.torch_utils import randn_tensor

from iartisanxl.nodes.node import Node


class LatentsNode(Node):
    REQUIRED_ARGS = [
        "width",
        "height",
        "seed",
        "num_channels_latents",
        "scale_factor",
        "device",
        "dtype",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)

    def __call__(self):
        shape = (
            1,
            self.num_channels_latents,
            self.height // self.scale_factor,
            self.width // self.scale_factor,
        )

        latents = randn_tensor(
            shape, generator=self.generator, device=self.device, dtype=self.dtype
        )

        return latents, self.generator
