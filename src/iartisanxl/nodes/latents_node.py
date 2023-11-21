# pylint: disable=no-member
import torch
from diffusers.utils.torch_utils import randn_tensor

from iartisanxl.nodes.node import Node


class LatentsNode(Node):
    PRIORITY = 4
    REQUIRED_ARGS = []
    INPUTS = ["vae_scale_factor", "width", "height", "num_channels_latents", "seed"]
    OUTPUTS = ["latents", "generator"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.torch_dtype = None
        self.device = None

    def __call__(self, vae_scale_factor, width, height, num_channels_latents, seed):
        generator = torch.Generator(device="cpu").manual_seed(seed)

        shape = (
            1,
            num_channels_latents,
            height // vae_scale_factor,
            width // vae_scale_factor,
        )

        latents = randn_tensor(
            shape, generator=generator, device=self.device, dtype=self.torch_dtype
        )

        return latents, generator
