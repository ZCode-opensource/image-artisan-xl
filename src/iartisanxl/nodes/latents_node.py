import torch
from diffusers.utils.torch_utils import randn_tensor

from iartisanxl.nodes.node import Node


class LatentsNode(Node):
    REQUIRED_INPUTS = [
        "vae_scale_factor",
        "width",
        "height",
        "num_channels_latents",
        "seed",
    ]
    OUTPUTS = ["latents", "generator"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.torch_dtype = None
        self.device = None

    def __call__(self):
        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        shape = (
            1,
            self.num_channels_latents,
            self.height // self.vae_scale_factor,
            self.width // self.vae_scale_factor,
        )

        latents = randn_tensor(
            shape, generator=generator, device=self.device, dtype=self.torch_dtype
        )

        self.values["latents"] = latents
        self.values["generator"] = generator

        return self.values
