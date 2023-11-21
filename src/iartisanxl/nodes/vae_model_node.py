# pylint: disable=no-member
import accelerate
from diffusers import AutoencoderKL

from iartisanxl.nodes.node import Node


class VaeModelNode(Node):
    PRIORITY = 3
    REQUIRED_ARGS = [
        "path",
    ]
    OUTPUTS = ["vae", "vae_scale_factor"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.can_offload = True

    def __call__(self) -> AutoencoderKL:
        device = "cpu" if self.sequential_offload else self.device

        vae = AutoencoderKL.from_pretrained(self.path, torch_dtype=self.torch_dtype).to(
            device
        )
        if self.sequential_offload:
            vae = accelerate.cpu_offload(vae, "cuda:0")

        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        return vae, vae_scale_factor
