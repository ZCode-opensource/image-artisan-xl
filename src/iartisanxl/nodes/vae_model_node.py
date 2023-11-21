import accelerate
from diffusers import AutoencoderKL

from iartisanxl.nodes.node import Node


class VaeModelNode(Node):
    OUTPUTS = ["vae", "vae_scale_factor"]

    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self.can_offload = True
        self.path = path

    def __call__(self) -> AutoencoderKL:
        device = "cpu" if self.sequential_offload else self.device

        vae = AutoencoderKL.from_pretrained(self.path, torch_dtype=self.torch_dtype).to(
            device
        )
        if self.sequential_offload:
            vae = accelerate.cpu_offload(vae, "cuda:0")

        self.values["vae"] = vae
        self.values["vae_scale_factor"] = 2 ** (len(vae.config.block_out_channels) - 1)

        return self.values
