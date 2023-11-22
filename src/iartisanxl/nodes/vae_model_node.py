import accelerate
from diffusers import AutoencoderKL

from iartisanxl.nodes.node import Node


class VaeModelNode(Node):
    OUTPUTS = ["vae", "vae_scale_factor"]

    def __init__(self, path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["path"] = self.path
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(VaeModelNode, cls).from_dict(node_dict)
        node.path = node_dict["path"]
        return node

    def __call__(self) -> AutoencoderKL:
        super().__call__()
        device = "cpu" if self.sequential_offload or self.cpu_offload else self.device

        vae = AutoencoderKL.from_pretrained(self.path, torch_dtype=self.torch_dtype).to(
            device
        )
        if self.sequential_offload:
            vae = accelerate.cpu_offload(vae, "cuda:0")

        self.values["vae"] = vae
        self.values["vae_scale_factor"] = 2 ** (len(vae.config.block_out_channels) - 1)

        return self.values
