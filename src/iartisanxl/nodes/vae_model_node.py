# pylint: disable=no-member
import torch
from diffusers import AutoencoderKL

from iartisanxl.nodes.node import Node


class VaeModelNode(Node):
    REQUIRED_ARGS = [
        "path",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.torch_dtype = kwargs.get("torch_dtype", torch.float16)

    def __call__(self) -> AutoencoderKL:
        vae = AutoencoderKL.from_pretrained(self.path, torch_dtype=self.torch_dtype)
        return vae
