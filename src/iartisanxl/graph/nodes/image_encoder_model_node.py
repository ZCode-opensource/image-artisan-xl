import accelerate
import torch
from transformers import CLIPVisionModelWithProjection

from iartisanxl.graph.nodes.node import Node


class ImageEncoderModelNode(Node):
    OUTPUTS = ["image_encoder"]

    def __init__(self, path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    def update_model(self, path: str):
        self.values["image_encoder"] = None
        torch.cuda.empty_cache()

        self.path = path
        self.set_updated()

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["path"] = self.path
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(ImageEncoderModelNode, cls).from_dict(node_dict)
        node.path = node_dict["path"]
        return node

    def update_inputs(self, node_dict):
        self.path = node_dict["path"]

    def __call__(self) -> CLIPVisionModelWithProjection:
        device = "cpu" if self.sequential_offload or self.cpu_offload else self.device

        image_encoder_model = CLIPVisionModelWithProjection.from_pretrained(
            self.path,
            torch_dtype=self.torch_dtype,
            use_safetensors=True,
        ).to(device)

        if self.sequential_offload:
            image_encoder_model = accelerate.cpu_offload(image_encoder_model, "cuda:0")

        self.values["image_encoder"] = image_encoder_model

        return self.values
