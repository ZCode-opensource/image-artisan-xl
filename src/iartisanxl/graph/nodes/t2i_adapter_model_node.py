import accelerate
import torch
from diffusers import T2IAdapter

from iartisanxl.graph.nodes.node import Node


class T2IAdapterModelNode(Node):
    OUTPUTS = ["t2i_adapter_model"]

    def __init__(self, path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    def update_model(self, path: str):
        self.values["t2i_adapter_model"] = None
        torch.cuda.empty_cache()

        self.path = path
        self.set_updated()

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["path"] = self.path
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(T2IAdapterModelNode, cls).from_dict(node_dict)
        node.path = node_dict["path"]
        return node

    def update_inputs(self, node_dict):
        self.path = node_dict["path"]

    def __call__(self) -> T2IAdapter:
        device = "cpu" if self.sequential_offload or self.cpu_offload else self.device

        t2i_adapter_model = T2IAdapter.from_pretrained(
            self.path,
            torch_dtype=self.torch_dtype,
            use_safetensors=True,
            variant="fp16",
        ).to(device)

        if self.sequential_offload:
            t2i_adapter_model = accelerate.cpu_offload(t2i_adapter_model, "cuda:0")

        self.values["t2i_adapter_model"] = t2i_adapter_model

        return self.values
