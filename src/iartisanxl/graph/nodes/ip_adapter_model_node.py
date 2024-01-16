import torch
from safetensors import safe_open

from iartisanxl.graph.nodes.node import Node


class IPAdapterModelNode(Node):
    OUTPUTS = ["ip_adapter_model"]

    def __init__(self, path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.ip_adapter_model = None

    def update_model(self, path: str):
        self.ip_adapter_model = None
        torch.cuda.empty_cache()

        self.path = path
        self.set_updated()

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["path"] = self.path
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(IPAdapterModelNode, cls).from_dict(node_dict)
        node.path = node_dict["path"]
        return node

    def update_inputs(self, node_dict):
        self.path = node_dict["path"]

    def __call__(self) -> dict:
        state_dict = {"image_proj": {}, "ip_adapter": {}}
        with safe_open(self.path, framework="pt", device="cpu") as model_file:
            for key in model_file.keys():
                if key.startswith("image_proj."):
                    state_dict["image_proj"][key.replace("image_proj.", "")] = model_file.get_tensor(key)
                elif key.startswith("ip_adapter."):
                    state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = model_file.get_tensor(key)

        self.values["ip_adapter_model"] = state_dict

        return self.values
