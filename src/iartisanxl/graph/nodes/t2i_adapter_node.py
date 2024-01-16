from PIL import Image

from iartisanxl.graph.nodes.node import Node


class T2IAdapterNode(Node):
    REQUIRED_INPUTS = ["t2i_adapter_model", "image"]
    OUTPUTS = ["t2i_adapter"]

    def __init__(self, type_index: int, adapter_type: str, conditioning_scale: float, conditioning_factor: float, **kwargs):
        super().__init__(**kwargs)
        self.type_index = type_index
        self.adapter_type = adapter_type
        self.conditioning_scale = conditioning_scale
        self.conditioning_factor = conditioning_factor

    def update_adapter(self, type_index: int, adapter_type: str, enabled: bool, conditioning_scale: float, conditioning_factor: float):
        self.type_index = type_index
        self.adapter_type = adapter_type
        self.enabled = enabled
        self.conditioning_scale = conditioning_scale
        self.conditioning_factor = conditioning_factor
        self.set_updated()

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["type_index"] = self.type_index
        node_dict["adapter_type"] = self.adapter_type
        node_dict["conditioning_scale"] = self.conditioning_scale
        node_dict["conditioning_factor"] = self.conditioning_factor
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(T2IAdapterNode, cls).from_dict(node_dict)
        node.type_index = node_dict["type_index"]
        node.adapter_type = node_dict["adapter_type"]
        node.conditioning_scale = node_dict["conditioning_scale"]
        node.conditioning_factor = node_dict["conditioning_factor"]
        return node

    def update_inputs(self, node_dict):
        self.type_index = node_dict["type_index"]
        self.adapter_type = node_dict["adapter_type"]
        self.conditioning_scale = node_dict["conditioning_scale"]
        self.conditioning_factor = node_dict["conditioning_factor"]

    def __call__(self):
        if not self.enabled:
            self.conditioning_scale = 0

        image = self.image

        if isinstance(image, Image.Image):
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGB")

        self.values["t2i_adapter"] = {
            "model": self.t2i_adapter_model,
            "image": image,
            "conditioning_scale": self.conditioning_scale,
            "conditioning_factor": self.conditioning_factor,
        }

        return self.values
