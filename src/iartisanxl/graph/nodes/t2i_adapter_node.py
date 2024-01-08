from iartisanxl.graph.nodes.node import Node


class T2IAdapterNode(Node):
    REQUIRED_INPUTS = ["t2i_adapter_model", "image"]
    OUTPUTS = ["t2i_adapter"]

    def __init__(
        self,
        conditioning_scale: float = None,
        conditioning_factor: float = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conditioning_scale = conditioning_scale
        self.conditioning_factor = conditioning_factor

    def update_adapter(self, conditioning_scale: float, conditioning_factor: float, enabled: bool):
        self.conditioning_scale = conditioning_scale
        self.conditioning_factor = conditioning_factor
        self.enabled = enabled
        self.set_updated()

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["conditioning_scale"] = self.conditioning_scale
        node_dict["conditioning_factor"] = self.conditioning_factor
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(T2IAdapterNode, cls).from_dict(node_dict)
        node.conditioning_scale = node_dict["conditioning_scale"]
        node.conditioning_factor = node_dict["conditioning_factor"]
        return node

    def update_inputs(self, node_dict):
        self.conditioning_scale = node_dict["conditioning_scale"]
        self.conditioning_factor = node_dict["conditioning_factor"]

    def __call__(self):
        if not self.enabled:
            self.conditioning_scale = 0

        self.values["t2i_adapter"] = {
            "model": self.t2i_adapter_model,
            "image": self.image,
            "conditioning_scale": self.conditioning_scale,
            "conditioning_factor": self.conditioning_factor,
        }

        return self.values
