from iartisanxl.graph.nodes.node import Node


class ControlnetNode(Node):
    REQUIRED_INPUTS = ["controlnet_model", "image"]
    OUTPUTS = ["controlnet"]

    def __init__(
        self,
        conditioning_scale: float = None,
        guidance_start: float = None,
        guidance_end: float = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conditioning_scale = conditioning_scale
        self.guidance_start = guidance_start
        self.guidance_end = guidance_end

    def update_controlnet(self, conditioning_scale: float, guidance_start: float, guidance_end: float, enabled: bool):
        self.conditioning_scale = conditioning_scale
        self.guidance_start = guidance_start
        self.guidance_end = guidance_end
        self.enabled = enabled
        self.set_updated()

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["conditioning_scale"] = self.conditioning_scale
        node_dict["guidance_start"] = self.guidance_start
        node_dict["guidance_end"] = self.guidance_end
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(ControlnetNode, cls).from_dict(node_dict)
        node.conditioning_scale = node_dict["conditioning_scale"]
        node.guidance_start = node_dict["guidance_start"]
        node.guidance_end = node_dict["guidance_end"]
        return node

    def update_inputs(self, node_dict):
        self.conditioning_scale = node_dict["conditioning_scale"]
        self.guidance_start = node_dict["guidance_start"]
        self.guidance_end = node_dict["guidance_end"]

    def __call__(self):
        super().__call__()

        if not self.enabled:
            self.conditioning_scale = 0

        self.values["controlnet"] = {
            "model": self.controlnet_model,
            "image": self.image,
            "conditioning_scale": self.conditioning_scale,
            "guidance_start": self.guidance_start,
            "guidance_end": self.guidance_end,
        }

        return self.values
