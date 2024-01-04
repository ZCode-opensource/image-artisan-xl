from PIL import Image

from iartisanxl.graph.nodes.node import Node


class ImageLoadNode(Node):
    OUTPUTS = ["image"]

    def __init__(self, path: str = None, image: Image = None, weight: float = None, noise: float = None, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.image = image
        self.weight = weight
        self.noise = noise

    def update_value(self, path: str):
        self.path = path
        self.set_updated()

    def update_image(self, image: Image):
        self.image = image
        self.set_updated()

    def update_path(self, path: str):
        self.path = path
        Image.open(self.path)
        self.set_updated()

    def update_weight(self, weight: float):
        self.weight = weight
        self.set_updated()

    def update_noise(self, noise: float):
        self.noise = noise
        self.set_updated()

    def update_path_weight_noise(self, path: str, weight: float, noise: float):
        self.path = path
        Image.open(self.path)
        self.weight = weight
        self.noise = noise
        self.set_updated()

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["path"] = self.path
        node_dict["image"] = self.image
        node_dict["weight"] = self.weight
        node_dict["noise"] = self.noise
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(ImageLoadNode, cls).from_dict(node_dict)
        node.path = node_dict["path"]
        node.image = node_dict["image"]
        node.weight = node_dict["weight"]
        node.noise = node_dict["noise"]
        return node

    def update_inputs(self, node_dict):
        self.path = node_dict["path"]
        self.image = node_dict["image"]
        self.weight = node_dict["weight"]
        self.noise = node_dict["noise"]

    def __call__(self):
        super().__call__()

        if self.image is None:
            pil_image = Image.open(self.path)
        else:
            pil_image = self.image

        if self.weight is not None and self.noise is not None:
            self.values["image"] = {"image": pil_image, "weight": self.weight, "noise": self.noise}
        else:
            self.values["image"] = pil_image

        return self.values
