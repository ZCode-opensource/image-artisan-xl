from PIL import Image

from iartisanxl.graph.nodes.node import Node


class ImageLoadNode(Node):
    OUTPUTS = ["image"]

    def __init__(self, path: str = None, image: Image = None, weight: float = None, noise: float = None, noise_index: int = 0):
        super().__init__()
        self.path = path
        self.image = image
        self.weight = weight
        self.noise = noise
        self.noise_index = noise_index

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

    def update_path_weight_noise(self, path: str, weight: float, noise: float, noise_index: int):
        self.path = path
        Image.open(self.path)
        self.weight = weight
        self.noise = noise
        self.noise_index = noise_index
        self.set_updated()

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["path"] = self.path
        node_dict["image"] = self.image
        node_dict["weight"] = self.weight
        node_dict["noise"] = self.noise
        node_dict["noise_index"] = self.noise_index
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(ImageLoadNode, cls).from_dict(node_dict)
        node.path = node_dict["path"]
        node.image = node_dict["image"]
        node.weight = node_dict["weight"]
        node.noise = node_dict["noise"]
        node.noise_index = node_dict["noise_index"]
        return node

    def update_inputs(self, node_dict):
        self.path = node_dict["path"]
        self.image = node_dict["image"]
        self.weight = node_dict["weight"]
        self.noise = node_dict["noise"]
        self.noise_index = node_dict["noise_index"]

    def __call__(self):
        if self.image is None:
            pil_image = Image.open(self.path)
        else:
            pil_image = self.image

        if self.weight is not None and self.noise is not None:
            self.values["image"] = {"image": pil_image, "weight": self.weight, "noise": self.noise, "noise_index": self.noise_index}
        else:
            self.values["image"] = pil_image

        return self.values
