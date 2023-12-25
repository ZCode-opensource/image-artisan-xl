from PIL import Image

from iartisanxl.graph.nodes.node import Node


class ImageLoadNode(Node):
    OUTPUTS = ["image"]

    def __init__(self, path: str = None, image: Image = None, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.image = image

    def update_value(self, path: str):
        self.path = path
        self.set_updated()

    def update_image(self, image: Image):
        self.image = image
        self.set_updated()

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["path"] = self.path
        node_dict["image"] = self.image
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(ImageLoadNode, cls).from_dict(node_dict)
        node.path = node_dict["path"]
        node.image = node_dict["image"]
        return node

    def update_inputs(self, node_dict):
        self.path = node_dict["path"]
        self.image = node_dict["image"]

    def __call__(self):
        super().__call__()

        if self.image is None:
            self.values["image"] = Image.open(self.path)
        else:
            self.values["image"] = self.image

        return self.values
