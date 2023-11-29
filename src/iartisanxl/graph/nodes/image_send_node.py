from iartisanxl.graph.nodes.node import Node


class ImageSendNode(Node):
    REQUIRED_INPUTS = ["image"]
    OUTPUTS = []

    def __init__(self, image_callback: callable = None, **kwargs):
        super().__init__(**kwargs)

        self.image_callback = image_callback

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["image_callback"] = (
            self.image_callback.__name__ if self.image_callback else None
        )
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, callbacks=None):
        node = super(ImageSendNode, cls).from_dict(node_dict)
        node.image_callback = (
            callbacks.get(node_dict["image_callback"]) if callbacks else None
        )
        return node

    def __call__(self):
        super().__call__()
        self.image_callback(self.image)
