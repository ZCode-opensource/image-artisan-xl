from iartisanxl.graph.nodes.node import Node


class TextNode(Node):
    PRIORITY = 2
    OUTPUTS = ["value"]
    INPUTS = []

    def __init__(self, text: str = None):
        super().__init__()
        self.text = text

    def update_value(self, text: str):
        self.text = text
        self.set_updated()

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["text"] = self.text
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(TextNode, cls).from_dict(node_dict)
        node.text = node_dict["text"]
        return node

    def update_inputs(self, node_dict):
        self.text = node_dict["text"]

    def __call__(self):
        self.values["value"] = self.text

        return self.values
