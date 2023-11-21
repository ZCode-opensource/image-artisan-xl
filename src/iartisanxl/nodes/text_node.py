from iartisanxl.nodes.node import Node


class TextNode(Node):
    OUTPUTS = ["value"]
    INPUTS = []

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def __call__(self):
        self.values["value"] = self.text

        return self.values
