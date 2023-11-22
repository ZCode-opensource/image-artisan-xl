from iartisanxl.nodes.node import Node


class TextNode(Node):
    OUTPUTS = ["value"]
    INPUTS = []

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def update_text(self, text: str):
        self.text = text
        self.set_updated()

    def __call__(self):
        super().__call__()
        self.values["value"] = self.text

        return self.values
