from typing import Union

from iartisanxl.nodes.node import Node


class NumberNode(Node):
    OUTPUTS = ["value"]
    INPUTS = []

    def __init__(self, number: Union[int, float]):
        super().__init__()
        self.number = number

    def __call__(self):
        self.values["value"] = self.number

        return self.values
