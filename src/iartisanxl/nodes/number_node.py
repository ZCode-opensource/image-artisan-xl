from typing import Union

from iartisanxl.nodes.node import Node


class NumberNode(Node):
    OUTPUTS = ["value"]
    INPUTS = []

    def __init__(self, number: Union[int, float]):
        super().__init__()
        self.number = number

    def update_number(self, number: Union[int, float]):
        self.number = number
        self.set_updated()

    def __call__(self):
        super().__call__()
        self.values["value"] = self.number

        return self.values
