from typing import Union

from iartisanxl.graph.nodes.node import Node


class NumberNode(Node):
    OUTPUTS = ["value"]
    INPUTS = []

    def __init__(self, number: Union[int, float] = None):
        super().__init__()
        self.number = number

    def update_value(self, number: Union[int, float]):
        self.number = number
        self.set_updated()

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["number"] = self.number
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(NumberNode, cls).from_dict(node_dict)
        node.number = node_dict["number"]
        return node

    def update_inputs(self, node_dict):
        self.number = node_dict["number"]

    def __call__(self):
        super().__call__()
        self.values["value"] = self.number

        return self.values
