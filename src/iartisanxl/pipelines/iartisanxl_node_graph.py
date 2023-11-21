from collections import deque

import torch

from iartisanxl.nodes.node import Node


class ImageArtisanNodeGraph:
    def __init__(self):
        self.nodes = []

        self.cpu_offload = False
        self.sequential_offload = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16

    def add_node(self, node: Node):
        self.nodes.append(node)

    @torch.no_grad()
    def __call__(self):
        sorted_nodes = deque()
        visited = set()
        visiting = set()

        def dfs(node):
            visiting.add(node)
            for dependency in node.dependencies:
                if dependency in visiting:
                    raise ValueError("Graph contains a cycle")
                if dependency not in visited:
                    dfs(dependency)
            visiting.remove(node)
            visited.add(node)
            sorted_nodes.append(node)

        for node in self.nodes:
            if node not in visited:
                dfs(node)

        for node in sorted_nodes:
            node.device = self.device
            node.torch_dtype = self.torch_dtype
            node.cpu_offload = self.cpu_offload
            node.sequential_offload = self.sequential_offload

            node()
