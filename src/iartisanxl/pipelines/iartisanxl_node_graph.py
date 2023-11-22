from collections import deque

import json
import torch

from iartisanxl.nodes.node import Node


class ImageArtisanNodeGraph:
    def __init__(self):
        self.node_counter = 0
        self.nodes = []

        self.cpu_offload = False
        self.sequential_offload = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16

    def add_node(self, node: Node):
        node.id = self.node_counter
        self.nodes.append(node)
        self.node_counter += 1

    def get_node(self, node_class, node_id):
        for node in self.nodes:
            if isinstance(node, node_class) and node.id == node_id:
                return node
        return None

    def delete_node(self, node_class, node_id):
        node = self.get_node(node_class, node_id)
        if node is not None:
            # Disconnect the node from all other nodes
            for other_node in self.nodes:
                other_node.disconnect_from_node(node)
            # Remove the node from the graph
            self.nodes.remove(node)

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
            if node.updated:
                node.device = self.device
                node.torch_dtype = self.torch_dtype
                node.cpu_offload = self.cpu_offload
                node.sequential_offload = self.sequential_offload
                node()

    def save_to_json(self, filename):
        graph_dict = {
            "nodes": [node.to_dict() for node in self.nodes],
            "connections": [],
        }
        for node in self.nodes:
            for input_name, connections in node.connections.items():
                for connected_node, output_name in connections:
                    connection_dict = {
                        "from_node_id": connected_node.id,
                        "from_output_name": output_name,
                        "to_node_id": node.id,
                        "to_input_name": input_name,
                    }
                    graph_dict["connections"].append(connection_dict)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(graph_dict, f)

    def load_from_json(self, filename, node_classes, callbacks=None):
        with open(filename, "r", encoding="utf-8") as f:
            graph_dict = json.load(f)

        id_to_node = {}
        for node_dict in graph_dict["nodes"]:
            NodeClass = node_classes[node_dict["class"]]
            node = NodeClass.from_dict(node_dict, callbacks)
            id_to_node[node.id] = node
            self.nodes.append(node)

        for connection_dict in graph_dict["connections"]:
            from_node = id_to_node[connection_dict["from_node_id"]]
            to_node = id_to_node[connection_dict["to_node_id"]]
            to_node.connect(
                connection_dict["to_input_name"],
                from_node,
                connection_dict["from_output_name"],
            )
