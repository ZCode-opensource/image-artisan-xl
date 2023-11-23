from collections import deque, defaultdict

import json
import torch

from iartisanxl.graph.nodes.node import Node


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

    def update_from_json(self, filename, node_classes, callbacks=None):
        with open(filename, "r", encoding="utf-8") as f:
            graph_dict = json.load(f)

        # Create a dictionary mapping node IDs to nodes for the current graph
        current_id_to_node = {node.id: node for node in self.nodes}

        # Store a copy of the original nodes and connections
        original_nodes = {node.id: node.to_dict() for node in self.nodes}
        original_connections = {node.id: node.connections.copy() for node in self.nodes}

        # Keep track of which nodes were marked as updated
        updated_nodes = set()

        # Load nodes from the JSON file
        new_id_to_node = {}
        for node_dict in graph_dict["nodes"]:
            NodeClass = node_classes[node_dict["class"]]
            if node_dict["id"] in current_id_to_node and isinstance(
                current_id_to_node[node_dict["id"]], NodeClass
            ):
                # If the node already exists and is of the same class, check if its inputs have changed
                node = current_id_to_node[node_dict["id"]]
                new_node = NodeClass.from_dict(node_dict, callbacks)
                if node.to_dict() != new_node.to_dict():
                    # If the inputs have changed, update the node and mark it as updated
                    node.update_inputs(node_dict)
                    node.set_updated(updated_nodes)
            else:
                # If the node does not exist or is of a different class, create a new node and mark it as updated
                node = NodeClass.from_dict(node_dict, callbacks)
                node.set_updated(updated_nodes)
                self.nodes.append(node)
            new_id_to_node[node.id] = node

        # Remove nodes that are not in the JSON file
        for node_id, node in current_id_to_node.items():
            if node_id not in new_id_to_node:
                self.delete_node(type(node), node_id)

        # Update connections
        for node in self.nodes:
            node.dependencies = []
            node.connections = defaultdict(list)
        for connection_dict in graph_dict["connections"]:
            from_node = new_id_to_node[connection_dict["from_node_id"]]
            to_node = new_id_to_node[connection_dict["to_node_id"]]
            to_node.connect(
                connection_dict["to_input_name"],
                from_node,
                connection_dict["from_output_name"],
            )
            if node.id not in updated_nodes:
                updated_nodes.add(node.id)

        # Restore any nodes and connections that haven't changed
        for node in self.nodes:
            if (
                node.id in original_nodes
                and node.connections == original_connections[node.id]
                and node.id not in updated_nodes
            ):
                node.connections = original_connections[node.id]
                node.updated = False
