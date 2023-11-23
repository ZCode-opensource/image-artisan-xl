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

    def to_json(self):
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

        return json.dumps(graph_dict)

    def from_json(self, json_str, node_classes, callbacks=None):
        graph_dict = json.loads(json_str)

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

    def save_to_json(self, filename):
        json_str = self.to_json()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(json_str)

    def load_from_json(self, filename, node_classes, callbacks=None):
        with open(filename, "r", encoding="utf-8") as f:
            json_str = f.read()
        self.from_json(json_str, node_classes, callbacks)

    def update_from_json(self, json_str, node_classes, callbacks=None):
        graph_dict = json.loads(json_str)

        # Create a dictionary mapping node IDs to nodes for the current graph
        current_id_to_node = {node.id: node for node in self.nodes}

        # Keep track of which nodes were marked as updated
        updated_nodes = set()

        # Load nodes from the JSON file
        new_id_to_node = {}
        for node_dict in graph_dict["nodes"]:
            node_class = node_classes[node_dict["class"]]
            if node_dict["id"] in current_id_to_node and isinstance(
                current_id_to_node[node_dict["id"]], node_class
            ):
                # If the node already exists and is of the same class, check if its inputs have changed
                node = current_id_to_node[node_dict["id"]]
                new_node = node_class.from_dict(node_dict, callbacks)
                if node.to_dict() != new_node.to_dict():
                    # If the inputs have changed, update the node and mark it as updated
                    node.update_inputs(node_dict)
                    node.set_updated(updated_nodes)
            else:
                # If the node does not exist or is of a different class, create a new node and mark it as updated
                node = node_class.from_dict(node_dict, callbacks)
                node.set_updated(updated_nodes)
                self.nodes.append(node)
            new_id_to_node[node.id] = node

        # Remove nodes that are not in the JSON file
        for node_id, node in current_id_to_node.items():
            if node_id not in new_id_to_node:
                self.delete_node(type(node), node_id)

        # Collect new connections for each node and map nodes to input names
        new_connections = defaultdict(list)
        input_names = {}
        for connection_dict in graph_dict["connections"]:
            from_node = new_id_to_node[connection_dict["from_node_id"]]
            to_node = new_id_to_node[connection_dict["to_node_id"]]
            new_connections[to_node.id].append(
                (from_node.id, connection_dict["from_output_name"])
            )
            input_names[
                (to_node.id, from_node.id, connection_dict["from_output_name"])
            ] = connection_dict["to_input_name"]

        # Update connections
        for node in self.nodes:
            if node.connections_changed(new_connections[node.id]):
                node.dependencies = []
                node.connections = defaultdict(list)
                for from_node_id, output_name in new_connections[node.id]:
                    from_node = new_id_to_node[from_node_id]
                    input_name = input_names[(node.id, from_node_id, output_name)]
                    node.connect(
                        input_name,
                        from_node,
                        output_name,
                    )

        # Restore any nodes that haven't changed
        for node in self.nodes:
            if node.id not in updated_nodes:
                node.updated = False

    def update_from_json_file(self, filename, node_classes, callbacks=None):
        with open(filename, "r", encoding="utf-8") as f:
            json_str = f.read()
        self.update_from_json(json_str, node_classes, callbacks)
