import gc
import json
import time
from collections import deque, defaultdict

import torch

from iartisanxl.graph.nodes.node import Node
from iartisanxl.graph.nodes.image_load_node import ImageLoadNode
from iartisanxl.graph.nodes.controlnet_model_node import ControlnetModelNode
from iartisanxl.graph.nodes.controlnet_node import ControlnetNode


class ImageArtisanNodeGraph:
    def __init__(self, torch_dtype=torch.float16):
        self.node_counter = 0
        self.nodes = []
        self.updated = False
        self.abort_function = lambda: None
        self.executing_node = None

        self.cpu_offload = False
        self.sequential_offload = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype

    def add_node(self, node: Node, name: str = None):
        # Check if the name is not None and is unique
        if name is not None:
            for existing_node in self.nodes:
                if existing_node.name == name:
                    raise ValueError(f"A node with the name {name} already exists in the graph.")
        node.name = name
        node.id = self.node_counter
        node.abort_callable = self.abort_function
        node.updated = True  # if the object already exists and the graph was run with it, this can be False
        self.nodes.append(node)
        self.node_counter += 1

    def get_node(self, node_id):
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_node_by_name(self, name: str):
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_all_nodes_class(self, node_class):
        return [node for node in self.nodes if isinstance(node, node_class)]

    def delete_node_by_id(self, node_id):
        node = self.get_node(node_id)
        if node is not None:
            self.delete_node(node)

    def delete_node_by_name(self, name: str):
        node = self.get_node_by_name(name)
        if node is not None:
            self.delete_node(node)

    def delete_node(self, node):
        # let the node do cleanup before its deletion if needed
        node.before_delete()

        # Disconnect the node from its dependencies and dependents
        for other_node in node.dependencies + node.dependents:
            other_node.disconnect_from_node(node)
            node.disconnect_from_node(other_node)

        # Call the node's delete method
        node.delete()
        # Remove the node from the graph
        self.nodes.remove(node)
        # Delete the node
        del node
        # Call the garbage collector
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    @torch.no_grad()
    def __call__(self):
        self.updated = False
        sorted_nodes = deque()
        visited = set()
        visiting = set()

        def dfs(node):
            visiting.add(node)
            for dependency in sorted(node.dependencies, key=lambda x: x.PRIORITY, reverse=True):
                if dependency in visiting:
                    raise ValueError("Graph contains a cycle")
                if dependency not in visited:
                    dfs(dependency)
            visiting.remove(node)
            visited.add(node)
            sorted_nodes.append(node)

        for node in sorted(self.nodes, key=lambda x: x.PRIORITY, reverse=True):
            if node not in visited:
                dfs(node)

        for node in sorted_nodes:
            if node.updated:
                node.device = self.device
                node.torch_dtype = self.torch_dtype
                node.cpu_offload = self.cpu_offload
                node.sequential_offload = self.sequential_offload
                start_time = time.time()

                try:
                    self.executing_node = node
                    node()
                    node.updated = False
                except KeyError as e:
                    raise KeyError("KeyError occurred in node: " + str(e)) from e

                end_time = time.time()
                node.elapsed_time = end_time - start_time
                self.updated = True
                self.executing_node = None

                if node.abort:
                    node.abort = False
                    self.abort_function()
                    break

    def to_json(self):
        # skip controlnet related nodes for the moment until I define where to save the relevant data
        graph_dict = {
            "nodes": [
                node.to_dict()
                for node in self.nodes
                if not isinstance(node, ImageLoadNode) and not isinstance(node, ControlnetModelNode) and not isinstance(node, ControlnetNode)
            ],
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

        # Clear the current graph
        self.nodes.clear()
        self.node_counter = 0

        id_to_node = {}
        max_id = 0
        for node_dict in graph_dict["nodes"]:
            NodeClass = node_classes[node_dict["class"]]
            node = NodeClass.from_dict(node_dict, callbacks)
            id_to_node[node.id] = node
            self.nodes.append(node)
            if node.id > max_id:
                max_id = node.id

        for connection_dict in graph_dict["connections"]:
            from_node = id_to_node[connection_dict["from_node_id"]]
            to_node = id_to_node[connection_dict["to_node_id"]]
            to_node.connect(
                connection_dict["to_input_name"],
                from_node,
                connection_dict["from_output_name"],
            )
        self.node_counter = max_id + 1

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
        max_id = 0
        for node_dict in graph_dict["nodes"]:
            node_class = node_classes[node_dict["class"]]
            if node_dict["id"] in current_id_to_node and isinstance(current_id_to_node[node_dict["id"]], node_class):
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
                if node_dict["id"] in current_id_to_node:
                    self.delete_node_by_id(node_dict["id"])
                self.nodes.append(node)
            new_id_to_node[node.id] = node
            if node.id > max_id:
                max_id = node.id

        # Remove nodes that are not in the JSON file
        for node_id, node in current_id_to_node.items():
            if node_id not in new_id_to_node:
                self.delete_node_by_id(node_id)

        # Collect new connections for each node and map nodes to input names
        new_connections = defaultdict(list)
        input_names = {}
        for connection_dict in graph_dict["connections"]:
            from_node = new_id_to_node[connection_dict["from_node_id"]]
            to_node = new_id_to_node[connection_dict["to_node_id"]]
            new_connections[to_node.id].append((from_node.id, connection_dict["from_output_name"]))
            input_names[(to_node.id, from_node.id, connection_dict["from_output_name"])] = connection_dict["to_input_name"]

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

        self.node_counter = max_id + 1

    def update_from_json_file(self, filename, node_classes, callbacks=None):
        with open(filename, "r", encoding="utf-8") as f:
            json_str = f.read()
        self.update_from_json(json_str, node_classes, callbacks)

    def abort_graph(self):
        if self.executing_node is not None:
            self.executing_node.abort = True

    def set_abort_function(self, abort_callable: callable):
        self.abort_function = abort_callable

    def clean_up(self):
        for i in range(len(self.nodes) - 1, -1, -1):
            self.delete_node(self.nodes[i])
        self.node_counter = 0
