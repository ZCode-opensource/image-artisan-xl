from collections import defaultdict


class Node:
    REQUIRED_INPUTS = []
    OPTIONAL_INPUTS = []
    OUTPUTS = []

    def __init__(self):
        self.id = None
        self.updated = True
        self.dependencies = []
        self.dependents = []
        self.values = {}
        self.connections = defaultdict(list)

        self.device = None
        self.torch_dtype = None
        self.cpu_offload = False
        self.sequential_offload = False

    def connect(self, input_name: str, node, output_name: str):
        if not isinstance(node, Node):
            raise TypeError("node must be an instance of Node or its subclass")

        if input_name not in self.REQUIRED_INPUTS + self.OPTIONAL_INPUTS:
            raise ValueError(
                f'The input "{input_name}" is not present in "{self.__class__.__name__}"'
            )
        if output_name not in node.OUTPUTS:
            raise ValueError(
                f'The output "{output_name}" is not present in "{node.__class__.__name__}"'
            )
        self.dependencies.append(node)
        self.connections[input_name].append((node, output_name))
        node.dependents.append(self)
        self.updated = True

    def disconnect(self, input_name: str, node, output_name: str):
        if input_name in self.connections:
            self.connections[input_name] = [
                (n, out_name)
                for n, out_name in self.connections[input_name]
                if not (n == node and out_name == output_name)
            ]
            if not self.connections[input_name]:
                del self.connections[input_name]
        node.dependents.remove(self)
        self.updated = True

    def disconnect_from_node(self, node):
        self.dependencies = [dep for dep in self.dependencies if dep != node]
        for input_name, conns in list(self.connections.items()):
            self.connections[input_name] = [
                (n, output_name) for n, output_name in conns if n != node
            ]
            if not self.connections[input_name]:
                del self.connections[input_name]
        if self in node.dependents:
            node.dependents.remove(self)
        self.updated = True

    def connections_changed(self, new_connections):
        # Convert current connections to a format that can be compared with new_connections
        current_connections = [
            (dep.id, output_name)
            for input_name, deps in self.connections.items()
            for dep, output_name in deps
        ]
        return set(current_connections) != set(new_connections)

    def set_updated(self, updated_nodes=None):
        self.updated = True
        if updated_nodes is not None:
            updated_nodes.add(self.id)
        for dependent in self.dependents:
            dependent.set_updated(updated_nodes)

    def __getattr__(self, name):
        if name in self.REQUIRED_INPUTS + self.OPTIONAL_INPUTS:
            return self.get_input_value(name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def get_input_value(self, input_name):
        if input_name in self.connections:
            values = [
                node.values[output_name]
                for node, output_name in self.connections[input_name]
            ]
            return values if len(values) > 1 else values[0]
        elif input_name in self.OPTIONAL_INPUTS:
            return None
        else:
            raise ValueError(
                f'The required input "{input_name}" is not connected in "{self.__class__.__name__}"'
            )

    def __call__(self):
        self.updated = False

    def to_dict(self):
        return {
            "class": type(self).__name__,
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = cls()
        node.id = node_dict["id"]
        return node

    def update_inputs(self, node_dict):
        pass
