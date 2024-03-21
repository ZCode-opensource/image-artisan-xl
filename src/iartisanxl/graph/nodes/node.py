from collections import defaultdict

from iartisanxl.graph.iartisan_node_error import IArtisanNodeError


class Node:
    PRIORITY = 0
    REQUIRED_INPUTS = []
    OPTIONAL_INPUTS = []
    OUTPUTS = []

    def __init__(self):
        self.id = None
        self.enabled = True
        self.name = None
        self.elapsed_time = None
        self.updated = True
        self.abort = False
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
            raise ValueError(f'The input "{input_name}" is not present in "{self.__class__.__name__}"')
        if output_name not in node.OUTPUTS:
            raise ValueError(f'The output "{output_name}" is not present in "{node.__class__.__name__}"')
        self.dependencies.append(node)
        self.connections[input_name].append((node, output_name))
        node.dependents.append(self)
        self.updated = True

        for dependent_node in self.dependents:
            dependent_node.set_updated()

    def disconnect(self, input_name: str, node, output_name: str):
        if input_name in self.connections:
            self.connections[input_name] = [
                (n, out_name)
                for n, out_name in self.connections[input_name]
                if not (n == node and out_name == output_name)
            ]
            if not self.connections[input_name]:
                del self.connections[input_name]
        if node in self.dependencies:
            self.dependencies.remove(node)
        if self in node.dependents:
            node.dependents.remove(self)

    def disconnect_from_node(self, node):
        self.dependencies = [dep for dep in self.dependencies if dep != node]
        for input_name, conns in list(self.connections.items()):
            self.connections[input_name] = [(n, output_name) for n, output_name in conns if n != node]
            if not self.connections[input_name]:
                del self.connections[input_name]
                self.set_updated()
        if self in node.dependents:
            node.dependents.remove(self)
            self.set_updated()

    def connections_changed(self, new_connections):
        # Convert current connections to a format that can be compared with new_connections
        current_connections = [
            (dep.id, output_name) for input_name, deps in self.connections.items() for dep, output_name in deps
        ]
        return set(current_connections) != set(new_connections)

    def set_updated(self, updated_nodes=None, update_dependents=True):
        if updated_nodes is None:
            updated_nodes = set()
        if self.id in updated_nodes:
            return
        self.updated = True
        updated_nodes.add(self.id)
        if update_dependents:
            for dependent in self.dependents:
                dependent.set_updated(updated_nodes)

    def __getattr__(self, name):
        if name in self.REQUIRED_INPUTS + self.OPTIONAL_INPUTS:
            return self.get_input_value(name)
        raise IArtisanNodeError(f"There is no attribute '{name}'", self.__class__.__name__)

    def get_input_value(self, input_name):
        if input_name in self.connections:
            values = []
            for node, output_name in self.connections[input_name]:
                if output_name not in node.values:
                    raise IArtisanNodeError(
                        f"The required output '{output_name}' is not in node.values.", {self.__class__.__name__}
                    )
                values.append(node.values[output_name])
            return values if len(values) > 1 else values[0]
        elif input_name in self.OPTIONAL_INPUTS:
            return None
        else:
            raise IArtisanNodeError(f"The required input '{input_name}' is not connected.", {self.__class__.__name__})

    def __call__(self):
        pass

    def to_dict(self):
        return {
            "class": type(self).__name__,
            "id": self.id,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = cls()
        node.id = node_dict["id"]
        if "name" in node_dict:
            node.name = node_dict["name"]
        return node

    def update_inputs(self, node_dict):
        pass

    def before_delete(self):
        pass

    def delete(self):
        # Clean up the node's data
        self.dependencies.clear()
        self.dependents.clear()
        self.values.clear()
        self.connections.clear()

        # Reset other attributes
        self.id = None
        self.name = None
        self.updated = False
        self.device = None
        self.torch_dtype = None
        self.cpu_offload = False
        self.sequential_offload = False

    def abort_call(self):
        self.abort = True
