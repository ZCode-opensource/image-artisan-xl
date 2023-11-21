from collections import defaultdict


class Node:
    REQUIRED_INPUTS = []
    OPTIONAL_INPUTS = []
    OUTPUTS = []

    def __init__(self):
        self.dependencies = []
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
