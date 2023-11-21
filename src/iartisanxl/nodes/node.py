from typing import Any


class Node:
    REQUIRED_ARGS = []
    PRIORITY = 0
    OUTPUTS = []
    INPUTS = []

    def __init__(self, **kwargs):
        self.device = None
        self.torch_dtype = None
        self.can_offload = False
        self.sequential_offload = False

        missing_args = [arg for arg in self.REQUIRED_ARGS if arg not in kwargs]
        if missing_args:
            raise ValueError(f"Missing required arguments: {missing_args}")
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def enable_cpu_offload(self):
        if self.can_offload:
            self.device = "cpu"

    def enable_sequential_cpu_offload(self):
        self.sequential_offload = True
