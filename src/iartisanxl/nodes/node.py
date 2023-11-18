from typing import Any


class Node:
    REQUIRED_ARGS = []

    def __init__(self, **kwargs):
        missing_args = [arg for arg in self.REQUIRED_ARGS if arg not in kwargs]
        if missing_args:
            raise ValueError(f"Missing required arguments: {missing_args}")
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
