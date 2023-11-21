# pylint: disable=no-member

from iartisanxl.nodes.node import Node


class PromptsNode(Node):
    PRIORITY = 2
    REQUIRED_ARGS = [
        "prompt_1",
    ]
    OUTPUTS = [
        "prompt_1",
        "prompt_2",
        "negative_prompt_1",
        "negative_prompt_2",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_2 = kwargs.get("prompt_2", None)
        self.negative_prompt_1 = kwargs.get("negative_prompt_1", "")
        self.negative_prompt_2 = kwargs.get("negative_prompt_2", None)

    def __call__(self) -> tuple[str, str, str, str]:
        return (
            self.prompt_1,
            self.prompt_2,
            self.negative_prompt_1,
            self.negative_prompt_2,
        )
