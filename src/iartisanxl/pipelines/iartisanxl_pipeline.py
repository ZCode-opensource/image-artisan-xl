import torch

from iartisanxl.nodes.node import Node


class ImageArtisanPipeline:
    def __init__(self):
        super().__init__()

        self.nodes: list[Node] = []
        self.data: dict[str, any] = {}

        self.cpu_offload = False
        self.sequential_offload = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16

    def add_node(self, node: Node):
        optional_inputs = [
            "global_lora_scale",
            "clip_skip",
            "cross_attention_kwargs",
            "active_loras",
        ]
        missing_inputs = [
            input
            for input in node.INPUTS
            if input not in self.data and input not in optional_inputs
        ]
        if missing_inputs:
            raise ValueError(f"Missing required inputs: {missing_inputs}")

        node.device = self.device
        node.torch_dtype = self.torch_dtype

        self.nodes.append(node)

        for output in node.OUTPUTS:
            self.data[output] = None

    def sort_nodes(self):
        self.nodes.sort(key=lambda node: node.PRIORITY)

    @torch.no_grad()
    def __call__(self):
        self.sort_nodes()

        # Identify the large models and store their original devices
        large_models = ["text_encoder_1", "text_encoder_2", "unet", "vae"]
        original_devices = {}

        for node in self.nodes:
            # Prepare the inputs for the node
            inputs = {input: self.data.get(input, None) for input in node.INPUTS}

            # If the inputs are one of the large models and they are not on the pipeline's device, move them to the pipeline's device
            if self.cpu_offload:
                node.enable_cpu_offload()

                for model in large_models:
                    if (
                        model in inputs
                        and isinstance(inputs[model], torch.nn.Module)
                        and inputs[model].device != self.device
                    ):
                        original_devices[model] = inputs[model].device
                        inputs[model] = inputs[model].to(self.device)
            else:
                if self.sequential_offload:
                    node.enable_sequential_cpu_offload()

            # Call the node with the prepared inputs
            outputs = node(**inputs)

            # Move the large models back to their original device
            if self.cpu_offload:
                for model, original_device in original_devices.items():
                    if (
                        model in self.data
                        and isinstance(self.data[model], torch.nn.Module)
                        and self.data[model].device != original_device
                    ):
                        self.data[model] = self.data[model].to(original_device)

            # If the outputs are PyTorch modules or tensors, move them to the correct device
            if (
                not self.cpu_offload
                and hasattr(node, "can_offload")
                and not node.can_offload
            ):
                if isinstance(outputs, tuple):
                    outputs = tuple(
                        output.to(self.device)
                        if isinstance(output, (torch.nn.Module, torch.Tensor))
                        else output
                        for output in outputs
                    )
                elif isinstance(outputs, (torch.nn.Module, torch.Tensor)):
                    outputs = outputs.to(self.device)

            # Store the outputs in the data dictionary
            if isinstance(outputs, tuple):
                for output, value in zip(node.OUTPUTS, outputs):
                    self.data[output] = value
            else:
                if len(node.OUTPUTS) > 0:
                    self.data[node.OUTPUTS[0]] = outputs

    def to(self, device):
        self.device = device  # store the device
        for node in self.nodes:
            for attr_name in dir(node):
                attr = getattr(node, attr_name)
                if isinstance(attr, torch.nn.Module) or isinstance(attr, torch.Tensor):
                    setattr(node, attr_name, attr.to(device))
