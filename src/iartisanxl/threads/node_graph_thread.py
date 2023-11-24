import logging

import torch
from PIL import Image

from PyQt6.QtCore import QThread, pyqtSignal

from iartisanxl.generation.generation_data_object import ImageGenData
from iartisanxl.graph.iartisanxl_node_graph import ImageArtisanNodeGraph
from iartisanxl.graph.nodes.stable_difussion_xl_model_node import (
    StableDiffusionXLModelNode,
)
from iartisanxl.graph.nodes.text_node import TextNode
from iartisanxl.graph.nodes.prompts_encoder_node import PromptsEncoderNode
from iartisanxl.graph.nodes.vae_model_node import VaeModelNode
from iartisanxl.graph.nodes.latents_node import LatentsNode
from iartisanxl.graph.nodes.number_node import NumberNode
from iartisanxl.graph.nodes.scheduler_node import SchedulerNode
from iartisanxl.graph.nodes.image_generation_node import ImageGenerationNode
from iartisanxl.graph.nodes.latents_decoder_node import LatentsDecoderNode
from iartisanxl.graph.nodes.image_send_node import ImageSendNode
from iartisanxl.graph.nodes.lora_node import LoraNode


class NodeGraphThread(QThread):
    status_changed = pyqtSignal(str)
    progress_update = pyqtSignal(int, torch.Tensor)
    generation_finished = pyqtSignal(Image.Image, float)
    generation_error = pyqtSignal(str, bool)
    generation_aborted = pyqtSignal()

    def __init__(
        self,
        node_graph: ImageArtisanNodeGraph = None,
        image_generation_data: ImageGenData = None,
        model_offload: bool = False,
        sequential_offload: bool = False,
        torch_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.logger = logging.getLogger()
        self.node_graph = node_graph
        self.image_generation_data = image_generation_data
        self.model_offload = model_offload
        self.sequential_offload = sequential_offload
        self.torch_dtype = torch_dtype
        self.abort = False

    def run(self):
        self.status_changed.emit("Generating image...")

        if self.node_graph.sequential_offload != self.sequential_offload:
            self.check_and_update(
                "sequential_offload", "sequential_offload", self.sequential_offload
            )
        elif self.node_graph.cpu_offload != self.model_offload:
            self.check_and_update("cpu_offload", "model_offload", self.model_offload)

        node_classes = {
            "StableDiffusionXLModelNode": StableDiffusionXLModelNode,
            "TextNode": TextNode,
            "PromptsEncoderNode": PromptsEncoderNode,
            "VaeModelNode": VaeModelNode,
            "NumberNode": NumberNode,
            "LatentsNode": LatentsNode,
            "SchedulerNode": SchedulerNode,
            "ImageGenerationNode": ImageGenerationNode,
            "LatentsDecoderNode": LatentsDecoderNode,
            "ImageSendNode": ImageSendNode,
            "LoraNode": LoraNode,
        }
        callbacks = {
            "abort": lambda: False,
            "step_progress_update": self.step_progress_update,
            "preview_image": self.preview_image,
        }

        self.node_graph.update_from_json(
            self.image_generation_data.to_json_graph(), node_classes, callbacks
        )

        self.node_graph()

        if not self.node_graph.updated:
            self.generation_error.emit("Nothing was changed", False)

    def step_progress_update(self, step, _timestep, latents):
        self.progress_update.emit(step, latents)

    def preview_image(self, image):
        self.generation_finished.emit(image, 0)

    def reset_model_path(self, model_name):
        model_node = self.node_graph.get_node_by_name(model_name)
        if model_node is not None:
            model_node.path = ""  # force reload of model

    def check_and_update(self, attr1, attr2, value):
        if getattr(self.node_graph, attr1) != getattr(self, attr2):
            self.reset_model_path("sdxl_model")
            self.reset_model_path("vae_model")
            setattr(self.node_graph, attr1, value)
