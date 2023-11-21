import logging

import torch
from PIL import Image

from PyQt6.QtCore import QThread, pyqtSignal

from iartisanxl.generation.generation_data_object import ImageGenData
from iartisanxl.pipelines.iartisanxl_node_graph import ImageArtisanNodeGraph
from iartisanxl.nodes.stable_difussion_xl_model_node import StableDiffusionXLModelNode
from iartisanxl.nodes.text_node import TextNode
from iartisanxl.nodes.prompts_encoder_node import PromptsEncoderNode
from iartisanxl.nodes.vae_model_node import VaeModelNode
from iartisanxl.nodes.latents_node import LatentsNode
from iartisanxl.nodes.number_node import NumberNode
from iartisanxl.nodes.scheduler_node import SchedulerNode
from iartisanxl.nodes.image_generation_node import ImageGenerationNode
from iartisanxl.nodes.latents_decoder_node import LatentsDecoderNode
from iartisanxl.nodes.image_send_node import ImageSendNode
from iartisanxl.nodes.lora_node import LoraNode


class NodeGraphThread(QThread):
    status_changed = pyqtSignal(str)
    progress_update = pyqtSignal(int, torch.Tensor)
    generation_finished = pyqtSignal(Image.Image, float)
    generation_error = pyqtSignal(str, bool)
    generation_aborted = pyqtSignal()

    def __init__(
        self,
        rendering_generation_data: ImageGenData,
        model_offload: bool = False,
        sequential_offload: bool = False,
        torch_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.logger = logging.getLogger()
        self.image_generation_data = rendering_generation_data
        self.model_offload = model_offload
        self.sequential_offload = sequential_offload
        self.torch_dtype = torch_dtype
        self.abort = False

    def run(self):
        self.status_changed.emit("Generating image...")

        node_graph = ImageArtisanNodeGraph()

        sdxl_model = StableDiffusionXLModelNode(
            path=self.image_generation_data.model.path
        )
        node_graph.add_node(sdxl_model)

        positive_prompt = TextNode(
            text=self.image_generation_data.positive_prompt_clipg
        )
        node_graph.add_node(positive_prompt)

        # check if there are loras
        lora_nodes = []
        lora_scale = None
        if len(self.image_generation_data.loras) > 0:
            lora_scale = NumberNode(self.image_generation_data.lora_scale)
            node_graph.add_node(lora_scale)

            for lora in self.image_generation_data.loras:
                if lora.enabled:
                    lora_node = LoraNode(
                        path=lora.path,
                        adapter_name=lora.filename,
                        scale=lora.weight,
                    )
                    lora_node.connect("unet", sdxl_model, "unet")
                    lora_node.connect("text_encoder_1", sdxl_model, "text_encoder_1")
                    lora_node.connect("global_lora_scale", lora_scale, "value")
                    lora_node.connect("text_encoder_2", sdxl_model, "text_encoder_2")
                    node_graph.add_node(lora_node)
                    lora_nodes.append(lora_node)

        prompts_encoder = PromptsEncoderNode()
        prompts_encoder.connect("tokenizer_1", sdxl_model, "tokenizer_1")
        prompts_encoder.connect("tokenizer_2", sdxl_model, "tokenizer_2")
        prompts_encoder.connect("text_encoder_1", sdxl_model, "text_encoder_1")
        prompts_encoder.connect("text_encoder_2", sdxl_model, "text_encoder_2")
        prompts_encoder.connect("prompt_1", positive_prompt, "value")
        if lora_scale is not None:
            prompts_encoder.connect("global_lora_scale", lora_scale, "value")
        node_graph.add_node(prompts_encoder)

        vae_model = VaeModelNode(self.image_generation_data.vae.path)
        node_graph.add_node(vae_model)

        seed = NumberNode(number=self.image_generation_data.seed)
        node_graph.add_node(seed)

        height = NumberNode(number=self.image_generation_data.image_height)
        node_graph.add_node(height)

        width = NumberNode(number=self.image_generation_data.image_width)
        node_graph.add_node(width)

        latents = LatentsNode()
        latents.connect("seed", seed, "value")
        latents.connect("num_channels_latents", sdxl_model, "num_channels_latents")
        latents.connect("height", height, "value")
        latents.connect("width", width, "value")
        latents.connect("vae_scale_factor", vae_model, "vae_scale_factor")
        node_graph.add_node(latents)

        scheduler = SchedulerNode(
            scheduler_index=self.image_generation_data.base_scheduler
        )
        node_graph.add_node(scheduler)

        steps = NumberNode(number=self.image_generation_data.steps)
        node_graph.add_node(steps)

        guidance_scale = NumberNode(number=self.image_generation_data.guidance)
        node_graph.add_node(guidance_scale)

        image_generation = ImageGenerationNode(callback=self.step_progress_update)
        image_generation.connect("scheduler", scheduler, "scheduler")
        image_generation.connect("num_inference_steps", steps, "value")
        image_generation.connect("latents", latents, "latents")
        image_generation.connect(
            "pooled_prompt_embeds", prompts_encoder, "pooled_prompt_embeds"
        )
        image_generation.connect("height", height, "value")
        image_generation.connect("width", width, "value")
        image_generation.connect("prompt_embeds", prompts_encoder, "prompt_embeds")
        image_generation.connect("guidance_scale", guidance_scale, "value")
        image_generation.connect(
            "negative_prompt_embeds", prompts_encoder, "negative_prompt_embeds"
        )
        image_generation.connect(
            "negative_pooled_prompt_embeds",
            prompts_encoder,
            "negative_pooled_prompt_embeds",
        )
        image_generation.connect("unet", sdxl_model, "unet")
        image_generation.connect("generator", latents, "generator")
        if lora_scale is not None:
            for lora_node in lora_nodes:
                image_generation.connect("lora", lora_node, "lora")
        node_graph.add_node(image_generation)

        decoder = LatentsDecoderNode()
        decoder.connect("vae", vae_model, "vae")
        decoder.connect("latents", image_generation, "latents")
        node_graph.add_node(decoder)

        image_send = ImageSendNode(image_callback=self.preview_image)
        image_send.connect("image", decoder, "image")
        node_graph.add_node(image_send)

        node_graph()

    def step_progress_update(self, step, _timestep, latents):
        self.progress_update.emit(step, latents)

    def preview_image(self, image):
        self.generation_finished.emit(image, 0)
