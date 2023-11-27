import copy
import json

import attr

from iartisanxl.generation.model_data_object import ModelDataObject
from iartisanxl.generation.vae_data_object import VaeDataObject
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


@attr.s(auto_attribs=True, slots=True)
class ImageGenerationData:
    module: str = "texttoimage"
    seed: int = 0
    image_width: int = 1024
    image_height: int = 1024
    steps: int = 20
    guidance: float = 7.5
    base_scheduler: int = 0
    lora_scale: float = 1.0
    model: ModelDataObject = attr.Factory(ModelDataObject)
    vae: VaeDataObject = attr.Factory(VaeDataObject)
    positive_prompt_clipl: str = None
    positive_prompt_clipg: str = ""
    negative_prompt_clipl: str = None
    negative_prompt_clipg: str = None
    clip_skip: int = 0

    previous_state: dict = attr.Factory(dict)

    def update_previous_state(self):
        self.previous_state = copy.deepcopy(attr.asdict(self))

    def get_changed_attributes(self):
        current_state = attr.asdict(self)
        changed_attributes = {
            k: v
            for k, v in current_state.items()
            if k != "previous_state"
            and self.previous_state
            and self.previous_state[k] != v
        }
        return changed_attributes

    def update_from_json(self, json_graph):
        data = json.loads(json_graph)

        loras = []

        for node in data["nodes"]:
            if node["name"] == "lora_scale":
                self.lora_scale = node["number"]
            elif node["name"] == "clip_skip":
                self.clip_skip = node["number"]
            elif node["name"] == "model":
                self.model.name = node["model_name"]
                self.model.path = node["path"]
                self.model.version = node["version"]
                self.model.type = node["model_type"]
            elif node["name"] == "positive_prompt_clipg":
                self.positive_prompt_clipg = node["text"]
            elif node["name"] == "positive_prompt_clipl":
                self.positive_prompt_clipl = node["text"]
            elif node["name"] == "negative_prompt_clipg":
                self.negative_prompt_clipg = node["text"]
            elif node["name"] == "negative_prompt_clipl":
                self.negative_prompt_clipl = node["text"]
            elif node["name"] == "vae_model":
                self.vae.name = node["vae_name"]
                self.vae.path = node["path"]
            elif node["name"] == "seed":
                self.seed = node["number"]
            elif node["name"] == "image_width":
                self.image_width = node["number"]
            elif node["name"] == "image_height":
                self.image_height = node["number"]
            elif node["name"] == "base_scheduler":
                self.base_scheduler = node["scheduler_index"]
            else:
                if node["class"] == "LoraNode":
                    loras.append(node)

        return loras

    def create_text_to_image_graph(self) -> ImageArtisanNodeGraph:
        node_graph = ImageArtisanNodeGraph()

        lora_scale = NumberNode(number=self.lora_scale)
        node_graph.add_node(lora_scale, "lora_scale")

        clip_skip = NumberNode(number=self.clip_skip)
        node_graph.add_node(clip_skip, "clip_skip")

        sdxl_model = StableDiffusionXLModelNode(
            path=self.model.path,
            model_name=self.model.name,
            version=self.model.version,
            model_type=self.model.type,
        )
        node_graph.add_node(sdxl_model, "model")

        positive_prompt_1 = TextNode(text=self.positive_prompt_clipg)
        node_graph.add_node(positive_prompt_1, "positive_prompt_clipg")

        positive_prompt_2 = TextNode(text=self.positive_prompt_clipl)
        node_graph.add_node(positive_prompt_2, "positive_prompt_clipl")

        negative_prompt_1 = TextNode(text=self.negative_prompt_clipg)
        node_graph.add_node(negative_prompt_1, "negative_prompt_clipg")

        negative_prompt_2 = TextNode(text=self.negative_prompt_clipl)
        node_graph.add_node(negative_prompt_2, "negative_prompt_clipl")

        prompts_encoder = PromptsEncoderNode()
        prompts_encoder.connect("tokenizer_1", sdxl_model, "tokenizer_1")
        prompts_encoder.connect("tokenizer_2", sdxl_model, "tokenizer_2")
        prompts_encoder.connect("text_encoder_1", sdxl_model, "text_encoder_1")
        prompts_encoder.connect("text_encoder_2", sdxl_model, "text_encoder_2")
        prompts_encoder.connect("positive_prompt_1", positive_prompt_1, "value")
        prompts_encoder.connect("positive_prompt_2", positive_prompt_2, "value")
        prompts_encoder.connect("negative_prompt_1", negative_prompt_1, "value")
        prompts_encoder.connect("negative_prompt_2", negative_prompt_2, "value")
        prompts_encoder.connect("global_lora_scale", lora_scale, "value")
        node_graph.add_node(prompts_encoder, "prompts_encoder")

        vae_model = VaeModelNode(path=self.vae.path, vae_name=self.vae.name)
        node_graph.add_node(vae_model, "vae_model")

        seed = NumberNode(number=self.seed)
        node_graph.add_node(seed, "seed")

        image_width = NumberNode(number=self.image_width)
        node_graph.add_node(image_width, "image_width")

        image_height = NumberNode(number=self.image_height)
        node_graph.add_node(image_height, "image_height")

        latents = LatentsNode()
        latents.connect("seed", seed, "value")
        latents.connect("num_channels_latents", sdxl_model, "num_channels_latents")
        latents.connect("width", image_width, "value")
        latents.connect("height", image_height, "value")
        latents.connect("vae_scale_factor", vae_model, "vae_scale_factor")
        node_graph.add_node(latents, "latents")

        base_scheduler = SchedulerNode(scheduler_index=self.base_scheduler)
        node_graph.add_node(base_scheduler, "base_scheduler")

        steps = NumberNode(number=self.steps)
        node_graph.add_node(steps, "steps")

        guidance_scale = NumberNode(number=self.guidance)
        node_graph.add_node(guidance_scale, "guidance")

        image_generation = ImageGenerationNode()
        image_generation.connect("scheduler", base_scheduler, "scheduler")
        image_generation.connect("num_inference_steps", steps, "value")
        image_generation.connect("latents", latents, "latents")
        image_generation.connect(
            "pooled_prompt_embeds", prompts_encoder, "pooled_prompt_embeds"
        )
        image_generation.connect("width", image_width, "value")
        image_generation.connect("height", image_height, "value")
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
        node_graph.add_node(image_generation, "image_generation")

        decoder = LatentsDecoderNode()
        decoder.connect("vae", vae_model, "vae")
        decoder.connect("latents", image_generation, "latents")
        node_graph.add_node(decoder, "decoder")

        image_send = ImageSendNode()
        image_send.connect("image", decoder, "image")
        node_graph.add_node(image_send, "image_send")

        return node_graph
