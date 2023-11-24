import attr
import copy
import json

from iartisanxl.generation.lora_data_object import LoraDataObject
from iartisanxl.generation.controlnet_data_object import ControlNetDataObject
from iartisanxl.generation.model_data_object import ModelDataObject
from iartisanxl.generation.vae_data_object import VaeDataObject


@attr.s(auto_attribs=True, slots=True)
class ImageGenData:
    module: str = "texttoimage"
    seed: int = 0
    image_width: int = 1024
    image_height: int = 1024
    steps: int = 20
    guidance: float = 7.5
    base_scheduler: int = 0
    lora_scale: float = 1.0
    loras: list[LoraDataObject] = attr.Factory(list)
    controlnets: list[ControlNetDataObject] = attr.Factory(list)
    model: ModelDataObject = attr.Factory(ModelDataObject)
    vae: VaeDataObject = attr.Factory(VaeDataObject)
    positive_prompt_clipl: str = ""
    positive_prompt_clipg: str = ""
    negative_prompt_clipl: str = ""
    negative_prompt_clipg: str = ""
    clip_skip: int = 0

    def add_lora(self, lora: LoraDataObject):
        if any(existing_lora.filename == lora.filename for existing_lora in self.loras):
            raise ValueError(f"A LoRA with filename {lora.filename} already exists.")
        else:
            self.loras.append(lora)

    def remove_lora(self, lora_to_remove: LoraDataObject):
        for index, lora in enumerate(self.loras):
            if lora == lora_to_remove:
                del self.loras[index]
                break

    def change_lora_enabled(self, lora_item: LoraDataObject, enabled: bool):
        for lora in self.loras:
            if lora == lora_item:
                lora.enabled = enabled
                break

    def add_controlnet(self, controlnet: ControlNetDataObject):
        if any(
            existing_controlnet.controlnet_id == controlnet.controlnet_id
            for existing_controlnet in self.controlnets
        ):
            raise ValueError(
                f"A ControlNet with an ID {controlnet.controlnet_id} already exists."
            )
        else:
            self.controlnets.append(controlnet)

    def remove_controlnet(self, controlnet_to_remove: ControlNetDataObject):
        for index, controlnet in enumerate(self.controlnets):
            if controlnet == controlnet_to_remove:
                del self.controlnets[index]
                break

    def change_controlnet_enabled(
        self, controlnet_item: ControlNetDataObject, enabled: bool
    ):
        for controlnet in self.controlnets:
            if controlnet == controlnet_item:
                controlnet.enabled = enabled
                break

    def copy(self):
        loras_copy = [lora.copy() for lora in self.loras]
        controlnets_copy = [controlnet.copy() for controlnet in self.controlnets]
        model_copy = self.model.copy()

        new_obj = attr.evolve(
            self, loras=loras_copy, controlnets=controlnets_copy, model=model_copy
        )
        return new_obj

    def deep_copy(self):
        new_obj = copy.deepcopy(self)
        return new_obj

    def update_attributes(self, data: dict):
        error = None
        for key, value in data.items():
            if key == "loras":
                self.loras.clear()
                for lora_data in value:
                    try:
                        lora = LoraDataObject(**lora_data)
                        self.add_lora(lora)
                    except TypeError:
                        error = "LoRA information is not compatible."
            elif key == "controlnets":
                self.controlnets.clear()
                for controlnet_data in value:
                    try:
                        controlnet = ControlNetDataObject(**controlnet_data)
                        self.add_controlnet(controlnet)
                    except TypeError:
                        error = "ControlNet information is not compatible."
            elif key == "model":
                try:
                    model = ModelDataObject(**value)
                    self.model = model
                except TypeError:
                    self.model = None
                    error = "Model information is not compatible."
            elif key == "vae":
                try:
                    vae = VaeDataObject(**value)
                except TypeError:
                    vae = VaeDataObject(name="Model default", path="")
                    error = "Vae information is not compatible, using model default."
                self.vae = vae
            else:
                if hasattr(self, key):
                    setattr(self, key, value)

        return error

    @classmethod
    def from_dict(cls, data: dict):
        loras = [LoraDataObject(**lora_data) for lora_data in data.get("loras", [])]

        try:
            controlnets = [
                ControlNetDataObject(**controlnet_data)
                for controlnet_data in data.get("controlnets", [])
            ]
        except TypeError:
            controlnets = []

        model = ModelDataObject(**data.get("model", {}))
        vae = VaeDataObject(**data.get("vae", {}))

        new_obj = cls(
            module=data.get("module", "texttoimage"),
            seed=data.get("seed", 0),
            image_width=data.get("image_width", 1024),
            image_height=data.get("image_height", 1024),
            steps=data.get("steps", 20),
            guidance=data.get("guidance", 7.5),
            base_scheduler=data.get("base_scheduler", 0),
            lora_scale=data.get("lora_scale", 1.0),
            loras=loras,
            controlnets=controlnets,
            model=model,
            vae=vae,
            positive_prompt_clipl=data.get("positive_prompt_clipl", ""),
            positive_prompt_clipg=data.get("positive_prompt_clipg", ""),
            negative_prompt_clipl=data.get("negative_prompt_clipl", ""),
            negative_prompt_clipg=data.get("negative_prompt_clipg", ""),
            clip_skip=data.get("clip_skip", 0),
        )
        return new_obj

    def to_json_graph(self):
        node_attributes = [
            (
                "StableDiffusionXLModelNode",
                0,
                {"path": self.model.path, "name": "sdxl_model"},
            ),
            ("TextNode", 1, {"text": self.positive_prompt_clipg}),
            ("PromptsEncoderNode", 2, {}),
            ("VaeModelNode", 3, {"path": self.vae.path, "name": "vae_model"}),
            ("NumberNode", 4, {"number": self.seed}),
            ("NumberNode", 5, {"number": self.image_height}),
            ("NumberNode", 6, {"number": self.image_width}),
            ("LatentsNode", 7, {}),
            ("SchedulerNode", 8, {"scheduler_index": self.base_scheduler}),
            ("NumberNode", 9, {"number": self.steps}),
            ("NumberNode", 10, {"number": self.guidance}),
            (
                "ImageGenerationNode",
                11,
                {
                    "abort": "<lambda>",
                    "callback": "step_progress_update",
                    "name": "image_generation",
                },
            ),
            ("LatentsDecoderNode", 12, {}),
            ("ImageSendNode", 13, {"image_callback": "preview_image"}),
        ]
        nodes = [
            {"class": class_name, "id": id, **attributes}
            for class_name, id, attributes in node_attributes
        ]

        connection_attributes = [
            (0, "tokenizer_1", 2, "tokenizer_1"),
            (0, "tokenizer_2", 2, "tokenizer_2"),
            (0, "text_encoder_1", 2, "text_encoder_1"),
            (0, "text_encoder_2", 2, "text_encoder_2"),
            (1, "value", 2, "prompt_1"),
            (4, "value", 7, "seed"),
            (0, "num_channels_latents", 7, "num_channels_latents"),
            (5, "value", 7, "height"),
            (6, "value", 7, "width"),
            (3, "vae_scale_factor", 7, "vae_scale_factor"),
            (8, "scheduler", 11, "scheduler"),
            (9, "value", 11, "num_inference_steps"),
            (7, "latents", 11, "latents"),
            (2, "pooled_prompt_embeds", 11, "pooled_prompt_embeds"),
            (5, "value", 11, "height"),
            (6, "value", 11, "width"),
            (2, "prompt_embeds", 11, "prompt_embeds"),
            (10, "value", 11, "guidance_scale"),
            (2, "negative_prompt_embeds", 11, "negative_prompt_embeds"),
            (2, "negative_pooled_prompt_embeds", 11, "negative_pooled_prompt_embeds"),
            (0, "unet", 11, "unet"),
            (7, "generator", 11, "generator"),
            (3, "vae", 12, "vae"),
            (11, "latents", 12, "latents"),
            (12, "image", 13, "image"),
        ]

        connections = [
            {
                "from_node_id": from_id,
                "from_output_name": from_name,
                "to_node_id": to_id,
                "to_input_name": to_name,
            }
            for from_id, from_name, to_id, to_name in connection_attributes
        ]

        data = {"nodes": nodes, "connections": connections}

        json_graph = json.dumps(data)
        return json_graph
