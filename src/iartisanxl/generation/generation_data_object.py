import attr
import copy

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
